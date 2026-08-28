"""The boundary between what a tool returned and what Python is willing to compare.

The deterministic override is only as trustworthy as the number handed to it,
and two defects sat underneath it:

1. Every SEC and Market tool catches its own exceptions and *returns* a result
   with ``success=False`` instead of raising. LangChain therefore reports the
   call's transport status as success, so a failed call was recorded as a
   successful one -- and a parse error whose message echoed the payload it
   choked on could be scraped for a financial-looking number.
2. The retrieved-value fallback regex-matched that number out of the
   *serialized, truncated* result string, so which value reached the comparator
   depended on where a 3000-character preview happened to be cut.

`ToolExecutionRecord` separates the two notions of success that were conflated,
and `resolve_trusted_observation` reads structured fields instead of text. A
numeric verdict may rest only on a `TrustedObservation`; anything else fails
closed to NOT_ENOUGH_INFO.
"""

from datetime import date
from math import isfinite
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, Field

from ..config.constants import CORROBORATION_METRICS
from ..config.metrics import METRIC_TO_CONCEPTS

# Market and macro tools return flat result models rather than line items, so
# their metrics map to a field name instead of an XBRL concept. Explicit by
# design: a metric absent from this map yields no observation, which fails
# closed rather than guessing which of several numbers on the result was meant.
_MARKET_FIELD_FOR_METRIC = {
    "closing_price": "price",
    "opening_price": "open",
    "intraday_high": "high",
    "intraday_low": "low",
    "price_change_absolute": "change",
    "price_change_percent": "change_percent",
    "volume": "volume",
    "market_cap": "market_cap",
    "pe_ratio": "pe_ratio",
    "dividend_yield": "dividend_yield",
    "52_week_high": "fifty_two_week_high",
    "52_week_low": "fifty_two_week_low",
}


class ToolExecutionRecord(BaseModel):
    """One tool call, with the two kinds of success kept apart.

    `transport_success` is LangChain's view: did the call itself complete.
    `application_success` is the tool's own view: did it retrieve anything.
    A tool that caught an exception and returned an error payload is a
    transport success and an application failure, and only the second answers
    "may I trust a number from this."

    `application_success` is None when the payload carries no ``success`` field
    -- an unknown shape, which `trusted_success` treats as untrusted rather
    than assuming the best.
    """

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    transport_success: bool = True
    application_success: Optional[bool] = None
    result_preview: str = ""

    def to_detail(self) -> Dict[str, Any]:
        """The audit/API view of this call.

        Deliberately not `model_dump()`: the response schema and the Streamlit
        evidence panel read `tool`/`args`/`result`/`success`, and the payload
        and two status flags are internal. Producing it here keeps the record
        the single source for both views, so they cannot drift.
        """
        return {
            "tool": self.tool,
            "args": self.args,
            "result": self.result_preview,
            "success": self.call_succeeded,
        }

    @property
    def call_succeeded(self) -> bool:
        """Did the call fail? Used for display and the audit trail.

        Lenient on purpose: several tools legitimately return a plain string or
        a shape with no ``success`` field, and showing those as FAILED would be
        wrong. Only an explicit application-level failure counts.
        """
        return self.transport_success and self.application_success is not False

    @property
    def trusted_success(self) -> bool:
        """May a number from this call reach the comparator?

        Strict on purpose: an unrecognised payload is not evidence. This is the
        question the deterministic override asks, and the only safe default is
        no.
        """
        return self.transport_success and self.application_success is True


class TrustedObservation(BaseModel):
    """A number Python is willing to compare against a claim.

    Every field records where the value came from, so a reviewer can find it in
    the filing rather than take the pipeline's word for it.
    """

    tool: str
    metric: str
    value: float
    units: Optional[str] = None
    # When the value's reporting period closed. Fiscal facts have one.
    period_end: Optional[str] = None
    # When the source observed the value. A current quote has one of these and
    # no period_end: it is a point in time, not a period that closed. Keeping
    # them apart stops a trading day being filed as though it ended a fiscal
    # quarter.
    observed_at: Optional[str] = None
    concept: Optional[str] = None
    source_id: Optional[str] = None


def _coerce_number(raw: Any) -> Optional[float]:
    """A finite, non-boolean number, or None.

    bool is a subclass of int in Python, so ``isinstance(True, (int, float))``
    is True and an unguarded cast turns a flag into the value 1.0.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if isfinite(value) else None


def tool_record_from_result(tool: str, args: Dict[str, Any],
                            result: Any) -> ToolExecutionRecord:
    """Build a record from a tool's return value.

    Used by tests and by any caller holding a tool result directly; the agent
    builds records from ToolMessages instead, where the transport status is
    also available.
    """
    payload = result if isinstance(result, dict) else {}
    if not payload and hasattr(result, "model_dump"):
        payload = result.model_dump()

    success = payload.get("success")
    return ToolExecutionRecord(
        tool=tool,
        args=args or {},
        payload=payload,
        transport_success=True,
        application_success=success if isinstance(success, bool) else None,
        result_preview=str(result)[:1000],
    )


def _period_matches(period_end: Optional[str],
                    expected_start: Optional[str],
                    expected_end: Optional[str]) -> bool:
    """Whether a fact's period is the one the claim is about.

    One definition, applied while candidates are being chosen rather than
    after. A window when the resolved period has a start -- the resolver only
    approximates an issuer's fiscal calendar -- and exact equality when it
    does not.
    """
    if not expected_end:
        return True
    if not period_end:
        return False
    if expected_start:
        return expected_start <= period_end <= expected_end
    return period_end == expected_end


def _observation_from_items(record: ToolExecutionRecord, metric: str,
                            expected_start: Optional[str] = None,
                            expected_end: Optional[str] = None,
                            ) -> Optional[TrustedObservation]:
    """Resolve a SEC line item by XBRL concept, from the claim's own period.

    The period is checked here, inside the candidate loop, not afterwards. A
    filing returns one context per concept and they are not all from its own
    year -- Nvidia's FY2025 10-K carries
    RevenueFromContractWithCustomerExcludingAssessedTax at 2017-01-29 next to
    Revenues at 2025-01-26. Returning the first concept in preference order and
    period-checking that one rejected the stale fact and never looked at the
    correct figure sitting beside it.
    """
    concepts = METRIC_TO_CONCEPTS.get(metric)
    if not concepts:
        return None

    for item in record.payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        # line_item carries the XBRL concept name (sec_edgar.py builds it from
        # the concept), and `concept` is populated alongside it.
        name = item.get("line_item") or item.get("concept")
        if name not in concepts:
            continue
        value = _coerce_number(item.get("value"))
        if value is None:
            continue
        # An explicitly unconsolidated fact is a segment or subsidiary figure,
        # not the entity-wide number the claim asks about. Absent means the
        # source did not say, which is the common case for XBRL concepts that
        # are entity-wide by definition.
        if item.get("consolidated") is False:
            continue
        period_end = item.get("period_end") or record.payload.get("period_end")
        if not _period_matches(period_end, expected_start, expected_end):
            continue
        return TrustedObservation(
            tool=record.tool,
            metric=metric,
            value=value,
            units=item.get("units"),
            period_end=period_end,
            concept=name,
            source_id=record.payload.get("filing_accession"),
        )
    return None


# Market tools whose results carry an observation time from the source.
# get_company_overview is deliberately absent: market cap, P/E, dividend yield
# and the 52-week range have no timestamp in the producer contract, so nothing
# can say when they were true. See RELEASE_A_DECISIONS.md, D10.
_OBSERVATION_TIMED_TOOLS = frozenset({"get_stock_quote"})


def _iso_date_or_none(raw: Any) -> Optional[str]:
    """A calendar date in YYYY-MM-DD, or None.

    Rejects anything else rather than passing it along: a malformed date is
    not a weaker observation time, it is the absence of one.
    """
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _observation_from_macro(record: ToolExecutionRecord, metric: str,
                            ) -> Optional[TrustedObservation]:
    """Resolve a macro indicator, which names its own metric on the result."""
    if record.payload.get("metric") != metric:
        return None
    value = _coerce_number(record.payload.get("value"))
    if value is None:
        return None
    return TrustedObservation(
        tool=record.tool,
        metric=metric,
        value=value,
        units=record.payload.get("units"),
        period_end=record.payload.get("observation_date"),
        source_id=record.payload.get("series_id"),
    )


def _observation_from_field(record: ToolExecutionRecord, metric: str,
                            ) -> Optional[TrustedObservation]:
    """Resolve a market metric from a named field.

    Only `get_stock_quote` qualifies. `_MARKET_FIELD_FOR_METRIC` spans two
    producers: the quote endpoint, which timestamps what it returns, and the
    company-overview endpoint, which does not. Keying on payload shape alone
    let a market-cap figure with no observation time become the number a
    verdict rests on. Which tool answered is the thing that decides.
    """
    if record.tool not in _OBSERVATION_TIMED_TOOLS:
        return None

    field = _MARKET_FIELD_FOR_METRIC.get(metric)
    if not field:
        return None

    value = _coerce_number(record.payload.get(field))
    if value is None:
        return None

    # Re-validated here even though the producer validates it. A boundary that
    # assumes its producer is correct is not a boundary -- and this one exists
    # because the producer was, for a while, substituting its own clock.
    observed_at = _iso_date_or_none(record.payload.get("latest_trading_day"))
    if observed_at is None:
        return None

    return TrustedObservation(
        tool=record.tool,
        metric=metric,
        value=value,
        # A quote is observed, not reported: period_end stays empty.
        observed_at=observed_at,
        source_id=record.payload.get("symbol"),
    )


# Tools whose results are attributable *text*, never structured facts. A
# passage can show what a company said; turning it into a comparable number
# means reading prose, which is exactly what the deterministic layer exists to
# avoid doing.
SUPPORTING_EVIDENCE_TOOLS = frozenset({"search_filing_text"})


# Tools that retrieve source material about the world, and the field each uses
# to report how much it found. `search_past_verifications` is deliberately
# absent: a prior verification is this system's own earlier output, and
# counting it as evidence would let a verdict cite itself.
RETRIEVAL_COUNT_FIELDS = {
    "search_filing_text": "total_found",
    "search_financial_news": "count",
}


def qualitative_evidence_gap(
    records: Sequence[ToolExecutionRecord],
) -> Optional[str]:
    """Why a claim naming no value has nothing behind it, or None if it does.

    A claim with no number produces no trusted observation, so the numeric
    guard in `_apply_override` never applies to it and the model's verdict is
    released untouched. That is how a News agent's four successful searches,
    every one returning zero articles, became REFUTES at 0.95: silence read as
    proof.

    The distinctions are kept rather than collapsed to a boolean because they
    have opposite remedies -- a source that was searched and said nothing is a
    fact about the record, while a source that could not be reached is a fact
    about the run, and only the reader can decide whether to re-run.
    """
    searched = False
    unavailable = False
    no_corpus = False

    for record in records:
        count_field = RETRIEVAL_COUNT_FIELDS.get(record.tool)
        if count_field is None:
            continue

        if not record.trusted_success:
            unavailable = True
            continue

        searched = True
        count = record.payload.get(count_field)
        if isinstance(count, int) and count > 0:
            # One productive search is evidence. Repeated empty ones are not
            # cumulatively less empty.
            return None
        if record.payload.get("reason") == "no_corpus":
            no_corpus = True

    if not searched:
        return "retrieval_unavailable" if unavailable else "no_retrieval_attempted"
    if no_corpus:
        # There was no document to be silent.
        return "no_indexed_source"
    return "no_supporting_evidence"


def qualitative_decline_reason(
    claimed_value: Optional[float],
    verdict: str,
    evidence_gap: Optional[str],
) -> Optional[str]:
    """Why a verdict on a valueless claim may not stand, or None if it may.

    One function rather than a condition written twice: `_apply_override`
    decides the verdict and `execute` records the limitation, and if those two
    drifted the response would either escalate a declared decline or declare a
    verdict that stood. Both call this.
    """
    if claimed_value is not None:
        # Governed by the trusted observation instead. The two guards must not
        # start overlapping.
        return None
    if evidence_gap is not None and verdict in ("SUPPORTS", "REFUTES"):
        return evidence_gap
    if verdict == "REFUTES":
        return "non_corroboration_is_not_contradiction"
    return None


# No exchange rate between major currencies spans an order of magnitude, so a
# gap that wide is decidable without knowing one. `ParsedClaim` carries no
# currency, so anything narrower than this against a non-USD filing figure
# would be comparing two different units and calling it a verdict.
FX_SAFE_MAGNITUDE_RATIO = 10.0


def _penalty_observation(
    parsed_claim: Any,
    records: Sequence[ToolExecutionRecord],
    metric: str,
    claim_text: Optional[str] = None,
) -> Optional[TrustedObservation]:
    """A fine or settlement Python lifted from the filing that discloses it.

    One observation or none. Several passages may quote the same penalty --
    the disclosure repeats verbatim across a 10-K and its 10-Qs -- so equal
    values are one fact, while genuinely different values are the case where
    choosing between them would decide a verdict by coin toss.
    """
    from ..tools.filing_amounts import currency_in_text, extract_amount

    found = []
    for record in records:
        if record.tool not in RETRIEVAL_COUNT_FIELDS or not record.trusted_success:
            continue
        for chunk in record.payload.get("chunks") or []:
            amount = extract_amount(chunk)
            if amount is not None:
                found.append(amount)

    if not found:
        return None
    if len({a.value for a in found}) != 1 or len({a.currency for a in found}) != 1:
        return None

    amount = found[0]
    claimed = getattr(parsed_claim, "value", None)
    # The claim states its own currency more often than not -- "fined 500
    # million euros" -- and reading it is cheaper and more honest than
    # assuming one. When it matches the filing there is no rate to guess.
    if claimed and currency_in_text(claim_text) != amount.currency:
        ratio = max(abs(claimed), amount.value) / max(min(abs(claimed), amount.value), 1e-9)
        if ratio < FX_SAFE_MAGNITUDE_RATIO:
            # Close enough that the answer would turn on an exchange rate this
            # release does not have. Declining is the honest outcome.
            return None

    return TrustedObservation(
        tool="search_filing_text",
        metric=metric,
        value=amount.value,
        units=amount.currency,
        period_end=None,
        concept=None,
        source_id=amount.evidence_id,
    )


def resolve_trusted_observation(
    parsed_claim: Any,
    records: Sequence[ToolExecutionRecord],
    expected_period_end: Optional[str] = None,
    expected_period_start: Optional[str] = None,
    narrative_metric: Optional[str] = None,
    claim_text: Optional[str] = None,
) -> Optional[TrustedObservation]:
    """The one number a numeric verdict may rest on, or None.

    Returns None -- and the caller must then fail closed -- when the claim
    names no metric, when no successful call produced a matching field, or when
    the value found covers a different period than the one resolved upstream.
    That last check matters: a correct figure from the wrong fiscal year is not
    weak evidence, it is the wrong evidence, and the comparator cannot tell.
    """
    # Penalties have no XBRL concept -- a fine is not a financial-statement
    # line item -- so the loop below can never resolve one, and every such
    # claim failed closed no matter what the filing said. The filing is the
    # authoritative record of the disclosure; what could not be trusted was a
    # *model* reading a number out of it. `extract_amount` has Python read it
    # instead, and declines wherever that would be a guess.
    penalty_metric = narrative_metric or getattr(parsed_claim, "metric", None)
    if penalty_metric in CORROBORATION_METRICS:
        return _penalty_observation(parsed_claim, records, penalty_metric,
                                    claim_text)

    metric = getattr(parsed_claim, "metric", None)
    if not metric:
        # A claim with no canonical metric (narrative claims parse this way)
        # has nothing to resolve against. The agent's prose may still discuss a
        # number; it does not become a trusted observation.
        return None

    for record in records:
        if not record.trusted_success:
            continue

        # Filing prose may not reach the resolvers below. They key on payload
        # shape rather than on which tool produced it, so a retrieval result
        # that happened to carry an `items` list would otherwise be trusted
        # like an XBRL fact.
        #
        # The one way prose does yield a number is `_penalty_observation`
        # above, and it is not this: there Python extracts the amount itself,
        # from the section that discloses penalties, only when exactly one
        # candidate is present. Nothing here reads prose.
        if record.tool in SUPPORTING_EVIDENCE_TOOLS:
            continue

        observation = (
            _observation_from_items(record, metric,
                                    expected_period_start, expected_period_end)
            or _observation_from_field(record, metric)
            or _observation_from_macro(record, metric))
        if observation is None:
            continue

        # A period-bound claim needs a fact that names its period. The guard
        # used to read "expected and observed and differ", so a fact carrying
        # no period at all passed -- it only rejected evidence already
        # labelled well enough to be checked.
        #
        # When the resolved period has a start, membership decides: does this
        # fact's period end inside the window the claim is about? Equality
        # cannot answer that. `period_resolver` runs before any agent has
        # asked SEC where the issuer's year ends, so "fiscal year 2024"
        # becomes the calendar window 2024-01-01..2024-12-31 -- and Apple's
        # fiscal 2024 ended 2024-09-28, Microsoft's 2024 on 2024-06-30. Exact
        # comparison rejected the correct filing for every issuer whose year
        # is not the calendar's, which is most of them.
        #
        # The window still rejects the wrong year and the wrong quarter; it
        # only stops requiring the pipeline to have guessed the issuer's
        # calendar correctly in advance.
        if not _period_matches(observation.period_end,
                               expected_period_start, expected_period_end):
            continue

        return observation

    return None
