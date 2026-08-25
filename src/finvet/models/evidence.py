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

from math import isfinite
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, Field

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
    period_end: Optional[str] = None
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


def _observation_from_items(record: ToolExecutionRecord, metric: str,
                            ) -> Optional[TrustedObservation]:
    """Resolve a SEC line item by XBRL concept."""
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
        return TrustedObservation(
            tool=record.tool,
            metric=metric,
            value=value,
            units=item.get("units"),
            period_end=item.get("period_end") or record.payload.get("period_end"),
            concept=name,
            source_id=record.payload.get("filing_accession"),
        )
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
    """Resolve a market metric from a named field."""
    field = _MARKET_FIELD_FOR_METRIC.get(metric)
    if not field:
        return None

    value = _coerce_number(record.payload.get(field))
    if value is None:
        return None
    return TrustedObservation(
        tool=record.tool,
        metric=metric,
        value=value,
        period_end=record.payload.get("latest_trading_day"),
        source_id=record.payload.get("symbol"),
    )


def resolve_trusted_observation(
    parsed_claim: Any,
    records: Sequence[ToolExecutionRecord],
    expected_period_end: Optional[str] = None,
) -> Optional[TrustedObservation]:
    """The one number a numeric verdict may rest on, or None.

    Returns None -- and the caller must then fail closed -- when the claim
    names no metric, when no successful call produced a matching field, or when
    the value found covers a different period than the one resolved upstream.
    That last check matters: a correct figure from the wrong fiscal year is not
    weak evidence, it is the wrong evidence, and the comparator cannot tell.
    """
    metric = getattr(parsed_claim, "metric", None)
    if not metric:
        # A claim with no canonical metric (narrative claims parse this way)
        # has nothing to resolve against. The agent's prose may still discuss a
        # number; it does not become a trusted observation.
        return None

    for record in records:
        if not record.trusted_success:
            continue

        observation = (_observation_from_items(record, metric)
                       or _observation_from_field(record, metric)
                       or _observation_from_macro(record, metric))
        if observation is None:
            continue

        if (expected_period_end and observation.period_end
                and observation.period_end != expected_period_end):
            continue

        return observation

    return None
