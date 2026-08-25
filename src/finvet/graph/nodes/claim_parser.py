"""Claim parser node — emits the fine-tuned claim parser's 7-field contract:

    claim_type | ticker | metric | operator | value | period | reject_reason

metric is resolved against the vendored whitelist (fail closed to null, in
which case the agent infers from claim text as before). operator carries the
seven comparators including approx and range. Raw model output crosses
normalize_parser_output — the one boundary where it becomes a trusted object.
"""

import json
import re
from pathlib import Path
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from ...config.metrics import METRIC_HARD_DROPS, METRIC_REMAPS, METRIC_WHITELIST
from ...models.state import VerificationState
from ...models.claim import ParsedClaim
from ...llm import create_llm
from ...audit import get_audit_logger
from ...utils.exceptions import ParsingError
from ...utils.logging import get_logger

logger = get_logger(__name__)


# The prompt lives with the other system prompts and is rendered at load
# time: the metric whitelist is injected from config/metrics.py, never
# hand-copied, so the prompt and the validator cannot drift apart. A copied
# list would eventually tell the model about metrics the resolver rejects,
# surfacing as a mysterious residual rate in the claim_parsed audit events.
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "agents" / "prompts" / "parser_system.txt"
)


def _render_metric_blocks() -> str:
    """The three whitelist blocks, grouped by claim_type as the scoping rule."""
    lines = []
    for claim_type in ("sec", "market", "news"):
        metrics = sorted(METRIC_WHITELIST[claim_type])
        lines.append(f'   ### claim_type == "{claim_type}"  ({len(metrics)})')
        for i in range(0, len(metrics), 4):
            lines.append("   " + "  ".join(metrics[i:i + 4]))
        lines.append("")
    lines.append('   ### claim_type == "reject": metric must be null.')
    return "\n".join(lines)


PARSER_SYSTEM_PROMPT = _PROMPT_PATH.read_text().replace(
    "__METRIC_WHITELIST__", _render_metric_blocks()
)

# Words that cannot be the start of a company name even when capitalised.
_NAME_SKIP = {
    'the', 'a', 'an', 'total', 'its', 'their', 'this', 'that', 'these',
    'those', 'some', 'all', 'both', 'each', 'every',
}
# Words excluded from company-name matching: legal suffixes AND common industry
# descriptors that appear in many company names and would produce false matches
# (e.g. "Health" in both "Coventry Health Care" and "CVS Health").
_CORP_GENERIC = {
    'inc', 'corp', 'ltd', 'plc', 'co', 'group', 'holdings', 'llc', 'lp',
    'health', 'care', 'financial', 'services', 'technologies', 'technology',
    'solutions', 'systems', 'industries', 'energy', 'capital', 'management',
    'enterprises', 'international', 'national', 'american', 'global',
    'company', 'companies',
}


def _extract_company_hint(claim_text: str) -> str:
    """Extract the longest consecutive run of capitalised tokens from claim text.

    Skips determiners and other non-name openers so "The total market cap of
    Broadcom exceeded..." yields "Broadcom" rather than "The".
    """
    tokens = claim_text.split()
    sequences: list[str] = []
    current: list[str] = []
    for token in tokens:
        clean = re.sub(r"[^a-zA-Z&']", "", token)
        if not clean:
            if current:
                sequences.append(" ".join(current))
                current = []
            continue
        if clean[0].isupper() and clean.lower() not in _NAME_SKIP:
            current.append(clean)
        else:
            if current:
                sequences.append(" ".join(current))
                current = []
    if current:
        sequences.append(" ".join(current))
    return max(sequences, key=len) if sequences else claim_text[:40]


def _ticker_matches_claim(company_name: str, claim_text: str) -> bool:
    """True when the Finnhub company name has at least one substantive word
    in common with the claim text (exact token match, not prefix/substring)."""
    name_words = {
        w.lower()
        for w in re.split(r"[\s\-&]+", company_name)
        if len(w) > 2 and w.lower() not in _CORP_GENERIC
    }
    claim_words = {
        w.lower()
        for w in re.split(r"[\s\-&]+", claim_text)
        if len(w) > 2
    }
    return bool(name_words & claim_words)


def _normalize_ticker(ticker: str, claim_text: str) -> str:
    """Validate a market ticker against Finnhub and correct stale/wrong tickers.

    Strategy:
    1. Fetch the company name for the parsed ticker via Finnhub profile2.
    2. If the ticker is unknown (delisted, typo) or the returned company name
       shares no substantive words with the claim, search Finnhub by company
       name extracted from the claim text and use the top US equity result.
    3. Falls back to the original ticker on any API error.
    """
    try:
        from ...mcp.finnhub import FinnhubClient
        client = FinnhubClient()
        if client.mock_mode:
            return ticker

        company_name = client.get_company_name(ticker)
        if company_name and _ticker_matches_claim(company_name, claim_text):
            return ticker

        hint = _extract_company_hint(claim_text)
        if not hint:
            return ticker

        found = client.search_ticker(hint)
        if found and found != ticker:
            logger.info(f"Ticker normalized: {ticker} → {found} (hint: '{hint}')")
            return found

        return ticker
    except Exception as e:
        logger.warning(f"Ticker normalization skipped for {ticker}: {e}")
        return ticker


def reconcile_reject_fields(parsed_data: Dict) -> Dict:
    """Make claim_type and reject_reason agree before ParsedClaim validates them.

    The model breaks the pairing on a small fraction of claims: it recognises a
    claim is unverifiable, sets reject_reason, and leaves claim_type as "sec".
    ParsedClaim forbids that combination, so the request died with an HTTP 500
    carrying a Pydantic stack trace — on a claim the model had judged correctly.

    reject_reason is only ever populated when rejecting, so the intent is
    unambiguous and worth honouring rather than crashing on. The mirror case, a
    reject with no reason given, is filled with "unspecified": defaulting to
    "incomplete" would tell the user the claim was missing a ticker or value,
    which may simply be untrue.

    Returns a new dict; the caller keeps the model's raw output intact.
    """
    data = dict(parsed_data)
    claim_type = data.get("claim_type")
    reject_reason = data.get("reject_reason")

    if reject_reason is not None and claim_type != "reject":
        logger.info(
            f"Parser set reject_reason='{reject_reason}' on claim_type="
            f"'{claim_type}'; treating as a reject"
        )
        data["claim_type"] = "reject"
    elif claim_type == "reject" and reject_reason is None:
        logger.info("Parser returned a reject with no reason; recording 'unspecified'")
        data["reject_reason"] = "unspecified"

    # §8 reject contract: a reject carries nothing but its reason. The model
    # often leaves the fields it extracted before deciding to reject; nulling
    # them here is what makes the strict model invariant safe to enforce.
    if data.get("claim_type") == "reject":
        for field in ("ticker", "metric", "operator", "value", "period"):
            data[field] = None

    return data


# Metric resolution decision codes, surfaced to the audit trail. A rising
# "residual" rate in production means the prompt and the vocabulary have
# drifted apart.
_METRIC_DECISIONS = (
    "absent", "whitelist", "remap", "normalized", "hard_drop",
    "residual", "reject_null",
)


def resolve_metric_field(data: Dict, claim_text: str) -> tuple:
    """Resolve the raw metric against the vendored vocabulary. Fail closed.

    First hit wins: exact whitelist -> alias remap -> normalise and retry
    both -> hard-drop -> null. Whitelist MUST precede the drop list: the
    source vocabulary lists operating_margin in both, and only this ordering
    keeps it alive (pinned in test_metrics_vocab).

    A wrong metric selects the wrong XBRL concept and produces a confidently
    wrong verdict; a null metric reverts to the agent inferring from claim
    text, which is exactly the pre-migration behaviour. So every doubtful
    path yields null — the layer may only ever add information.

    Returns (new data dict, decision code).
    """
    data = dict(data)
    claim_type = data.get("claim_type")
    metric = data.get("metric")

    if claim_type == "reject":
        data["metric"] = None
        return data, "reject_null"
    if metric is None:
        return data, "absent"

    allowed = METRIC_WHITELIST.get(claim_type, frozenset())
    remaps = METRIC_REMAPS.get(claim_type, {})

    if metric in allowed:
        return data, "whitelist"
    if metric in remaps:
        data["metric"] = remaps[metric]
        return data, "remap"

    normalised = re.sub(r"[\s\-]+", "_", str(metric).strip().lower())
    normalised = re.sub(r"_(ratio|expense)$", "", normalised)
    if normalised in allowed:
        data["metric"] = normalised
        return data, "normalized"
    if normalised in remaps:
        data["metric"] = remaps[normalised]
        return data, "remap"

    if metric in METRIC_HARD_DROPS or normalised in METRIC_HARD_DROPS:
        logger.info(f"Metric '{metric}' is a known trap (segment/KPI/event); nulled")
        data["metric"] = None
        return data, "hard_drop"

    logger.info(f"Metric '{metric}' unresolvable for claim_type '{claim_type}'; nulled")
    data["metric"] = None
    return data, "residual"


def normalize_parser_output(raw: Dict, claim_text: str) -> tuple:
    """Turn raw model JSON into contract-valid data, in load-bearing order.

    Reconciliation runs FIRST because it can change claim_type to "reject",
    and claim_type scopes the metric whitelist — resolving the metric before
    reconciling would validate against the wrong class. Operator/value
    pairing runs last, on the settled fields.

    Returns (data, decisions) where decisions carries one code per concern
    for the claim_parsed audit event.
    """
    # Legacy keys (pre-CONTRACT schema, or a regressing model): comparison is
    # honoured as operator when operator itself is absent, currency dropped.
    # This runs BEFORE reconciliation so the §8 nulling of a reject's
    # companions is final — honouring afterwards would resurrect an operator
    # on a nulled reject.
    data = dict(raw)
    if data.get("operator") is None and data.get("comparison") is not None:
        data["operator"] = data["comparison"]
    data.pop("comparison", None)
    data.pop("currency", None)
    data = reconcile_reject_fields(data)
    if data.get("claim_type") != raw.get("claim_type"):
        reject_decision = "coerced_reject"
    elif data.get("reject_reason") != raw.get("reject_reason"):
        reject_decision = "filled_unspecified"
    else:
        reject_decision = "none"

    data, metric_decision = resolve_metric_field(data, claim_text)

    # operator non-null iff value non-null (§8). The prompt documents eq as
    # the default comparator, so a value with no operator is completed rather
    # than crashed on; an operator with no value anchors nothing and is
    # dropped.
    operator_decision = "none"
    op = data.get("operator")
    if data.get("claim_type") != "reject":
        if data.get("value") is not None and op is None:
            data["operator"] = "eq"
            operator_decision = "defaulted_eq"
        elif data.get("value") is None and op is not None:
            data["operator"] = None
            operator_decision = "dropped_operator_without_value"

    # Range bounds. A model that omits them leaves None -- the band is not
    # reconstructable from a midpoint, and guessing one would silently answer
    # a claim nobody stated. A range without both bounds is downgraded to
    # approx, which is honest about having only a point estimate; bounds on a
    # non-range operator are dropped rather than allowed to imply an interval.
    range_decision = "none"
    if data.get("claim_type") != "reject":
        has_bounds = (data.get("range_min") is not None
                      and data.get("range_max") is not None)
        if data.get("operator") == "range" and not has_bounds:
            data["operator"] = "approx"
            data["range_min"] = data["range_max"] = None
            range_decision = "range_without_bounds_downgraded_to_approx"
        elif data.get("operator") != "range" and (
                data.get("range_min") is not None
                or data.get("range_max") is not None):
            data["range_min"] = data["range_max"] = None
            range_decision = "dropped_bounds_without_range"
        elif has_bounds and data["range_min"] > data["range_max"]:
            data["range_min"], data["range_max"] = (
                data["range_max"], data["range_min"])
            range_decision = "swapped_inverted_bounds"

    return data, {"reject": reject_decision, "metric": metric_decision,
                  "operator": operator_decision, "range": range_decision}


def claim_parser(state: VerificationState) -> Dict:
    """
    Parse natural language claim into simplified 6-field structure.

    Uses DeepSeek to extract:
    - claim_type: Routing category
    - ticker: Company identifier
    - value: Numeric claim
    - period: Time reference
    - currency: Currency code
    - reject_reason: Why claim is invalid (if applicable)

    Args:
        state: Current verification state with claim_raw

    Returns:
        Dictionary with parsed_claim and parser metadata
    """
    # Support both claim_raw and claim_normalized for flexibility
    claim_text = state.get("claim_normalized") or state.get("claim_raw", "")
    request_id = state.get("request_id", "unknown")

    if not claim_text:
        raise ParsingError("No claim text provided")

    try:
        llm = create_llm("parser")

        messages = [
            SystemMessage(content=PARSER_SYSTEM_PROMPT),
            HumanMessage(content=f'Parse this claim:\n\n"{claim_text}"'),
        ]

        logger.info(f"Parsing claim (request: {request_id})")
        response = llm.invoke(messages)

        # Extract and clean response
        response_text = response.content
        if isinstance(response_text, list):
            response_text = "".join([
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in response_text
            ])

        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON
        parsed_data = json.loads(response_text)

        # One boundary between untrusted model output and the trusted claim:
        # reconcile the reject fields, resolve the metric against the
        # vocabulary (fail closed), and settle the operator/value pairing.
        normalized, decisions = normalize_parser_output(parsed_data, claim_text)
        parsed_claim = ParsedClaim(**normalized)

        # The parse is the pipeline's most consequential single decision and
        # previously left no audit record at all. This goes through
        # log_event() — the path that persists — NOT state["audit_events"],
        # which is write-only (nothing reads it; its event types have zero
        # rows in the database).
        get_audit_logger().log_event(
            event_type="claim_parsed",
            request_id=request_id,
            data={
                "fields": {
                    f: getattr(parsed_claim, f)
                    for f in ("claim_type", "ticker", "metric", "operator",
                              "value", "period", "reject_reason")
                },
                "raw": parsed_data,
                "decisions": decisions,
                "parser": "deepseek",
            },
        )

        # Validate ticker for market claims — catches stale/delisted tickers
        # (e.g. BRCM→AVGO, CVH→CVS, INTU→ISRG) that the fine-tuned parser
        # cannot know about as companies merge, rename, or delist over time.
        if parsed_claim.claim_type == "market" and parsed_claim.ticker:
            normalized = _normalize_ticker(parsed_claim.ticker, claim_text)
            if normalized != parsed_claim.ticker:
                parsed_claim = parsed_claim.model_copy(update={"ticker": normalized})

        # Calculate tokens used
        tokens_used = 0
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

        logger.info(
            f"Claim parsed: type={parsed_claim.claim_type}, "
            f"ticker={parsed_claim.ticker}, value={parsed_claim.value}, "
            f"operator={parsed_claim.operator} metric={parsed_claim.metric} "
            f"(request: {request_id})"
        )

        return {
            "parsed_claim": parsed_claim,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise ParsingError(f"Parser returned invalid JSON: {e}")

    except Exception as e:
        logger.error(f"Claim parsing failed: {e}")
        raise ParsingError(f"Failed to parse claim: {e}")
