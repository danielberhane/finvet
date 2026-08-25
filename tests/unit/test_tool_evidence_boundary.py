"""The boundary between tool execution and numeric verdicts.

The deterministic override is FinVet's headline guarantee, and it is only as
good as the number it is handed. Two defects sit under it:

1. Every SEC and Market tool catches its exceptions and *returns* a result with
   success=False rather than raising. LangChain therefore reports the call's
   transport status as success, and _extract_tool_info records the failed call
   as successful.
2. The retrieved-value fallback regex-scraped numbers out of the serialized
   result string, truncated to TOOL_RESULT_PREVIEW_CHARS -- so what the
   comparator received depended on where a string was cut. It now reads
   structured fields from ToolExecutionRecord instead.

These tests drive the real decorated tools. An earlier test in
test_base_agent.py hand-built ToolMessage(status="error"), a shape production
never emits, which is why it passed while the defect shipped.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent
from finvet.mcp.sec_edgar import FinancialItem
from finvet.models.evidence import resolve_trusted_observation
from finvet.tools.sec_tools import get_income_statement


class _Claim:
    """Minimal ParsedClaim stand-in for metric resolution."""
    metric = "revenue"
    value = 150_000_000_000.0
    operator = "eq"


def _records_for(result):
    """Run a tool result through the real extraction, as the agent does."""
    call = {"name": "get_income_statement", "args": {}, "id": "call_1"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=str(result), tool_call_id="call_1",
                    name="get_income_statement"),
    ]
    _, detail, _, records = BaseVerificationAgent._extract_tool_info(
        BaseVerificationAgent, messages)
    return detail, records


def _invoke_failing(message):
    with patch("finvet.tools.sec_tools._sec_client", None), \
         patch("finvet.tools.sec_tools._get_client",
               side_effect=Exception(message)):
        return get_income_statement.invoke(
            {"cik": "1", "accession_number": "acc", "period": "annual"})


def _invoke_succeeding(items):
    with patch("finvet.tools.sec_tools._get_client") as client:
        client.return_value.get_financials.return_value = items
        return get_income_statement.invoke(
            {"cik": "1", "accession_number": "acc", "period": "annual"})


# ---------------------------------------------------------------------------
# Application-level failure must not be recorded as a successful call
# ---------------------------------------------------------------------------

class TestApplicationFailureIsNotSuccess:

    @pytest.mark.parametrize("message", [
        "Connection refused to MCP server at localhost:9870",
        "1 validation error for FinancialsResult items.0.value Input should be "
        "a valid number [input_value=[{'line_item': 'Revenues', "
        "'value': 150000000000.0}]]",
    ])
    def test_caught_failure_is_recorded_as_failed(self, message):
        """The tool reports success=False; the audit trail must agree.

        This holds for every caught failure, not only ones a number can be
        scraped from -- a failed call recorded as successful corrupts the
        record regardless of what happens downstream.
        """
        result = _invoke_failing(message)
        assert result["success"] is False, "precondition: the tool caught the error"

        detail, _ = _records_for(result)
        assert detail[0]["success"] is False

    def test_successful_call_is_still_recorded_as_successful(self):
        """The guard must not swing the other way."""
        result = _invoke_succeeding([
            FinancialItem(line_item="Revenues", concept="Revenues",
                          value=391_035_000_000.0, units="USD",
                          period="annual", period_end="2024-09-28"),
        ])
        assert _records_for(result)[0][0]["success"] is True


# ---------------------------------------------------------------------------
# No number may be taken from a failed call
# ---------------------------------------------------------------------------

class TestNoValueFromFailedCall:

    def test_error_text_carrying_a_line_item_yields_no_value(self):
        """A parse error that echoes the payload it choked on puts a
        financial-looking dict into the error string. Scraping it produced
        150,000,000,000 from a call that retrieved nothing, which the override
        then promoted to SUPPORTS at 0.90.
        """
        result = _invoke_failing(
            "1 validation error for FinancialsResult items.0.value Input "
            "should be a valid number [input_value=[{'line_item': 'Revenues', "
            "'value': 150000000000.0}]]")

        _, records = _records_for(result)
        observation = resolve_trusted_observation(_Claim(), records)

        assert observation is None


# ---------------------------------------------------------------------------
# Evidence must not depend on where the preview string was cut
# ---------------------------------------------------------------------------

class TestEvidenceSurvivesTruncation:

    def test_value_is_found_beyond_the_preview_window(self):
        """TOOL_RESULT_PREVIEW_CHARS truncates the serialized result at 3000
        characters, and the fallback scrapes that truncated copy. A filing with
        enough line items pushes the relevant one past the cut, and the value
        silently disappears.
        """
        items = [
            FinancialItem(line_item=f"Filler{i}", concept="F", value=1234.0,
                          units="USD", period="annual", period_end="2024-09-28")
            for i in range(30)
        ]
        items.append(FinancialItem(
            line_item="Revenues", concept="Revenues", value=391_035_000_000.0,
            units="USD", period="annual", period_end="2024-09-28"))

        result = _invoke_succeeding(items)
        assert len(str(result)) > 3000, "precondition: result exceeds the preview"

        _, records = _records_for(result)
        observation = resolve_trusted_observation(_Claim(), records)

        assert observation is not None, "value lost past the 3000-char preview"
        assert observation.value == 391_035_000_000.0
        assert observation.concept == "Revenues"


class TestFailClosedScopeIsRecorded:
    """What failing closed costs, pinned so the cost stays visible.

    Task 2 makes a trusted observation the only numeric input, and a trusted
    observation can only come from a structured field: an XBRL line item, a
    market quote field, or a FRED series. All 35 metrics in SERVABLE_METRICS
    have one.

    The parser's whitelist is wider than SERVABLE_METRICS, and 39 of the 74
    metrics it may emit have no structured source at all. Numeric claims on
    those now return NOT_ENOUGH_INFO instead of a verdict derived from a number
    the model read out of prose. That is the intended direction -- but it is a
    real reduction in what the system will answer, and two of the affected
    metrics are the A2A corroboration metrics, which has a consequence recorded
    below.

    OPEN DECISION for the release gate: a claim whose metric has no structured
    source is not verifiable, and the settled split says the parser owns
    verifiability rejections. Rejecting these at parse time would be more
    honest than answering NOT_ENOUGH_INFO from the agent. Not done here --
    it is Task 3/Task 10 scope.
    """

    def test_every_servable_metric_still_has_a_structured_source(self):
        from finvet.config.metrics import METRIC_TO_CONCEPTS, SERVABLE_METRICS
        from finvet.mcp.fred import FRED_SERIES
        from finvet.models.evidence import _MARKET_FIELD_FOR_METRIC

        resolvable = (set(METRIC_TO_CONCEPTS) | set(_MARKET_FIELD_FOR_METRIC)
                      | set(FRED_SERIES))
        for kind in ("sec", "market", "news"):
            missing = SERVABLE_METRICS[kind] - resolvable
            assert not missing, f"{kind} metrics with no structured source: {missing}"

    def test_corroboration_metrics_have_no_structured_source(self):
        """Characterises the consequence rather than endorsing it.

        fine_amount and settlement_amount are the only metrics the News -> SEC
        delegation acts on, and neither can produce a trusted observation. The
        parent verdict is therefore always NOT_ENOUGH_INFO, and
        classify_status never returns CONTRADICTS unless *both* verdicts are
        decisive -- so the source_disagreement escalation cannot fire for the
        only claims that trigger the delegation.

        If this test starts failing, someone gave those metrics a structured
        source and the escalation is reachable again. That is the goal.
        """
        from finvet.config.constants import CORROBORATION_METRICS
        from finvet.config.metrics import METRIC_TO_CONCEPTS
        from finvet.mcp.fred import FRED_SERIES
        from finvet.models.a2a import A2A_CONTRADICTS, classify_status
        from finvet.models.evidence import _MARKET_FIELD_FOR_METRIC

        resolvable = (set(METRIC_TO_CONCEPTS) | set(_MARKET_FIELD_FOR_METRIC)
                      | set(FRED_SERIES))
        assert not (CORROBORATION_METRICS & resolvable)

        for target in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            assert classify_status("NOT_ENOUGH_INFO", target) != A2A_CONTRADICTS
