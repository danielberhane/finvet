"""The boundary between tool execution and numeric verdicts.

The deterministic override is FinVet's headline guarantee, and it is only as
good as the number it is handed. Two defects sit under it:

1. Every SEC and Market tool catches its exceptions and *returns* a result with
   success=False rather than raising. LangChain therefore reports the call's
   transport status as success, and _extract_tool_info records the failed call
   as successful.
2. The retrieved-value fallback regex-scrapes numbers out of the serialized
   result string, truncated to TOOL_RESULT_PREVIEW_CHARS -- so what the
   comparator receives depends on where a string was cut.

These tests drive the real decorated tools. An earlier test in
test_base_agent.py hand-built ToolMessage(status="error"), a shape production
never emits, which is why it passed while the defect shipped.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent
from finvet.mcp.sec_edgar import FinancialItem
from finvet.tools.sec_tools import get_income_statement


class _Claim:
    """Minimal ParsedClaim stand-in for the fallback's metric lookup."""
    metric = "revenue"
    value = 150_000_000_000.0
    operator = "eq"


def _detail_for(result):
    """Run a tool result through the real provenance extraction."""
    call = {"name": "get_income_statement", "args": {}, "id": "call_1"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=str(result), tool_call_id="call_1",
                    name="get_income_statement"),
    ]
    _, detail, _ = BaseVerificationAgent._extract_tool_info(
        BaseVerificationAgent, messages)
    return detail


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
        assert result.success is False, "precondition: the tool caught the error"

        detail = _detail_for(result)
        assert detail[0]["success"] is False

    def test_successful_call_is_still_recorded_as_successful(self):
        """The guard must not swing the other way."""
        result = _invoke_succeeding([
            FinancialItem(line_item="Revenues", concept="Revenues",
                          value=391_035_000_000.0, units="USD",
                          period="annual", period_end="2024-09-28"),
        ])
        assert _detail_for(result)[0]["success"] is True


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

        value = BaseVerificationAgent._extract_retrieved_value(
            _detail_for(result), _Claim())

        assert value is None


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

        value = BaseVerificationAgent._extract_retrieved_value(
            _detail_for(result), _Claim())

        assert value == 391_035_000_000.0
