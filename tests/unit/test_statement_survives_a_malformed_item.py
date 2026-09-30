"""One malformed line item must not cost the whole statement.

`get_balance_sheet` failed with "1 validation error for FinancialItem /
period: Input should be a valid string". The MCP server returns some concepts
with a null `period`, the item constructor raised, and the exception discarded
every other line with it -- including the `Assets` fact the claim was about.
A total-assets claim for JPMorgan, Goldman Sachs, Wells Fargo or Morgan
Stanley went to a reviewer for want of a line that had arrived intact.

Driven through the tool, because that is where the failure was observed.
"""

from unittest.mock import patch

from finvet.mcp.sec_edgar import SECEdgarClient
from finvet.tools.sec_tools import get_balance_sheet

ACCN = "0001628280-26-008131"

MCP_WITH_A_NULL_PERIOD = {
    "success": True, "cik": 19617, "accession_number": ACCN,
    "concepts": {
        "Assets": {"value": 4_424_900_000_000.0, "unit": "USD",
                   "context": "c-4", "period": "2025-12-31"},
        "Liabilities": {"value": 4_062_462_000_000.0, "unit": "USD",
                        "context": "c-4", "period": None},
    },
    "filing_reference": {"filing_date": "2026-02-13"},
}

MCP_WITH_AN_UNBUILDABLE_ITEM = {
    "success": True, "cik": 19617, "accession_number": ACCN,
    "concepts": {
        "Assets": {"value": 4_424_900_000_000.0, "unit": "USD",
                   "context": "c-4", "period": "2025-12-31"},
        "Liabilities": {"value": 4_062_462_000_000.0, "unit": {"bad": "shape"},
                        "context": "c-4", "period": "2025-12-31"},
    },
    "filing_reference": {"filing_date": "2026-02-13"},
}


def _balance_sheet(mcp_payload):
    client = SECEdgarClient()
    with patch.object(client._mcp, "call_tool", return_value=mcp_payload), \
         patch.object(client, "_fetch_company_concept", return_value=None), \
         patch.object(client, "_fetch_frame_facts", return_value={}), \
         patch("finvet.tools.sec_tools._get_client", return_value=client):
        return get_balance_sheet.invoke(
            {"cik": "0000019617", "accession_number": ACCN})


class TestANullPeriodDoesNotFailTheTool:

    def test_the_call_succeeds(self):
        assert _balance_sheet(MCP_WITH_A_NULL_PERIOD)["success"] is True

    def test_the_line_that_arrived_intact_is_returned(self):
        items = {i["line_item"]: i
                 for i in _balance_sheet(MCP_WITH_A_NULL_PERIOD)["items"]}
        assert items["Assets"]["value"] == 4_424_900_000_000.0

    def test_the_line_without_a_period_is_kept_and_dated_by_its_filing(self):
        items = {i["line_item"]: i
                 for i in _balance_sheet(MCP_WITH_A_NULL_PERIOD)["items"]}
        assert items["Liabilities"]["value"] == 4_062_462_000_000.0
        assert items["Liabilities"]["period_end"] == "2026-02-13"


class TestAnUnbuildableItemIsSkippedAlone:

    def test_the_others_survive(self):
        result = _balance_sheet(MCP_WITH_AN_UNBUILDABLE_ITEM)
        assert result["success"] is True
        assert [i["line_item"] for i in result["items"]] == ["Assets"]
