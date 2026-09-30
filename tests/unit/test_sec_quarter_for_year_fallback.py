"""A quarter's fact must never be served as the year's.

Found live on a GE fiscal 2023 revenue claim. GE tags
its total revenue under `Revenues`; the `RevenueFromContractWithCustomer...`
concept carries only the Q4 three-month figure at the year-end date. The
period-targeted selector correctly rejected that fact (92 days is not a
year), but the fallback that then confirms the filing's OWN period matched on
the end date alone, took the Q4 figure ($18.514B), marked it entity-wide,
and the comparator refuted a true annual claim at 0.85 confidence.
Same outcome under two different models: no model was involved.

Two behaviours pinned here, both driven through the real client and tool:
the fallback applies the same duration window as the primary selector, and
once the contract-revenue item is unverified the observation falls through
to `Revenues`, or declines when nothing of the right span exists anywhere.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent
from finvet.mcp.sec_edgar import SECEdgarClient, _select_entity_wide_fact
from finvet.models.claim import ParsedClaim
from finvet.models.evidence import resolve_trusted_observation
from finvet.tools.sec_tools import get_income_statement

CIK = "0000040545"
ACCN = "0000040545-24-000027"          # GE fiscal 2023 10-K
CONTRACT = "RevenueFromContractWithCustomerExcludingAssessedTax"

# What the MCP server returned for the filing: one arbitrary context per concept.
GE_MCP = {
    "success": True, "cik": 40545, "accession_number": ACCN,
    "concepts": {
        CONTRACT: {"value": 26_793_000_000.0, "unit": "USD",
                   "context": "c-7", "period": "2023-12-31"},
        "Revenues": {"value": 67_954_000_000.0, "unit": "USD",
                     "context": "c-1", "period": "2023-12-31"},
    },
    "filing_reference": {"filing_date": "2024-02-02"},
}

# SEC companyconcept, as it really is for GE: the contract-revenue concept
# has only the fourth quarter at the year-end date; Revenues has the year.
Q4_ONLY = {"units": {"USD": [
    {"start": "2023-10-01", "end": "2023-12-31", "val": 18_514_000_000,
     "accn": ACCN, "form": "10-K", "fy": 2023, "fp": "FY"},
]}}
ANNUAL = {"units": {"USD": [
    {"start": "2023-01-01", "end": "2023-12-31", "val": 67_954_000_000,
     "accn": ACCN, "form": "10-K", "fy": 2023, "fp": "FY"},
    {"start": "2023-10-01", "end": "2023-12-31", "val": 19_423_000_000,
     "accn": ACCN, "form": "10-K", "fy": 2023, "fp": "FY"},
]}}


def _items(concept_payloads):
    """Run the real client with the network patched to GE's shape."""
    client = SECEdgarClient()

    def concept(_cik, name):
        return concept_payloads.get(name)

    with patch.object(client._mcp, "call_tool", return_value=GE_MCP), \
         patch.object(client, "_fetch_company_concept", side_effect=concept), \
         patch.object(client, "_fetch_frame_facts", return_value={}):
        items = client.get_financials(CIK, ACCN, "income",
                                      period="annual", period_end="2023-12-31")
    return {i.line_item: i for i in items}


def _observation(items):
    """Feed those items through the tool and the agent's extraction, as live."""
    with patch("finvet.tools.sec_tools._get_client") as client:
        client.return_value.get_financials.return_value = list(items.values())
        result = get_income_statement.invoke(
            {"cik": CIK, "accession_number": ACCN, "period": "annual"})
    call = {"name": "get_income_statement", "args": {}, "id": "call_1"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=str(result), tool_call_id="call_1",
                    name="get_income_statement"),
    ]
    _, _, _, records = BaseVerificationAgent._extract_tool_info(
        BaseVerificationAgent, messages)
    claim = ParsedClaim(claim_type="sec", ticker="GE", metric="revenue",
                        operator="eq", value=35_300_000_000.0, period="FY2023")
    return resolve_trusted_observation(
        claim, records, expected_period_start="2023-01-01",
        expected_period_end="2023-12-31")


class TestSelectEntityWideFactRespectsDuration:

    def test_a_quarter_fact_is_not_the_year(self):
        assert _select_entity_wide_fact(Q4_ONLY, ACCN, "2023-12-31",
                                        period="annual") is None

    def test_the_annual_fact_is_still_found(self):
        assert _select_entity_wide_fact(ANNUAL, ACCN, "2023-12-31",
                                        period="annual") == 67_954_000_000

    def test_without_a_period_kind_the_end_date_alone_decides_as_before(self):
        """The no-period consolidation path passes no kind and is unchanged."""
        assert _select_entity_wide_fact(Q4_ONLY, ACCN, "2023-12-31") == 18_514_000_000


class TestGetFinancialsNeverServesAQuarterAsTheYear:

    def test_the_quarter_fact_does_not_replace_the_filing_value(self):
        contract = _items({CONTRACT: Q4_ONLY, "Revenues": ANNUAL})[CONTRACT]
        assert contract.value == 26_793_000_000.0
        assert contract.consolidated is False

    def test_the_annual_revenues_fact_is_confirmed_beside_it(self):
        revenues = _items({CONTRACT: Q4_ONLY, "Revenues": ANNUAL})["Revenues"]
        assert revenues.value == 67_954_000_000
        assert revenues.consolidated is True


class TestObservationFallsThroughToRevenues:

    def test_the_observation_is_the_annual_revenues_fact(self):
        obs = _observation(_items({CONTRACT: Q4_ONLY, "Revenues": ANNUAL}))
        assert obs is not None
        assert obs.concept == "Revenues"
        assert obs.value == 67_954_000_000

    def test_nothing_of_the_right_span_anywhere_declines(self):
        """Both concepts quarterly only: no observation, so the verdict
        fails closed instead of comparing a quarter to a year."""
        q4_revenues = {"units": {"USD": [
            {"start": "2023-10-01", "end": "2023-12-31", "val": 19_423_000_000,
             "accn": ACCN, "form": "10-K"},
        ]}}
        assert _observation(_items({CONTRACT: Q4_ONLY,
                                    "Revenues": q4_revenues})) is None
