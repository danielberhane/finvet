"""A later filing's figure for a period supersedes an earlier one.

GE, fiscal 2023 revenue. GE's fiscal 2023 10-K reports
$67.954B; after the Vernova spin, its fiscal 2024 and 2025 10-Ks recast the
same period to $35.348B. The agent opens the fiscal 2023 filing, the selector
preferred the filing under inspection, and a true claim was refuted at
0.9 confidence. A restatement is the issuer telling the SEC the earlier number
no longer describes that period; SEC's frames feed resolves the same conflict
by taking the latest filing. So does this selector now.

The rule is narrow: only when filings DISAGREE on the same (start, end) does
the most recently filed fact win. Agreeing filings keep the previous order
(the filing under inspection first), and two different periods near one
fiscal anchor still decline rather than guess.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent
from finvet.mcp.sec_edgar import (
    SECEdgarClient,
    _choose_fact_for_period,
    _select_entity_wide_fact,
)
from finvet.models.claim import ParsedClaim
from finvet.models.evidence import resolve_trusted_observation
from finvet.tools.sec_tools import get_income_statement

CIK = "0000040545"
FY23 = "0000040545-24-000027"
FY24 = "0000040545-25-000015"
FY25 = "0000040545-26-000008"

ORIGINAL = {"start": "2023-01-01", "end": "2023-12-31", "val": 67_954_000_000,
            "accn": FY23, "form": "10-K", "fy": 2023, "fp": "FY",
            "filed": "2024-02-02"}
RECAST = {"start": "2023-01-01", "end": "2023-12-31", "val": 35_348_000_000,
          "accn": FY24, "form": "10-K", "fy": 2024, "fp": "FY",
          "filed": "2025-02-03"}
RECAST_AGAIN = {"start": "2023-01-01", "end": "2023-12-31", "val": 35_348_000_000,
                "accn": FY25, "form": "10-K", "fy": 2025, "fp": "FY",
                "filed": "2026-02-03"}
GE_REVENUES = {"units": {"USD": [ORIGINAL, RECAST, RECAST_AGAIN]}}

GE_MCP = {
    "success": True, "cik": 40545, "accession_number": FY23,
    "concepts": {
        "Revenues": {"value": 67_954_000_000.0, "unit": "USD",
                     "context": "c-1", "period": "2023-12-31"},
    },
    "filing_reference": {"filing_date": "2024-02-02"},
}


class TestChooseFactForPeriod:

    def test_the_latest_filing_wins_when_filings_disagree(self):
        got = _choose_fact_for_period(GE_REVENUES, FY23, "2023-12-31", "annual")
        assert got["val"] == 35_348_000_000
        assert got["accn"] == FY25

    def test_agreeing_filings_keep_the_filing_under_inspection(self):
        agreeing = {"units": {"USD": [
            {**ORIGINAL, "val": 35_348_000_000}, RECAST, RECAST_AGAIN]}}
        got = _choose_fact_for_period(agreeing, FY23, "2023-12-31", "annual")
        assert got["accn"] == FY23

    def test_recency_is_the_filing_date_not_the_accession_string(self):
        """Accession prefixes belong to filing agents, so they do not order
        filings; `filed` does."""
        late_by_date = {**RECAST, "accn": "0001193125-25-000001"}
        payload = {"units": {"USD": [ORIGINAL, late_by_date]}}
        got = _choose_fact_for_period(payload, FY23, "2023-12-31", "annual")
        assert got["val"] == 35_348_000_000

    def test_different_periods_are_not_a_restatement(self):
        """A quarter and the year share an end date; that is two periods, and
        the duration filter, not recency, decides between them."""
        payload = {"units": {"USD": [
            ORIGINAL,
            {"start": "2023-10-01", "end": "2023-12-31", "val": 19_423_000_000,
             "accn": FY24, "filed": "2025-02-03"},
        ]}}
        got = _choose_fact_for_period(payload, FY23, "2023-12-31", "annual")
        assert got["val"] == 67_954_000_000


class TestSelectEntityWideFact:

    def test_the_fallback_path_supersedes_the_same_way(self):
        assert _select_entity_wide_fact(GE_REVENUES, FY23, "2023-12-31",
                                        period="annual") == 35_348_000_000


class TestThroughTheClientAndTool:

    def _items(self):
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=GE_MCP), \
             patch.object(client, "_fetch_company_concept",
                          side_effect=lambda _c, name:
                          GE_REVENUES if name == "Revenues" else None), \
             patch.object(client, "_fetch_frame_facts", return_value={}):
            items = client.get_financials(CIK, FY23, "income", period="annual",
                                          period_end="2023-12-31")
        return items

    def test_get_financials_reports_the_recast_figure(self):
        revenues = next(i for i in self._items() if i.line_item == "Revenues")
        assert revenues.value == 35_348_000_000
        assert revenues.consolidated is True

    def test_the_trusted_observation_is_the_recast_figure(self):
        with patch("finvet.tools.sec_tools._get_client") as client:
            client.return_value.get_financials.return_value = self._items()
            result = get_income_statement.invoke(
                {"cik": CIK, "accession_number": FY23, "period": "annual"})
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
        obs = resolve_trusted_observation(
            claim, records, expected_period_start="2023-01-01",
            expected_period_end="2023-12-31")
        assert obs is not None
        assert obs.value == 35_348_000_000


class TestOnlyAStatementSupersedes:
    """The rule above needs a second one. FedEx's proxy statement (DEF 14A, filed after the 10-K) tags
    net income as 4,433 -- millions -- and MetLife's tags income available to
    common shareholders under the net income concept. Each was the most
    recently filed fact for its period, so each became the value of record and
    a false claim was supported at 0.95. Payloads recorded from SEC."""

    @staticmethod
    def _recorded(ticker, concept):
        import json
        from pathlib import Path
        path = (Path(__file__).resolve().parents[1] / "fixtures"
                / "sec_companyfacts" / f"{ticker}.json")
        return json.loads(path.read_text())["facts"]["us-gaap"][concept]

    def test_a_proxy_in_millions_does_not_replace_the_annual_report(self):
        payload = self._recorded("FDX", "NetIncomeLoss")
        got = _choose_fact_for_period(payload, None, "2026-05-31", "annual")
        assert got["val"] == 4_433_000_000
        assert got["form"] == "10-K"

    def test_a_proxy_reporting_another_measure_does_not_either(self):
        payload = self._recorded("MET", "NetIncomeLoss")
        got = _choose_fact_for_period(payload, None, "2025-12-31", "annual")
        assert got["val"] == 3_379_000_000

    def test_the_fallback_selector_applies_the_same_rule(self):
        payload = self._recorded("FDX", "NetIncomeLoss")
        assert _select_entity_wide_fact(payload, None, "2026-05-31",
                                        period="annual") == 4_433_000_000
