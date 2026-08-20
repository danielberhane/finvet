"""Tests for period-targeted XBRL fact selection.

The defect: get_financials asked the MCP server for a company and a concept but
never for a period. The server returns one arbitrary XBRL context per concept, so
a filing's comparative years came back interchangeably — Ford's FY2025 10-K
answered a fiscal-2024 question with the 2023 figure.

SEC's companyconcept feed carries every fact with its start and end dates, so the
target period is selectable there. These tests pin that selection, including the
duration ambiguity a single end date cannot resolve on its own.
"""

from unittest.mock import patch

from finvet.mcp.sec_edgar import SECEdgarClient, _select_fact_for_period

ACCN = "0000004962-25-000083"
OTHER_ACCN = "0000004962-24-000052"

# Amex NetIncomeLoss, end 2024-06-30: SEC carries both the 90-day quarter and the
# 181-day year-to-date figure. Matching on the end date alone cannot tell them apart.
AMEX = {"units": {"USD": [
    {"start": "2024-01-01", "end": "2024-06-30", "val": 5_452_000_000,
     "accn": OTHER_ACCN, "form": "10-Q", "fy": 2024, "fp": "Q2"},
    {"start": "2024-04-01", "end": "2024-06-30", "val": 3_015_000_000,
     "accn": OTHER_ACCN, "form": "10-Q", "fy": 2024, "fp": "Q2"},
    {"start": "2024-04-01", "end": "2024-06-30", "val": 3_015_000_000,
     "accn": ACCN, "form": "10-Q", "fy": 2025, "fp": "Q2"},
]}}

# Balance-sheet concepts are instant facts — no start date at all.
INSTANT = {"units": {"USD": [
    {"end": "2024-09-28", "val": 364_980_000_000, "accn": ACCN, "form": "10-K"},
    {"end": "2023-09-30", "val": 352_583_000_000, "accn": ACCN, "form": "10-K"},
]}}


class TestSelectFactForPeriod:

    def test_quarterly_picks_the_three_month_fact_not_year_to_date(self):
        got = _select_fact_for_period(AMEX, ACCN, "2024-06-30", "quarterly")
        assert got == 3_015_000_000

    def test_annual_picks_the_twelve_month_fact(self):
        annual = {"units": {"USD": [
            {"start": "2024-01-01", "end": "2024-12-31", "val": 8_818_000_000,
             "accn": ACCN, "form": "10-K"},
            {"start": "2024-10-01", "end": "2024-12-31", "val": 2_100_000_000,
             "accn": ACCN, "form": "10-K"},
        ]}}
        assert _select_fact_for_period(annual, ACCN, "2024-12-31", "annual") == 8_818_000_000

    def test_prefers_the_filing_under_inspection_when_durations_tie(self):
        """Two filings report the same quarter; the one we asked about wins."""
        assert _select_fact_for_period(AMEX, ACCN, "2024-06-30", "quarterly") == 3_015_000_000
        assert _select_fact_for_period(AMEX, OTHER_ACCN, "2024-06-30", "quarterly") == 3_015_000_000

    def test_instant_facts_have_no_duration_and_still_resolve(self):
        assert _select_fact_for_period(INSTANT, ACCN, "2024-09-28", "annual") == 364_980_000_000

    def test_wrong_period_end_returns_nothing(self):
        assert _select_fact_for_period(AMEX, ACCN, "2024-03-31", "quarterly") is None

    def test_no_matching_duration_returns_nothing_rather_than_guessing(self):
        """Only a 181-day fact exists; a quarterly request must not accept it."""
        ytd_only = {"units": {"USD": [
            {"start": "2024-01-01", "end": "2024-06-30", "val": 5_452_000_000,
             "accn": ACCN, "form": "10-Q"},
        ]}}
        assert _select_fact_for_period(ytd_only, ACCN, "2024-06-30", "quarterly") is None

    def test_missing_period_end_returns_nothing(self):
        assert _select_fact_for_period(AMEX, ACCN, None, "quarterly") is None


class TestGetFinancialsTargetsThePeriod:

    MCP_WRONG_PERIOD = {
        "success": True,
        "cik": 4962,
        "accession_number": ACCN,
        "concepts": {
            "NetIncomeLoss": {
                "value": 1_895_000_000.0, "unit": "USD",
                "context": "c-16", "period": "2025-06-30",
            }
        },
        "filing_reference": {"filing_date": "2025-07-19"},
    }

    def _client(self):
        return SECEdgarClient()

    def test_period_end_overrides_the_arbitrary_mcp_fact(self):
        """The live defect: MCP returned the 2025 comparative for a 2024 question."""
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=self.MCP_WRONG_PERIOD), \
             patch.object(client, "_fetch_company_concept", return_value=AMEX):
            items = client.get_financials(
                "0000004962", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        item = next(i for i in items if i.line_item == "NetIncomeLoss")
        assert item.value == 3_015_000_000
        assert item.period_end == "2024-06-30"

    def test_period_targeted_value_is_marked_confirmed(self):
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=self.MCP_WRONG_PERIOD), \
             patch.object(client, "_fetch_company_concept", return_value=AMEX):
            items = client.get_financials(
                "0000004962", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        assert next(i for i in items if i.line_item == "NetIncomeLoss").consolidated is True

    def test_unresolvable_period_keeps_the_filing_value_but_flags_it(self):
        """Never silently pass off an unverified figure as the requested period."""
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=self.MCP_WRONG_PERIOD), \
             patch.object(client, "_fetch_company_concept", return_value=None):
            items = client.get_financials(
                "0000004962", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        item = next(i for i in items if i.line_item == "NetIncomeLoss")
        assert item.value == 1_895_000_000.0
        assert item.consolidated is False

    def test_without_a_target_period_behaviour_is_unchanged(self):
        """Callers that pass no period_end must behave exactly as before."""
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=self.MCP_WRONG_PERIOD), \
             patch.object(client, "_fetch_company_concept", return_value=AMEX):
            items = client.get_financials("0000004962", ACCN, "income", period="quarterly")
        item = next(i for i in items if i.line_item == "NetIncomeLoss")
        assert item.value == 1_895_000_000.0
        assert item.period_end == "2025-06-30"


class TestRequestPeriodMissFallsBackToConsolidation:
    """Regression: Apple, 'fiscal 2024'. period_resolver maps fiscal years to
    calendar dates, so the requested end (2024-12-31) matches nothing for a
    company whose year ends 2024-09-28. The original period fix kept the raw
    MCP value on that miss — skipping the consolidation overlay entirely —
    and the segment-shadowed Products figure ($294.866B) reached the verdict:
    REFUTES 0.95 against a true $391B claim, live.

    On a miss, each item now falls back to the pre-period-fix logic: confirm
    the entity-wide fact for the item's OWN period. The fiscal/calendar
    misalignment itself remains tracked (F-05) — this restores the safety
    net underneath it."""

    APPLE_MCP = {
        "success": True, "cik": 320193,
        "accession_number": "0000320193-24-000123",
        "concepts": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "value": 294_866_000_000.0, "unit": "USD",
                "context": "c-13", "period": "2024-09-28",   # Products segment
            },
            "NetIncomeLoss": {
                "value": 93_736_000_000.0, "unit": "USD",
                "context": "c-1", "period": "2024-09-28",
            },
        },
        "filing_reference": {"filing_date": "2024-11-01"},
    }
    APPLE_CC = {"units": {"USD": [
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000,
         "accn": "0000320193-24-000123", "form": "10-K"},
    ]}}

    def _get(self, cc_payload):
        from unittest.mock import patch
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=self.APPLE_MCP), \
             patch.object(client, "_fetch_company_concept", return_value=cc_payload):
            items = client.get_financials(
                "0000320193", "0000320193-24-000123", "income",
                period="annual", period_end="2024-12-31",   # the calendar miss
            )
        return {i.line_item: i for i in items}

    def test_segment_shadowed_value_is_still_corrected_on_a_miss(self):
        revenue = self._get(self.APPLE_CC)[
            "RevenueFromContractWithCustomerExcludingAssessedTax"]
        assert revenue.value == 391_035_000_000
        assert revenue.consolidated is True
        assert revenue.period_end == "2024-09-28"   # the honest period

    def test_non_sensitive_concept_keeps_flag_semantics_on_a_miss(self):
        """None means 'not consolidation-sensitive' everywhere else; a miss
        must not relabel NetIncomeLoss as unverified."""
        assert self._get(self.APPLE_CC)["NetIncomeLoss"].consolidated is None

    def test_nothing_entity_wide_anywhere_stays_flagged(self):
        revenue = self._get(None)[
            "RevenueFromContractWithCustomerExcludingAssessedTax"]
        assert revenue.value == 294_866_000_000.0
        assert revenue.consolidated is False
