"""Which filed number answers a claim about a fiscal period.

Every payload here was recorded from SEC's companyfacts feed by
scripts/record_companyfacts_fixture.py and is unedited. The cases are issuers
whose filings defeat a simpler rule, alongside issuers whose conventions
differ from theirs, so a rule that fixes one convention and breaks another
fails here rather than in a paid run.
"""

import json
from pathlib import Path

import pytest

from finvet.mcp.fact_selection import (
    Decline,
    FiscalPeriod,
    Selection,
    build_fiscal_calendar,
    select_fact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sec_companyfacts"
CONTRACT = "RevenueFromContractWithCustomerExcludingAssessedTax"


def gaap(ticker):
    return json.loads((FIXTURES / f"{ticker}.json").read_text())["facts"]["us-gaap"]


def select(ticker, concept, year, period):
    facts = gaap(ticker)
    return select_fact(facts.get(concept), build_fiscal_calendar(facts), year, period)


class TestTheIssuerNamesItsOwnFiscalYear:
    """Home Depot names a fiscal year by the year it starts, Walmart and Nvidia
    by the year it ends. All three end within a week of 1 February."""

    def test_home_depot_fiscal_2025_ends_in_2026(self):
        assert build_fiscal_calendar(gaap("HD"))[(2025, "FY")] == FiscalPeriod(
            start="2025-02-03", end="2026-02-01")

    def test_home_depot_fiscal_2024_is_the_year_before(self):
        assert build_fiscal_calendar(gaap("HD"))[(2024, "FY")].end == "2025-02-02"

    def test_target_fiscal_2025_ends_in_2026(self):
        assert build_fiscal_calendar(gaap("TGT"))[(2025, "FY")].end == "2026-01-31"

    def test_walmart_fiscal_2026_ends_in_2026(self):
        assert build_fiscal_calendar(gaap("WMT"))[(2026, "FY")].end == "2026-01-31"

    def test_nvidia_fiscal_2025_ends_in_2025(self):
        assert build_fiscal_calendar(gaap("NVDA"))[(2025, "FY")].end == "2025-01-26"

    def test_apple_fiscal_2024_ends_in_september(self):
        assert build_fiscal_calendar(gaap("AAPL"))[(2024, "FY")].end == "2024-09-28"

    def test_vf_corp_fiscal_2024_ends_in_march(self):
        assert build_fiscal_calendar(gaap("VFC"))[(2024, "FY")].end == "2024-03-30"

    def test_a_quarter_is_the_three_months_not_the_year_to_date(self):
        assert build_fiscal_calendar(gaap("BAC"))[(2025, "Q2")] == FiscalPeriod(
            start="2025-04-01", end="2025-06-30")


class TestAMislabelledFilingDoesNotMoveThePeriod:
    """Target's 10-Q for the quarter ending 2022-10-29 is stamped fiscal 2023,
    and so is the one for the quarter ending 2023-10-28. Both claim to be
    fiscal 2023 Q3. A quarter lies inside its own fiscal year; only one does."""

    def test_the_quarter_inside_the_fiscal_year_is_the_one(self):
        calendar = build_fiscal_calendar(gaap("TGT"))
        assert calendar[(2023, "FY")] == FiscalPeriod(start="2023-01-29",
                                                      end="2024-02-03")
        assert calendar[(2023, "Q3")] == FiscalPeriod(start="2023-07-30",
                                                      end="2023-10-28")


class TestOnlyFinancialStatementsSupplyTheNumber:

    def test_fedex_proxy_in_millions_is_ignored(self):
        got = select("FDX", "NetIncomeLoss", 2026, "FY")
        assert got.value == 4_433_000_000
        assert got.form == "10-K"
        assert got.superseded == []

    def test_metlife_proxy_reporting_a_different_measure_is_ignored(self):
        got = select("MET", "NetIncomeLoss", 2025, "FY")
        assert got.value == 3_379_000_000
        assert got.superseded == []


class TestTheValueOfRecord:

    def test_home_depot_revenue(self):
        assert select("HD", CONTRACT, 2025, "FY").value == 164_683_000_000

    def test_a_balance_sheet_figure_is_the_instant_at_year_end(self):
        got = select("JPM", "Assets", 2025, "FY")
        assert got.value == 4_424_900_000_000
        assert got.end == "2025-12-31"

    def test_a_recast_year_returns_the_recast_and_remembers_the_original(self):
        got = select("GE", "Revenues", 2023, "FY")
        assert got.value == 35_348_000_000
        assert [s.value for s in got.superseded] == [67_954_000_000]

    def test_a_recast_quarter_does_the_same(self):
        got = select("BAC", "Revenues", 2025, "Q2")
        assert got.value == 27_443_000_000
        assert [s.value for s in got.superseded] == [26_463_000_000]

    def test_the_concept_the_narrow_feed_left_empty_resolves(self):
        got = select("VFC", CONTRACT, 2024, "FY")
        assert got.value == 9_915_678_000
        assert [s.value for s in got.superseded] == [10_454_667_000]

    def test_the_selection_says_where_it_came_from(self):
        got = select("GE", "Revenues", 2023, "FY")
        assert isinstance(got, Selection)
        assert (got.fiscal_year, got.fiscal_period) == (2023, "FY")
        assert (got.start, got.end) == ("2023-01-01", "2023-12-31")
        assert got.form == "10-K" and got.filed >= "2025-01-01"
        assert got.superseded[0].filed == "2024-02-02"


class TestDecliningRatherThanGuessing:

    def test_a_concept_carrying_only_the_fourth_quarter_is_not_the_year(self):
        got = select("GE", CONTRACT, 2023, "FY")
        assert isinstance(got, Decline)
        assert got.reason == "no_statement_fact_for_period"

    def test_a_year_nobody_has_filed(self):
        got = select("HD", CONTRACT, 2031, "FY")
        assert isinstance(got, Decline)
        assert got.reason == "fiscal_period_not_filed"

    def test_a_fourth_quarter_has_no_report_of_its_own(self):
        assert isinstance(select("BAC", "Revenues", 2025, "Q4"), Decline)

    def test_a_concept_the_issuer_does_not_report(self):
        got = select_fact(None, build_fiscal_calendar(gaap("HD")), 2025, "FY")
        assert isinstance(got, Decline)
        assert got.reason == "concept_not_reported"


class TestAReportCannotDescribeAPeriodThatEndsAfterItWasFiled:
    """The one perturbed payload in this file: a real fact, copied, with its
    dates moved a year forward, standing in for the forward-looking contexts
    some filings carry. Without the guard it would become the fiscal year."""

    def test_a_future_dated_fact_does_not_move_the_fiscal_year(self):
        facts = gaap("HD")
        real = next(f for f in facts[CONTRACT]["units"]["USD"]
                    if f["end"] == "2026-02-01" and f["form"] == "10-K")
        facts[CONTRACT]["units"]["USD"].append(
            {**real, "start": "2026-02-02", "end": "2027-01-31", "val": 1})
        assert build_fiscal_calendar(facts)[(2025, "FY")].end == "2026-02-01"


@pytest.mark.parametrize("ticker", sorted(p.stem for p in FIXTURES.glob("*.json")))
def test_no_fiscal_period_in_any_recorded_issuer_is_ambiguous(ticker):
    calendar = build_fiscal_calendar(gaap(ticker))
    assert calendar, "the calendar is empty"
    assert all(period is not None for period in calendar.values())
