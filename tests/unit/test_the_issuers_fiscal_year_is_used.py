"""The fiscal year the issuer keeps, not the calendar year the claim implies.

`Nvidia's revenue was $130.5 billion in fiscal year 2025` returned Human Review
while the correct fact sat in SEC's data unasked for.

`period_resolver` maps "fiscal year 2025" to the calendar window
2025-01-01..2025-12-31, so the client targets 2025-12-31. Nvidia's fiscal year
ends in January, so its FY2025 runs Jan 2024 - Jan 2025 and lies mostly in
calendar *2024*. SEC's frames API is keyed by calendar frame and therefore puts
Nvidia's FY2026 in CY2025, so the fallback returned 215,938,000,000 (period end
2026-01-25). The period guard rejected it -- correctly -- leaving nothing to
compare, and the agent burned its tool budget searching filing text.

Measured before this was written, and it ruled out two other explanations:

    frames at the calendar end
      AAPL -> (391,035,000,000, '2024-09-28')   correct
      MSFT -> (281,724,000,000, '2025-06-30')   correct
      AMZN -> (637,959,000,000, '2024-12-31')   correct
      NVDA -> (215,938,000,000, '2026-01-25')   wrong year

So frames is not broken, and `_select_fact_for_period` is not broken either:
given the true fiscal end it returns the right value for all four. The anchor
date was wrong, and only the anchor is changed here.

`get_company_info` already returns `fiscal_year_end` (MMDD) and is the agent's
first tool call on every SEC run, so the anchor costs no extra round trip.

Fixtures are real companyconcept facts, trimmed to annual periods from 2023 on.
"""

import json
import pathlib

import pytest

from finvet.mcp.sec_edgar import _select_fact_for_period

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent.parent / "fixtures"
     / "xbrl_revenue_concepts.json").read_text())

# Truth from the filings themselves, quoted so a reader can check rather than
# trust: the fiscal year each issuer labels, and what it reported.
TRUTH = {
    "AAPL": (2024, "0926", 391_035_000_000, "2024-09-28"),
    "MSFT": (2025, "0630", 281_724_000_000, "2025-06-30"),
    "NVDA": (2025, "0131", 130_497_000_000, "2025-01-26"),
    "AMZN": (2024, "1231", 637_959_000_000, "2024-12-31"),
}


def _anchor(ticker, year):
    from finvet.mcp.sec_edgar import fiscal_anchor_for

    return fiscal_anchor_for(FIXTURE[ticker]["fiscal_year_end"], year)


class TestEachIssuersOwnYearIsResolved:

    @pytest.mark.parametrize("ticker", sorted(TRUTH))
    def test_the_filed_figure_is_returned(self, ticker):
        year, _fye, expected, _end = TRUTH[ticker]
        calendar_end = f"{year}-12-31"

        value = _select_fact_for_period(
            FIXTURE[ticker], None, calendar_end, "annual",
            anchor=_anchor(ticker, year))

        assert value == expected

    def test_nvidia_is_the_regression_case(self):
        """The defect verbatim: FY2026 (215,938,000,000, ending 2026-01-25) is
        what the calendar-keyed lookup returned. It must never be chosen for a
        claim about FY2025."""
        value = _select_fact_for_period(
            FIXTURE["NVDA"], None, "2025-12-31", "annual",
            anchor=_anchor("NVDA", 2025))

        assert value == 130_497_000_000
        assert value != 215_938_000_000


class TestNothingElseChanges:

    def test_without_an_anchor_exact_matching_still_applies(self):
        """A caller with no fiscal calendar to anchor on behaves as before."""
        amzn = _select_fact_for_period(
            FIXTURE["AMZN"], None, "2024-12-31", "annual")
        aapl = _select_fact_for_period(
            FIXTURE["AAPL"], None, "2024-12-31", "annual")

        assert amzn == 637_959_000_000   # calendar filer: exact match hits
        assert aapl is None              # September filer: no fact on 12-31

    def test_a_missing_fiscal_year_end_yields_no_anchor(self):
        from finvet.mcp.sec_edgar import fiscal_anchor_for

        assert fiscal_anchor_for(None, 2025) is None
        assert fiscal_anchor_for("", 2025) is None
        assert fiscal_anchor_for("nonsense", 2025) is None

    def test_a_quarterly_fact_near_the_anchor_is_not_used(self):
        """The duration filter, not the anchor, is what excludes a quarter."""
        payload = {"units": {"USD": [
            {"start": "2024-10-27", "end": "2025-01-26",
             "val": 39_331_000_000, "accn": "q"},          # ~91 days: Q4
        ]}}

        value = _select_fact_for_period(
            payload, None, "2025-12-31", "annual", anchor=_anchor("NVDA", 2025))

        assert value is None


class TestAmbiguityDeclines:

    def test_two_distinct_values_equidistant_from_the_anchor_decline(self):
        """A company that changed its fiscal year could file two annual periods
        near one anchor. Choosing either would be a coin toss."""
        # Anchor is 2025-01-31 (fiscal_year_end "0131"); both ends are 10 days
        # from it, one either side, so nearness cannot separate them.
        payload = {"units": {"USD": [
            {"start": "2024-01-21", "end": "2025-01-21", "val": 100.0, "accn": "a"},
            {"start": "2024-02-10", "end": "2025-02-10", "val": 200.0, "accn": "b"},
        ]}}

        value = _select_fact_for_period(
            payload, None, "2025-12-31", "annual", anchor=_anchor("NVDA", 2025))

        assert value is None

    def test_the_same_value_reported_twice_is_one_fact(self):
        """The disclosure repeats across filings; equal values are not
        ambiguity."""
        payload = {"units": {"USD": [
            {"start": "2024-01-29", "end": "2025-01-26", "val": 130_497_000_000,
             "accn": "a"},
            {"start": "2024-01-29", "end": "2025-01-26", "val": 130_497_000_000,
             "accn": "b"},
        ]}}

        value = _select_fact_for_period(
            payload, None, "2025-12-31", "annual", anchor=_anchor("NVDA", 2025))

        assert value == 130_497_000_000

    def test_the_filing_under_inspection_breaks_a_tie(self):
        payload = {"units": {"USD": [
            {"start": "2024-01-20", "end": "2025-01-18", "val": 100.0, "accn": "other"},
            {"start": "2024-02-07", "end": "2025-02-07", "val": 200.0, "accn": "mine"},
        ]}}

        value = _select_fact_for_period(
            payload, "mine", "2025-12-31", "annual", anchor=_anchor("NVDA", 2025))

        assert value == 200.0


class TestTheRecordedPeriodIsTheFactsOwn:
    """The value now comes from a fact whose end differs from the requested
    calendar date, so stamping the request onto the item would record a period
    the number does not have.

    Before the anchor existed this could not happen: a fact was only selected
    when its end *equalled* the request, so the stamp was accurate. It is the
    same failure the rest of this work removed -- a record asserting something
    that is not so -- and it arrived as a side effect of the fix.
    """

    def _resolve(self, monkeypatch):
        from finvet.mcp.sec_edgar import FinancialItem, SECEdgarClient

        client = SECEdgarClient.__new__(SECEdgarClient)
        client._concept_cache = {}
        client._frame_cache = {}
        client._fiscal_year_ends = {"0001045810": "0131"}
        monkeypatch.setattr(client, "_fetch_company_concept",
                            lambda cik, concept: FIXTURE["NVDA"], raising=False)

        item = FinancialItem(line_item="Revenues", concept="us-gaap:Revenues",
                             value=215_938_000_000.0, units="USD",
                             period="2026-01-25", period_end="2026-01-25")
        return client._resolve_period([item], "0001045810", None,
                                      "2025-12-31", "annual")[0]

    def test_the_right_value_is_taken(self, monkeypatch):
        assert self._resolve(monkeypatch).value == 130_497_000_000

    def test_the_period_recorded_is_the_facts_own(self, monkeypatch):
        item = self._resolve(monkeypatch)

        assert item.period_end == "2025-01-26", (
            "the item recorded the requested calendar date, not the period "
            "the value actually covers")
        assert item.period == "2025-01-26"
