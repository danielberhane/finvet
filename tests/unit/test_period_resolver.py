"""Tests for period resolution.

Regression cover for the branch-ordering bug: the explicit ISO-date arm sat
below the annual arm, whose pattern matches any string containing four
consecutive digits. Every ISO date was therefore widened to its calendar year —
a balance-sheet claim about 2024-03-31 resolved to a 12-month span and reported
period_type "annual".
"""

from finvet.graph.nodes.period_resolver import period_resolver
from finvet.models.claim import ParsedClaim


def _resolve(period):
    state = {
        "parsed_claim": ParsedClaim(claim_type="sec", ticker="AAPL", period=period),
        "request_id": "test",
    }
    return period_resolver(state)["canonical_period"]


class TestExplicitDates:

    def test_iso_date_resolves_to_a_single_day(self):
        period = _resolve("2024-03-31")
        assert period.period_type == "date"
        assert period.start_date == "2024-03-31"
        assert period.end_date == "2024-03-31"

    def test_iso_date_is_not_an_assumption(self):
        """An explicit date is stated, not inferred — nothing to disclose."""
        period = _resolve("2024-03-31")
        assert period.is_assumption is False
        assert period.assumptions == []

    def test_year_end_iso_date_is_not_widened_to_the_year(self):
        """2024-12-31 shares its digits with calendar 2024; it is still one day."""
        period = _resolve("2024-12-31")
        assert period.period_type == "date"
        assert period.start_date == period.end_date == "2024-12-31"

    def test_a_string_merely_starting_with_a_date_is_not_a_single_day(self):
        """start_date is an unvalidated str, so a partial match would store prose
        in a date field. Only a bare ISO date may take the date branch."""
        period = _resolve("2024-03-31 through 2024-06-30")
        assert period.period_type != "date"
        assert period.start_date == "2024-01-01"


class TestTrailingLetterNotation:
    """"4Q 2024" and "1H 2024" put the digit before the letter. Both regexes
    required the letter first, so both forms fell through to the annual arm and
    silently widened a three- or six-month claim to twelve months."""

    def test_trailing_q_quarter(self):
        period = _resolve("4Q 2024")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-10-01", "2024-12-31")

    def test_trailing_q_first_quarter(self):
        period = _resolve("1Q 2024")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-01-01", "2024-03-31")

    def test_trailing_h_first_half(self):
        period = _resolve("1H 2024")
        assert period.period_type == "half_year"
        assert (period.start_date, period.end_date) == ("2024-01-01", "2024-06-30")

    def test_trailing_h_second_half(self):
        period = _resolve("2H 2023")
        assert period.period_type == "half_year"
        assert (period.start_date, period.end_date) == ("2023-07-01", "2023-12-31")

    def test_trailing_h_without_a_space(self):
        period = _resolve("1H2024")
        assert period.period_type == "half_year"
        assert (period.start_date, period.end_date) == ("2024-01-01", "2024-06-30")


class TestExistingFormatsStillResolve:

    def test_quarter(self):
        period = _resolve("Q4 2024")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-10-01", "2024-12-31")

    def test_fiscal_year(self):
        period = _resolve("fiscal 2024")
        assert period.period_type == "annual"
        assert (period.start_date, period.end_date) == ("2024-01-01", "2024-12-31")

    def test_half_year(self):
        period = _resolve("H1 2024")
        assert period.period_type == "half_year"
        assert (period.start_date, period.end_date) == ("2024-01-01", "2024-06-30")

    def test_year_first_quarter_notation(self):
        period = _resolve("2024Q3")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-07-01", "2024-09-30")

    def test_event_relative_period_is_flagged_not_resolved(self):
        period = _resolve("after the earnings announcement")
        assert period.period_type == "event_relative"

    def test_leading_q_with_fiscal_marker(self):
        period = _resolve("Q4 FY2024")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-10-01", "2024-12-31")

    def test_year_then_space_then_quarter(self):
        period = _resolve("2024 Q2")
        assert period.period_type == "quarterly"
        assert (period.start_date, period.end_date) == ("2024-04-01", "2024-06-30")

    def test_unparseable_period_falls_back_to_current(self):
        period = _resolve("sometime recently")
        assert period.period_type == "current"

    def test_prose_date_still_misreads_as_a_quarter(self):
        """Characterization, not endorsement. The leading-Q pattern makes the Q
        optional, so "March 31 2024" matches on the bare "1 2024" and resolves to
        Q1. Pinned here so the trailing-Q work below cannot quietly change it —
        it is a separate defect, tracked but not fixed."""
        period = _resolve("March 31 2024")
        assert period.period_type == "quarterly"
        assert period.fiscal_quarter == "Q1"
