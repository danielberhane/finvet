"""A Q4 claim is declined because it says Q4, not because a resolver agreed.

`Microsoft's Q4 fiscal 2025 revenue was $76 billion` returned three different
answers in three consecutive live runs: PENDING, REFUTES, and the correct
NOT_ENOUGH_INFO with `unsupported_q4_derivation`.

The REFUTES is the dangerous one. It carried `retrieved_value =
281,724,000,000` -- Microsoft's *annual* FY2025 revenue -- compared against a
claimed *quarterly* $76B and refuted on that basis. A correct number from the
wrong period scope produced a confident wrong verdict, which is the exact
failure the trusted-observation boundary exists to prevent.

The cause is that the decline tested `canonical_period.fiscal_quarter == "Q4"`.
Period resolution for "Q4 fiscal 2025" is not deterministic; when it yielded an
annual period instead, the claim proceeded, the agent retrieved the annual
fact, and the annual fact satisfied the annual window the guard checked.

Whether a claim *names* a quarter is a property of the claim, not a resolver
outcome. It is read from the parsed period text as well, so the decline no
longer depends on a step that may disagree with itself.
"""

import pytest


class _Period:
    def __init__(self, fiscal_quarter=None):
        self.fiscal_quarter = fiscal_quarter
        self.start_date = "2024-07-01"
        self.end_date = "2025-06-30"


class _Claim:
    claim_type = "sec"
    metric = "revenue"
    operator = "eq"
    value = 76_000_000_000.0

    def __init__(self, period):
        self.period = period


def _declined(claim, canonical):
    from finvet.graph.nodes.domain_agents import _unsupported_claim

    return _unsupported_claim({"parsed_claim": claim,
                               "canonical_period": canonical})


class TestTheQuarterIsReadFromTheClaim:

    @pytest.mark.parametrize("period_text", [
        "Q4 fiscal 2025", "Q4 2025", "fiscal Q4 2025", "q4 fy2025",
        "fourth quarter of fiscal 2025",
    ])
    def test_a_named_fourth_quarter_is_declined_without_the_resolver(
            self, period_text):
        """The resolver disagreed; the claim still says Q4."""
        declined = _declined(_Claim(period_text), _Period(fiscal_quarter=None))

        assert declined is not None, (
            f"{period_text!r} names Q4 and was allowed through to an agent")
        assert declined["limitation"] == "unsupported_q4_derivation"

    def test_the_resolver_still_settles_it_when_it_agrees(self):
        """The original path, unchanged."""
        declined = _declined(_Claim("fiscal 2025"), _Period("Q4"))

        assert declined is not None
        assert declined["limitation"] == "unsupported_q4_derivation"

    @pytest.mark.parametrize("period_text", [
        "fiscal 2025", "Q1 fiscal 2025", "Q3 2025", "fiscal year 2024",
    ])
    def test_other_periods_are_not_declined(self, period_text):
        """The control. Over-triggering would decline claims the system can
        answer, which is its own kind of wrong."""
        assert _declined(_Claim(period_text), _Period(fiscal_quarter=None)) is None

    def test_a_quarter_named_with_no_value_is_not_a_q4_numeric_claim(self):
        """The decline is about deriving a *number*; a claim with no value has
        nothing to derive."""
        claim = _Claim("Q4 fiscal 2025")
        claim.value = None

        assert _declined(claim, _Period(fiscal_quarter=None)) is None
