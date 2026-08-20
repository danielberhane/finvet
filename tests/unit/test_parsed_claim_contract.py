"""Tests for the expanded ParsedClaim — stage 03 of the parser-contract migration.

EXPAND phase: `metric` and `operator` are added, `comparison` and `currency`
are kept, `reject_reason` opens to any string. `operator` and `comparison`
mirror each other so old readers and new writers coexist; the pair diverging
is a bug, not a preference.

Two of the three §8 invariants are deliberately deferred to stage 04: the
"reject nulls every other field" and "operator iff value" rules would 500 the
live path today, because reconcile_reject_fields coerces claim_type without
nulling the other fields and DeepSeek can emit a comparison without a value.
They land with the boundary normalization that makes them safe.
"""

import pytest
from pydantic import ValidationError

from finvet.models.claim import ParsedClaim


class TestMetricField:

    def test_whitelisted_metric_is_accepted(self):
        claim = ParsedClaim(claim_type="sec", metric="revenue")
        assert claim.metric == "revenue"

    def test_metric_defaults_to_none(self):
        assert ParsedClaim(claim_type="sec").metric is None

    def test_metric_from_the_wrong_class_is_rejected(self):
        """closing_price is valid for market — a sec claim carrying it means
        the claim_type and metric disagree, which fail-closed cannot catch
        downstream. The model is the last gate."""
        with pytest.raises(ValidationError, match="metric"):
            ParsedClaim(claim_type="sec", metric="closing_price")

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ValidationError, match="metric"):
            ParsedClaim(claim_type="sec", metric="iphone_revenue")

    def test_reject_claims_cannot_carry_a_metric(self):
        with pytest.raises(ValidationError, match="metric"):
            ParsedClaim(claim_type="reject", reject_reason="question",
                        metric="revenue")


class TestOperatorMirrorsComparison:
    """During EXPAND both names are live: the gold writes `operator`, every
    existing reader and test writes `comparison`. They must be one value."""

    def test_operator_fills_comparison(self):
        claim = ParsedClaim(claim_type="sec", value=1.0, operator="gt")
        assert claim.comparison == "gt"

    def test_comparison_fills_operator(self):
        claim = ParsedClaim(claim_type="sec", value=1.0, comparison="lte")
        assert claim.operator == "lte"

    def test_agreeing_duplicates_are_fine(self):
        claim = ParsedClaim(claim_type="sec", value=1.0,
                            operator="eq", comparison="eq")
        assert claim.operator == claim.comparison == "eq"

    def test_diverging_values_are_an_error(self):
        with pytest.raises(ValidationError, match="operator"):
            ParsedClaim(claim_type="sec", value=1.0,
                        operator="gt", comparison="lt")

    def test_both_none_stays_none(self):
        claim = ParsedClaim(claim_type="news")
        assert claim.operator is None and claim.comparison is None

    @pytest.mark.parametrize("op", ["approx", "range"])
    def test_new_operators_are_accepted_on_both_names(self, op):
        claim = ParsedClaim(claim_type="sec", value=1.0, operator=op)
        assert claim.comparison == op


class TestOpenRejectReason:

    @pytest.mark.parametrize("reason", [
        "future_prediction",                  # in gold, failed the old Literal
        "prompt_injection_or_instruction",
        "no_verifiable_quantity",
        "unspecified",                        # ours, from 05d8300
    ])
    def test_gold_vocabulary_is_accepted(self, reason):
        claim = ParsedClaim(claim_type="reject", reject_reason=reason)
        assert claim.reject_reason == reason

    def test_reject_pairing_is_still_enforced(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="reject")
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", reject_reason="incomplete")


class TestExpandPhaseKeepsTheOldSurface:

    def test_currency_still_accepted(self):
        assert ParsedClaim(claim_type="sec", currency="USD").currency == "USD"

    def test_a_full_gold_row_constructs_without_field_loss(self):
        gold = {"claim_type": "sec", "ticker": "JPM", "metric": "revenue",
                "operator": "eq", "value": 158100000000.0,
                "period": "fiscal 2024", "reject_reason": None}
        claim = ParsedClaim(**gold)
        for field, expected in gold.items():
            assert getattr(claim, field) == expected, field
