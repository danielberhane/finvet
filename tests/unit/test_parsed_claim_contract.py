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


class TestContractPhaseFieldsAreGone:
    """CONTRACT (2026-08-20): comparison and currency are deleted, and the
    model forbids unknown fields — a stray operator= kwarg must raise, not
    be silently ignored while operator lands as None. Legacy keys from a
    regressing LLM are the BOUNDARY's job to strip, not the model's to
    tolerate."""

    def test_comparison_is_no_longer_a_field(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", value=1.0, comparison="eq")

    def test_currency_is_no_longer_a_field(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", currency="USD")

    def test_unknown_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", frobnicate=1)

    @pytest.mark.parametrize("op", ["eq", "gt", "gte", "lt", "lte", "approx"])
    def test_point_operators_stand_alone(self, op):
        claim = ParsedClaim(claim_type="sec", value=1.0, operator=op)
        assert claim.operator == op

    def test_range_carries_its_band(self):
        """The seventh operator is an interval, so it needs both ends.

        It used to stand alone on a midpoint, which made membership untestable
        and refuted values plainly inside the stated band.
        """
        claim = ParsedClaim(claim_type="sec", value=1.0, operator="range",
                            range_min=0.5, range_max=1.5)
        assert (claim.range_min, claim.range_max) == (0.5, 1.5)

    def test_range_without_a_band_is_rejected(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", value=1.0, operator="range")


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


class TestGoldConstructsExactly:

    def test_a_full_gold_row_constructs_without_field_loss(self):
        gold = {"claim_type": "sec", "ticker": "JPM", "metric": "revenue",
                "operator": "eq", "value": 158100000000.0,
                "period": "fiscal 2024", "reject_reason": None}
        claim = ParsedClaim(**gold)
        for field, expected in gold.items():
            assert getattr(claim, field) == expected, field
