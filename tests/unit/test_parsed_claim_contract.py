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


class TestMalformedRangesAreRejectedNotRewritten:
    """The parser rewrote a malformed band into a different question.

    "between $100B and $50B" had its bounds swapped, and "between ..." with no
    bounds became `approx` around a midpoint. Both then produced a decisive
    verdict about an interval the claimant never stated. When the stated band
    is unusable the honest outcome is to decline the claim, not to guess which
    band was meant.
    """

    def _normalize(self, **overrides):
        from finvet.graph.nodes.claim_parser import normalize_parser_output

        raw = {"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",
               "operator": "range", "value": 75.0, "period": "FY2024"}
        raw.update(overrides)
        return normalize_parser_output(raw, "a claim about a band")

    def test_inverted_bounds_are_rejected(self):
        data, decisions = self._normalize(range_min=100.0, range_max=50.0)
        assert data["claim_type"] == "reject", (
            "an inverted band was silently swapped into a different claim")
        assert decisions["range"] == "inverted_bounds_rejected"

    def test_a_range_without_bounds_is_rejected(self):
        data, decisions = self._normalize(range_min=None, range_max=None)
        assert data["claim_type"] == "reject", (
            "a bandless range became approx, answering a point question")
        assert "operator" not in data, "a reject carries nothing but its reason"
        assert decisions["range"] == "range_without_bounds_rejected"

    def test_a_half_open_range_is_rejected(self):
        data, _ = self._normalize(range_min=50.0, range_max=None)
        assert data["claim_type"] == "reject"

    def test_a_well_formed_band_survives(self):
        data, decisions = self._normalize(range_min=50.0, range_max=100.0)
        assert data["claim_type"] == "sec"
        assert (data["range_min"], data["range_max"]) == (50.0, 100.0)
        assert decisions["range"] == "none"

    def test_bounds_without_a_range_operator_are_still_dropped(self):
        """Not a malformed band -- a non-range claim that carries stray
        bounds. Dropping them removes an interval nobody asserted."""
        data, decisions = self._normalize(operator="eq", range_min=50.0,
                                          range_max=100.0)
        assert data["claim_type"] == "sec"
        assert data["range_min"] is None and data["range_max"] is None
        assert decisions["range"] == "dropped_bounds_without_range"

    def test_a_rejected_band_reaches_the_claim_as_a_rejection(self):
        """Through the real node: the rejection must survive construction of
        the ParsedClaim, not raise a validation error deeper in."""
        import json
        from unittest.mock import MagicMock, patch

        # nodes/__init__ re-exports the node function under its module's own
        # name, so both `from ... import claim_parser` and a dotted import
        # bind the function. import_module returns the module itself.
        import importlib

        parser_mod = importlib.import_module(
            "finvet.graph.nodes.claim_parser")

        raw = {"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",
               "operator": "range", "value": 75.0, "period": "FY2024",
               "range_min": 100.0, "range_max": 50.0}

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=json.dumps(raw))
        with patch.object(parser_mod, "create_llm", return_value=llm), \
             patch.object(parser_mod, "get_audit_logger", return_value=MagicMock()):
            result = parser_mod.claim_parser(
                {"claim_raw": "revenue was between $100B and $50B",
                 "request_id": "req_000000000003"})

        parsed = result.get("parsed_claim")
        assert parsed is not None
        assert parsed.claim_type == "reject", (
            f"a malformed band produced a verifiable claim: {parsed}")
