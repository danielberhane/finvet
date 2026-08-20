"""Tests for reconciling the parser's reject fields before validation.

ParsedClaim requires claim_type and reject_reason to agree: a reason may be set
only on a reject, and a reject must carry one. DeepSeek breaks that pairing on a
small fraction of claims — it recognises a claim is unverifiable, populates
reject_reason, and leaves claim_type as "sec". Validation then throws,
claim_parser converts it to ParsingError, and /verify returns HTTP 500 with a
Pydantic stack trace.

Measured at 5 of 906 claims (~0.55%). The model's judgement was right each time;
only the field pairing was wrong, and the intent is unambiguous because
reject_reason is populated only when rejecting.

So the raw LLM output is reconciled at the boundary and the model itself stays
strict — the same division as the verdict override: the model supplies the
judgement, Python guarantees the shape.
"""

import pytest
from pydantic import ValidationError

from finvet.graph.nodes.claim_parser import reconcile_reject_fields
from finvet.models.claim import ParsedClaim


class TestReasonWithoutRejectType:
    """The live crash: a reason set on a non-reject claim."""

    def test_the_claim_that_returned_http_500(self):
        raw = {"claim_type": "sec", "ticker": None, "value": None, "period": None,
               "reject_reason": "incomplete"}
        assert reconcile_reject_fields(raw)["claim_type"] == "reject"

    @pytest.mark.parametrize("claim_type", ["sec", "market", "news"])
    def test_coerces_from_any_non_reject_type(self, claim_type):
        raw = {"claim_type": claim_type, "reject_reason": "non_financial"}
        assert reconcile_reject_fields(raw)["claim_type"] == "reject"

    def test_the_models_stated_reason_is_preserved(self):
        raw = {"claim_type": "sec", "reject_reason": "question"}
        assert reconcile_reject_fields(raw)["reject_reason"] == "question"

    def test_result_constructs_without_raising(self):
        raw = {"claim_type": "sec", "ticker": None, "value": None, "period": None,
               "reject_reason": "incomplete"}
        claim = ParsedClaim(**reconcile_reject_fields(raw))
        assert claim.claim_type == "reject"


class TestRejectTypeWithoutReason:
    """The mirror case, which raises the same way."""

    def test_missing_reason_is_filled(self):
        raw = {"claim_type": "reject", "reject_reason": None}
        assert reconcile_reject_fields(raw)["reject_reason"] == "unspecified"

    def test_absent_key_is_filled(self):
        assert reconcile_reject_fields({"claim_type": "reject"})["reject_reason"] == "unspecified"

    def test_result_constructs_without_raising(self):
        claim = ParsedClaim(**reconcile_reject_fields({"claim_type": "reject"}))
        assert claim.reject_reason == "unspecified"

    def test_unspecified_is_not_dressed_up_as_a_specific_reason(self):
        """Defaulting to "incomplete" would tell the user the claim was missing a
        ticker or value, which may simply be untrue. The honest default says the
        model gave no reason."""
        assert reconcile_reject_fields({"claim_type": "reject"})["reject_reason"] != "incomplete"


class TestConsistentInputIsUntouched:

    def test_a_well_formed_reject_passes_through(self):
        """claim_type and reason are untouched; since stage 04 the §8 contract
        also guarantees every companion field is explicitly null."""
        raw = {"claim_type": "reject", "reject_reason": "question"}
        data = reconcile_reject_fields(raw)
        assert data["claim_type"] == "reject"
        assert data["reject_reason"] == "question"
        for field in ("ticker", "metric", "operator", "value", "period"):
            assert data[field] is None

    def test_a_well_formed_claim_passes_through(self):
        raw = {"claim_type": "sec", "ticker": "AAPL", "value": 1.0, "reject_reason": None}
        assert reconcile_reject_fields(raw) == raw

    def test_the_input_dict_is_not_mutated(self):
        """The caller keeps the model's raw output for the audit trail."""
        raw = {"claim_type": "sec", "reject_reason": "incomplete"}
        reconcile_reject_fields(raw)
        assert raw["claim_type"] == "sec"


class TestModelStaysStrict:
    """Reconciliation happens at the LLM boundary. ParsedClaim keeps enforcing the
    invariant, so other callers cannot construct an inconsistent claim."""

    def test_reason_without_reject_still_rejected_by_the_model(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", reject_reason="incomplete")

    def test_reject_without_reason_still_rejected_by_the_model(self):
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="reject")
