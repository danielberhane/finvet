"""Tests for the input_guardrails node's raise policy.

Safety belongs to the guards; "is this a verifiable claim?" belongs to the parser.
So an advice-only Llama Guard verdict (S6 alone) must NOT raise — it passes through
to the parser, which rejects it as advice_seeking and returns an auditable HTTP 200
rejection. Everything genuinely unsafe (a real S-category, alone or co-occurring with
S6) and every regex violation (injection, PII) still raises → HTTP 400.
"""

from unittest.mock import patch

import pytest

from finvet.graph.nodes.input_guardrails import input_guardrails
from finvet.guards.provider import GuardResult
from finvet.utils.exceptions import GuardrailViolation

_TARGET = "finvet.graph.nodes.input_guardrails._input_guard.classify_input"


def _run(result: GuardResult):
    with patch(_TARGET, return_value=result):
        return input_guardrails({"claim_raw": "x", "request_id": "t"})


def _guard(safe, categories=None, violation_type=None):
    return GuardResult(
        safe=safe,
        categories=categories or [],
        scrubbed_text="x",
        violation_type=violation_type,
        provider="composite",
    )


class TestAdviceIsNotBlocked:

    def test_s6_only_passes_through_to_the_parser(self):
        """Not blocking is the property. It used to be asserted via a
        `guardrails_passed` field in state that nothing ever read; the guard's
        outcome is on the audit trail via log_event, and the pipeline evidence
        that it passed is that the node returns normally with the claim."""
        out = _run(_guard(False, ["S6"], "LLAMA_GUARD_UNSAFE"))
        assert out["claim_normalized"] == "x"

    def test_safe_input_passes_as_before(self):
        out = _run(_guard(True))
        assert out["claim_normalized"] == "x"

    def test_the_guard_outcome_reaches_the_audit_trail(self):
        """Where the record actually lives, now that state carries no copy."""
        import importlib
        from unittest.mock import MagicMock, patch

        # nodes/__init__ re-exports the node function under its module's own
        # name, so a plain import binds the function. import_module returns
        # the module itself.
        node = importlib.import_module("finvet.graph.nodes.input_guardrails")

        audit = MagicMock()
        with patch.object(node, "get_audit_logger", lambda: audit):
            _run(_guard(False, ["S6"], "LLAMA_GUARD_UNSAFE"))

        data = audit.log_event.call_args.kwargs["data"]
        # `flags` and `categories` are separate fields on GuardResult; the
        # event carries flags. Asserting the keys reach the trail is the
        # point -- state no longer holds a copy of any of this.
        assert data["guard_provider"] == "composite"
        assert "guard_flags" in data
        assert data["claim_normalized"] == "x"


class TestUnsafeStillRaises:

    def test_a_real_unsafe_category_raises(self):
        with pytest.raises(GuardrailViolation):
            _run(_guard(False, ["S1"], "LLAMA_GUARD_UNSAFE"))

    def test_advice_co_occurring_with_an_attack_still_raises(self):
        """S6 alongside a genuinely unsafe category is not advice-only."""
        with pytest.raises(GuardrailViolation):
            _run(_guard(False, ["S6", "S10"], "LLAMA_GUARD_UNSAFE"))

    def test_regex_injection_still_raises(self):
        with pytest.raises(GuardrailViolation):
            _run(_guard(False, ["injection"], "INJECTION_DETECTED"))

    def test_pii_still_raises(self):
        with pytest.raises(GuardrailViolation):
            _run(_guard(False, [], "PII_DETECTED"))
