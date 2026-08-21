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
        out = _run(_guard(False, ["S6"], "LLAMA_GUARD_UNSAFE"))
        assert out["claim_normalized"] == "x"
        assert out["guardrails_passed"] == ["composite_guard"]

    def test_safe_input_passes_as_before(self):
        out = _run(_guard(True))
        assert out["guardrails_passed"] == ["composite_guard"]


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
