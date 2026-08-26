"""A claim the system knows it cannot answer does not go to a human.

The capability contract's required failure behaviour for an unsupported
capability is `NOT_ENOUGH_INFO`, confidence no greater than the HITL threshold,
and a machine-readable limitation. The first and third were not being met.

`_unsupported_claim` declines Q4 derivation, macro metrics and anything else no
tool can serve, returning evidence with `verdict=NOT_ENOUGH_INFO` and a low
confidence. That low confidence then tripped `low_confidence` in
`output_guardrails`, so the claim was routed to review — and the client saw
`status="pending_review"`, `verdict="PENDING"`, with the limitation dropped on
the way.

There is nothing for a reviewer to weigh. No tool serves the metric, and no
amount of human attention changes that; the reviewer can only agree. Sending
these to the queue is the same mistake as escalating filing silence: it fills a
person's queue with items they cannot act on and teaches them to stop reading
the flag.

Confidence stays low — the system is not confident, and saying otherwise would
be worse. What changes is that a *declared limitation* is not a reason to ask
somebody, because the system already knows the answer is "cannot".
"""

import pytest

from finvet.graph.nodes.output_guardrails import output_guardrails


def _state(evidence, confidence=0.2):
    return {
        "request_id": "req_declined",
        "agent_evidence": evidence,
        "verdict": evidence.get("verdict", "NOT_ENOUGH_INFO"),
        "confidence": confidence,
    }


def _declined(limitation="unsupported_metric"):
    return {
        "agent": "news", "verdict": "NOT_ENOUGH_INFO", "confidence": 0.2,
        "reasoning": "No available tool serves the metric 'cpi_inflation'.",
        "limitation": limitation, "execution_status": "completed",
        "error": None, "tools_called": [],
    }


def _genuinely_uncertain():
    """Same low confidence, but an agent actually ran and found little."""
    return {
        "agent": "sec", "verdict": "NOT_ENOUGH_INFO", "confidence": 0.2,
        "reasoning": "The filing was ambiguous about the figure.",
        "execution_status": "completed", "error": None,
        "tools_called": ["get_income_statement"],
    }


class TestADeclaredLimitationDoesNotEscalate:

    def test_a_declined_claim_does_not_require_review(self):
        out = output_guardrails(_state(_declined()))

        assert out["hitl_required"] is False
        assert "low_confidence" not in (out.get("hitl_triggers") or [])

    @pytest.mark.parametrize("limitation", [
        "unsupported_q4_derivation",
        "unsupported_metric",
        "unresolved_period",
    ])
    def test_every_declared_limitation_terminates(self, limitation):
        out = output_guardrails(_state(_declined(limitation)))
        assert out["hitl_required"] is False

    def test_genuine_uncertainty_still_escalates(self):
        """The control, and the reason this is narrow. An agent that ran and
        could not decide is exactly what human review is for."""
        out = output_guardrails(_state(_genuinely_uncertain()))

        assert out["hitl_required"] is True
        assert "low_confidence" in out["hitl_triggers"]

    def test_an_unsafe_output_still_escalates_even_when_declined(self):
        """Safety is not waived by a limitation."""
        import importlib
        from unittest.mock import MagicMock, patch

        # nodes/__init__ re-exports the node function under its module's name.
        node = importlib.import_module("finvet.graph.nodes.output_guardrails")

        unsafe = MagicMock()
        unsafe.safe = False
        unsafe.violation_type = "financial_advice"
        unsafe.model_dump.return_value = {"safe": False}

        guard = MagicMock()
        guard.classify_output.return_value = unsafe
        with patch.object(node, "_output_guard", guard):
            out = output_guardrails(_state(_declined()))

        assert out["hitl_required"] is True
        assert "output_safety_violation" in out["hitl_triggers"]

    def test_a_contradiction_still_escalates_even_when_declined(self):
        from finvet.models.a2a import A2A_CONTRADICTS

        state = _state(_declined())
        state["corroboration_result"] = {
            "status": A2A_CONTRADICTS, "source_agent": "news",
            "target_agent": "sec", "verdict": "REFUTES"}

        out = output_guardrails(state)

        assert out["hitl_required"] is True
        assert "source_disagreement" in out["hitl_triggers"]


class TestTheLimitationReachesTheClient:
    """A machine-readable reason, not just prose in the explanation."""

    def _response(self, evidence):
        from finvet.graph.nodes.response_generator import response_generator
        from finvet.models.claim import ParsedClaim

        result = response_generator({
            "request_id": "req_declined",
            "claim_raw": "US CPI inflation was 3.1 percent",
            "parsed_claim": ParsedClaim(
                claim_type="news", ticker="US", metric="cpi_inflation",
                operator="eq", value=3.1),
            "agent_evidence": evidence,
            "verdict": "NOT_ENOUGH_INFO", "confidence": 0.2,
            "hitl_required": False,
        })
        return result["final_response"]

    def test_the_limitation_is_in_the_response_metadata(self):
        body = self._response(_declined("unsupported_metric"))

        assert body["metadata"]["limitation"] == "unsupported_metric", (
            "a client cannot distinguish 'we cannot answer this' from "
            "'we looked and found nothing' without it")

    def test_the_verdict_is_not_enough_info_not_pending(self):
        body = self._response(_declined())

        assert body["verdict"] == "NOT_ENOUGH_INFO"
        assert body["status"] == "success"

    def test_a_normal_run_carries_no_limitation(self):
        body = self._response(_genuinely_uncertain())
        assert body["metadata"].get("limitation") is None
