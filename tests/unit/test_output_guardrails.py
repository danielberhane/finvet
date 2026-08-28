"""Only a source disagreement sends a claim to a person.

The A2A escalation matrix, stated exhaustively. A delegation can end in nine
ways and exactly one of them is a conflict between two sources: `CONTRADICTS`.
Everything else is either an absence of evidence or a failure to look, and
neither is a reason to spend a reviewer's attention.

The status this replaces, `UNDISCLOSED_MATERIAL_CLAIM`, escalated on filing
silence about a fine or settlement. Reaching it required deciding that the
issuer *should* have disclosed the amount — a materiality judgment with no
calibration behind it, made by comparing a metric name against a set. Release A
does not make that judgment, so the status is gone rather than merely unused.

Filing silence is not contradiction: a periodic report omits most things, and a
report filed before an event was never going to mention it.
"""

import pytest

from finvet.graph.nodes.output_guardrails import output_guardrails
from finvet.models.a2a import (
    A2A_CONTRADICTS,
    A2A_CORROBORATES,
    A2A_FOUND_UNCERTIFIED,
    A2A_FAILED,
    A2A_NO_CORPUS,
    A2A_NO_MATCHING_DISCLOSURE,
    A2A_NOT_APPLICABLE_YET,
    A2A_PENDING_CLASSIFICATION,
    A2A_SOURCE_UNAVAILABLE,
    A2AStatus,
)


def _state(status=None, *, verdict="NOT_ENOUGH_INFO", confidence=0.9):
    state = {
        "request_id": "req_guard",
        "agent_evidence": {"verdict": verdict, "confidence": confidence,
                           "reasoning": "some reasoning"},
        "verdict": verdict,
        "confidence": confidence,
    }
    if status is not None:
        state["corroboration_result"] = {
            "status": status, "source_agent": "news", "target_agent": "sec",
            "verdict": "NOT_ENOUGH_INFO", "metric": "fine_amount",
            "claimed_value": 5e8,
        }
    return state


def _triggers(state):
    return output_guardrails(state).get("hitl_triggers", []) or []


class TestOnlyContradictionEscalates:

    def test_a_contradiction_sends_the_claim_to_a_person(self):
        assert "source_disagreement" in _triggers(_state(A2A_CONTRADICTS))

    @pytest.mark.parametrize("status", [
        A2A_CORROBORATES,
        A2A_NO_MATCHING_DISCLOSURE,
        A2A_FOUND_UNCERTIFIED,
        A2A_NOT_APPLICABLE_YET,
        A2A_SOURCE_UNAVAILABLE,
        A2A_NO_CORPUS,
        A2A_FAILED,
        A2A_PENDING_CLASSIFICATION,
    ])
    def test_no_other_status_escalates(self, status):
        """Silence, an unreachable source, an empty corpus and a failed
        delegation are all absences of evidence. None is a conflict."""
        assert "source_disagreement" not in _triggers(_state(status))

    def test_the_matrix_covers_every_status(self):
        """A status added later must be classified deliberately, not inherit
        'does not escalate' by being forgotten here."""
        import typing

        declared = set(typing.get_args(A2AStatus))
        covered = {
            A2A_CONTRADICTS, A2A_CORROBORATES, A2A_NO_MATCHING_DISCLOSURE,
            # Passages were read and no amount could be extracted from them.
            # An absence of certification, not a conflict between sources.
            A2A_FOUND_UNCERTIFIED,
            A2A_NOT_APPLICABLE_YET, A2A_SOURCE_UNAVAILABLE, A2A_NO_CORPUS,
            A2A_FAILED, A2A_PENDING_CLASSIFICATION,
        }
        assert declared == covered, (
            f"unclassified statuses: {declared ^ covered}")

    def test_no_corroboration_at_all_does_not_escalate(self):
        assert "source_disagreement" not in _triggers(_state(None))

    def test_a_malformed_result_does_not_escalate(self):
        state = _state(None)
        state["corroboration_result"] = "not a dict"
        assert "source_disagreement" not in _triggers(state)


class TestTheMaterialityTriggerIsGone:
    """It was the only escalation that rested on a judgment rather than a
    comparison, and nothing calibrated it."""

    @pytest.mark.parametrize("status", [
        A2A_NO_MATCHING_DISCLOSURE, A2A_SOURCE_UNAVAILABLE, A2A_NO_CORPUS,
        A2A_FAILED, A2A_NOT_APPLICABLE_YET,
    ])
    def test_no_status_produces_an_unsupported_material_claim_trigger(self,
                                                                      status):
        assert "unsupported_material_claim" not in _triggers(_state(status))

    def test_the_status_no_longer_exists(self):
        import typing

        import finvet.models.a2a as a2a

        assert not hasattr(a2a, "A2A_UNDISCLOSED_MATERIAL_CLAIM")
        assert "UNDISCLOSED_MATERIAL_CLAIM" not in typing.get_args(A2AStatus)

    def test_silence_about_a_fine_is_still_only_silence(self):
        """The exact case the removed status escalated on: a material amount,
        a datable event, and a filing that says nothing."""
        state = _state(A2A_NO_MATCHING_DISCLOSURE)
        state["corroboration_result"]["temporal_scope"] = "claim_period"

        assert _triggers(state) == [] or "source_disagreement" not in _triggers(state)


class TestUnrelatedTriggersStillFire:
    """Removing one escalation must not disturb the others."""

    def test_low_confidence_still_escalates(self):
        assert "low_confidence" in _triggers(
            _state(A2A_CORROBORATES, confidence=0.10))

    def test_a_contradiction_and_low_confidence_both_appear(self):
        triggers = _triggers(_state(A2A_CONTRADICTS, confidence=0.10))
        assert "source_disagreement" in triggers
        assert "low_confidence" in triggers
