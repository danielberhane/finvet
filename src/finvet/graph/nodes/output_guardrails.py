"""Output guardrails node — confidence threshold + output safety guards."""

from typing import Dict

from ...config.settings import settings
from ...guards.composite import CompositeGuardProvider
from ...guards.financial import FinancialGuardProvider
from ...guards.llama_guard import LlamaGuardProvider
from ...models.a2a import A2A_CONTRADICTS
from ...models.state import VerificationState
from ...audit import get_audit_logger
from ...utils.logging import get_logger

logger = get_logger(__name__)

# Build output guard chain at module load
_output_providers = []
if settings.enable_llama_guard:
    _output_providers.append(LlamaGuardProvider())
_output_providers.append(FinancialGuardProvider())
_output_guard = CompositeGuardProvider(_output_providers)


def output_guardrails(state: VerificationState) -> Dict:
    """
    Check if verification results require human review (HITL).

    Two checks:
    1. Confidence below threshold → HITL
    2. Output guard (financial advice / Llama Guard S6) → HITL
    """
    request_id = state.get("request_id", "unknown")
    verdict = state.get("verdict")
    confidence = state.get("confidence", 0.5)

    if not verdict:
        logger.warning(f"No verdict in state (request: {request_id})")
        return {
            "hitl_required": False,
            "hitl_triggers": [],
        }

    logger.info(f"Output guardrails checking (request: {request_id})")

    hitl_triggers = []
    hitl_required = False

    # Check 1: Low confidence.
    #
    # Unless the pipeline already knows why. `_unsupported_claim` declines a
    # claim before any agent runs -- Q4 derivation, a metric no tool serves, a
    # period this route cannot resolve -- and returns evidence carrying a
    # `limitation`. Its confidence is deliberately low, which used to trip this
    # check and send the claim to review.
    #
    # There is nothing there for a reviewer to weigh. No tool serves the
    # metric, and no amount of attention changes that; they can only agree.
    # Queueing these is the same mistake as escalating filing silence: it fills
    # a person's queue with items they cannot act on and teaches them to stop
    # reading the flag. The confidence stays low, because the system is not
    # confident -- what changes is that it does not ask.
    agent_evidence = state.get("agent_evidence") or {}
    declined_with_reason = bool(agent_evidence.get("limitation"))

    threshold = settings.confidence_threshold_hitl
    if confidence < threshold and not declined_with_reason:
        hitl_triggers.append("low_confidence")
        hitl_required = True

    # Check 2: Output safety guard. Safety is never waived by a limitation.
    reasoning_text = agent_evidence.get("reasoning", "") if agent_evidence else ""
    claim_raw = state.get("claim_raw", "")

    guard_result = _output_guard.classify_output(reasoning_text, claim_raw)
    guard_result_dict = guard_result.model_dump()

    if not guard_result.safe:
        hitl_triggers.append("output_safety_violation")
        hitl_required = True
        logger.warning(
            f"Output guard triggered: {guard_result.violation_type} (request: {request_id})"
        )

    # Check 3: the primary source contradicts the claim.
    #
    # CONTRADICTS is the only delegation outcome that escalates: both agents
    # reached decisive, opposite verdicts, and a primary source contradicting
    # the one the verdict rests on is a question for a person, not a confidence
    # score.
    #
    # Every other status is an absence of evidence, not a conflict.
    # NO_MATCHING_DISCLOSURE and NOT_APPLICABLE_YET are silences -- a periodic
    # filing omits most things, and one that closed before the event was never
    # going to mention it. SOURCE_UNAVAILABLE, NO_CORPUS and FAILED are
    # failures to look at all. Treating any of them as conflict would route
    # much of the traffic to a reviewer and teach them to ignore the flag.
    #
    # An earlier UNDISCLOSED_MATERIAL_CLAIM branch escalated on filing silence
    # about a fine or settlement. Reaching it meant deciding the issuer should
    # have disclosed the amount -- a materiality judgment with nothing
    # calibrating it, made by testing a metric name against a set. Release A
    # does not make that judgment.
    corroboration = state.get("corroboration_result")
    corroboration_status = (corroboration.get("status")
                            if isinstance(corroboration, dict) else None)

    # Every contradiction is queued, including one the filing already settled.
    #
    # Suppressing those was tried and was wrong twice over. CONTRADICTS needs
    # the SEC side to be decisive, which for a numeric claim means it holds a
    # trusted observation -- which is precisely what makes the verdict get
    # adopted. So the exemption fired on every case there is, and the trigger
    # was unreachable again.
    #
    # The `declined_with_reason` exemption above does not transfer either. It
    # covers metrics no tool serves, where a reviewer has nothing to weigh.
    # Here two sources report different numbers for the same event, and that is
    # actionable: the press may be wrong, or the filing may be stale.
    if corroboration_status == A2A_CONTRADICTS:
        hitl_triggers.append("source_disagreement")
        hitl_required = True
        logger.warning(
            f"Source disagreement: {corroboration.get('source_agent')} said "
            f"{state.get('verdict')}, {corroboration.get('target_agent')} said "
            f"{corroboration.get('verdict')} (request: {request_id})"
        )

    # Audit event
    # log_event() persists; state["audit_events"] did not. Without this, a run
    # released after Llama Guard degraded to advice-only left no record that
    # semantic safety had been unavailable.
    get_audit_logger().log_event(
        event_type="output_guardrails_checked",
        request_id=request_id,
        data={
            "verdict": verdict,
            "confidence": confidence,
            "hitl_triggers": hitl_triggers,
            "hitl_required": hitl_required,
            "guard_result": guard_result_dict,
        },
    )

    if hitl_required:
        logger.info(f"HITL required: {hitl_triggers} (request: {request_id})")
    else:
        logger.info(f"All guardrails passed (request: {request_id})")

    # guard_result_output is not returned: nothing read it, and the
    # output_guardrails_checked event above already records the outcome.
    return {
        "hitl_required": hitl_required,
        "hitl_triggers": hitl_triggers,
    }
