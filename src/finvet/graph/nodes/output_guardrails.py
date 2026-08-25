"""Output guardrails node — confidence threshold + output safety guards."""

from typing import Dict

from ...config.settings import settings
from ...guards.composite import CompositeGuardProvider
from ...guards.financial import FinancialGuardProvider
from ...guards.llama_guard import LlamaGuardProvider
from ...models.a2a import (
    A2A_CONTRADICTS,
    A2A_UNDISCLOSED_MATERIAL_CLAIM,
)
from ...models.audit import AuditEvent
from ...models.state import VerificationState
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

    # Check 1: Low confidence
    threshold = settings.confidence_threshold_hitl
    if confidence < threshold:
        hitl_triggers.append("low_confidence")
        hitl_required = True

    # Check 2: Output safety guard
    agent_evidence = state.get("agent_evidence", {})
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

    # Check 3: the primary source undermines the claim. Two ways that happens,
    # and they escalate for different reasons.
    #
    # CONTRADICTS — both agents reached decisive, opposite verdicts. A primary
    # source contradicting the one the verdict rests on is a question for a
    # person, not a confidence score.
    #
    # UNDISCLOSED_MATERIAL_CLAIM — the claim asserted a fine or settlement, a
    # filing covering the period exists, and it does not mention it. Neither
    # agent can be decisive about a narrative amount (there is no XBRL concept
    # for a penalty), so disagreement is unreachable for exactly the claims the
    # delegation exists to check. An unsupported material assertion is the
    # reachable signal, and it is the one worth a reviewer's time.
    #
    # Plain NO_MATCHING_DISCLOSURE and NOT_APPLICABLE_YET still do not escalate:
    # a periodic filing is silent about most things, and one that closed before
    # the event was never going to mention it. Treating either as conflict would
    # route half the traffic to a reviewer and teach them to ignore the flag.
    corroboration = state.get("corroboration_result")
    corroboration_status = (corroboration.get("status")
                            if isinstance(corroboration, dict) else None)

    if corroboration_status == A2A_CONTRADICTS:
        hitl_triggers.append("source_disagreement")
        hitl_required = True
        logger.warning(
            f"Source disagreement: {corroboration.get('source_agent')} said "
            f"{state.get('verdict')}, {corroboration.get('target_agent')} said "
            f"{corroboration.get('verdict')} (request: {request_id})"
        )
    elif corroboration_status == A2A_UNDISCLOSED_MATERIAL_CLAIM:
        hitl_triggers.append("unsupported_material_claim")
        hitl_required = True
        logger.warning(
            f"Unsupported material claim: {corroboration.get('metric')} of "
            f"{corroboration.get('claimed_value')} is not disclosed in the "
            f"issuer's filing for the period (request: {request_id})"
        )

    # Audit event
    audit_event = AuditEvent(
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

    return {
        "hitl_required": hitl_required,
        "hitl_triggers": hitl_triggers,
        "guard_result_output": guard_result_dict,
        "audit_events": state.get("audit_events", []) + [audit_event],
    }
