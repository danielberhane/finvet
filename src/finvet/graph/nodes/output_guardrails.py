"""Output guardrails node — confidence threshold + output safety guards."""

from typing import Dict

from ...config.settings import settings
from ...guards.composite import CompositeGuardProvider
from ...guards.financial import FinancialGuardProvider
from ...guards.llama_guard import LlamaGuardProvider
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
