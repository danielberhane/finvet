"""Input guardrails node — delegates to modular guard providers."""

from typing import Dict

from ...config.settings import settings
from ...guards.composite import CompositeGuardProvider
from ...guards.llama_guard import LlamaGuardProvider
from ...guards.regex import RegexGuardProvider
from ...models.audit import AuditEvent
from ...models.state import VerificationState
from ...utils.exceptions import GuardrailViolation
from ...utils.logging import get_logger

logger = get_logger(__name__)

# Build input guard chain at module load
_input_providers = [RegexGuardProvider()]
if settings.enable_llama_guard:
    _input_providers.append(LlamaGuardProvider())
_input_guard = CompositeGuardProvider(_input_providers)

# Llama Guard's S6 ("specialized advice") is a verifiability judgement, not a
# safety one — "should I buy AAPL?" is not a checkable claim. The parser owns
# that rejection (auditable HTTP 200), so an advice-ONLY verdict is passed
# through rather than raised. Any real unsafe category, alone or alongside S6,
# and every regex violation (injection/PII) still raises.
_ADVICE_ONLY_CATEGORIES = {"S6"}


def input_guardrails(state: VerificationState) -> Dict:
    """
    Execute input guardrails via the composite guard provider.

    Returns:
        Dictionary with claim_normalized, guard_result_input, guardrails_passed/failed,
        and audit_events.
    """
    claim_raw = state["claim_raw"]
    user_id = state.get("user_id", "anonymous")
    request_id = state.get("request_id", "unknown")

    result = _input_guard.classify_input(claim_raw)

    advice_only = (
        result.violation_type == "LLAMA_GUARD_UNSAFE"
        and bool(result.categories)
        and set(result.categories).issubset(_ADVICE_ONLY_CATEGORIES)
    )

    if not result.safe and not advice_only:
        logger.warning(
            f"Guardrail violation: {result.violation_type}",
            extra={"request_id": request_id, "violation_type": result.violation_type},
        )
        raise GuardrailViolation(
            f"Input guard failed: {result.violation_type}",
            violation_type=result.violation_type or "GUARD_VIOLATION",
            details={
                "categories": result.categories,
                "provider": result.provider,
                "latency_ms": result.latency_ms,
            },
        )

    if advice_only:
        logger.info(
            "Advice-seeking input (S6 only) passed to the parser for classification",
            extra={"request_id": request_id},
        )

    # Llama Guard does not scrub, so an advice-only passthrough carries no
    # scrubbed_text; fall back to the raw claim.
    claim_normalized = result.scrubbed_text or claim_raw

    audit_event = AuditEvent(
        event_type="input_received",
        user_id=user_id,
        request_id=request_id,
        data={
            "claim_raw": claim_raw,
            "claim_normalized": claim_normalized,
            "guard_provider": result.provider,
            "guard_flags": result.flags,
            "guard_latency_ms": result.latency_ms,
        },
    )

    logger.info(
        f"Input guardrails passed for request {request_id}",
        extra={"request_id": request_id, "latency_ms": result.latency_ms},
    )

    return {
        "claim_normalized": claim_normalized,
        "guard_result_input": result.model_dump(),
        "guardrails_passed": ["composite_guard"],
        "guardrails_failed": [],
        "audit_events": state.get("audit_events", []) + [audit_event],
    }
