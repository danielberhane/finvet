"""
Claim verification endpoint.

This is the main entry point for the FinVet pipeline. A claim comes in,
flows through the LangGraph verification graph (guardrails → parsing →
agent → consensus → output check), and either:
  - Returns a verdict immediately (SUPPORTS / REFUTES / NOT_ENOUGH_INFO)
  - Pauses for human review if confidence is too low (HITL flow)
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...audit import AuditCallbackHandler, get_audit_logger
from ...utils.exceptions import AuditPersistenceError, GuardrailViolation
from ..execution import (
    ExecutionFinalizer,
    begin_request,
    build_pending_response,
    store_completed_claim,
)
from ...utils.helpers import build_preliminary_analysis
from ...utils.logging import get_logger
from .. import deps
from ..models import VerifyClaimRequest

logger = get_logger(__name__)  # Terminal/console logging for debugging

router = APIRouter()


@router.post("/verify")
def verify_claim(request: VerifyClaimRequest):
    """
    Verify a financial claim.

    Returns a verdict immediately for normal claims.
    For HITL claims (low confidence), returns status="pending_review" — the graph
    is paused at the HITL checkpoint and waits for human review via POST /review/.
    """

    # Generate a unique ID for this request — used everywhere:
    # audit trail, HITL checkpointer thread_id, memory storage
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "anonymous"
    start_time = datetime.utcnow()

    logger.info(
        f"Verification request received (request: {request_id})",
        extra={"request_id": request_id, "user_id": user_id, "claim_length": len(request.claim)}
    )

    audit = get_audit_logger()

    finalizer = ExecutionFinalizer(
        audit=audit,
        request_id=request_id,
        claim_text=request.claim,
        started_at=start_time,
    )

    try:
        # Records the request's start and builds the state every node reads.
        initial_state = begin_request(
            audit,
            request_id=request_id,
            claim_text=request.claim,
            user_id=user_id,
            started_at=start_time,
            memory_context=request.memory_context,
        )

        # thread_id = the checkpointer save slot. If the graph pauses for HITL,
        # we resume it later using this same thread_id via POST /review/{request_id}
        config = {
            "configurable": {"thread_id": request_id},
            "callbacks": [AuditCallbackHandler(audit, request_id)],
        }

        # Run the full LangGraph pipeline:
        # input_guardrails → claim_parser → period_resolver → agent → consensus
        # → output_guardrails → (HITL checkpoint or response_generator)
        logger.info(f"Starting verification graph (request: {request_id})")
        result = deps.verification_graph.invoke(initial_state, config)

        # --- HITL interrupt: graph paused because confidence < 0.70 ---
        # The graph stopped at the hitl_checkpoint node. Return a "pending_review"
        # response so a human reviewer can make the final call.
        if result.get("hitl_required") and not result.get("hitl_checkpoint_passed"):
            return _handle_hitl_interrupt(request_id, request.claim, result, finalizer)

        # --- Normal completion: pipeline finished with a verdict ---
        final_response = result.get("final_response")

        if not final_response:
            logger.error(f"No final response generated (request: {request_id})")
            raise HTTPException(
                status_code=500,
                detail="Verification completed but no response was generated"
            )

        # Record the run before returning it. Shared with /verify-stream so
        # both routes end a request identically.
        finalizer.finish(
            status="success",
            state=result,
            final_response=final_response,
        )

        logger.info(
            f"Verification completed (request: {request_id}, "
            f"verdict: {final_response.get('verdict')}, "
            f"confidence: {final_response.get('confidence', 0):.2f})"
        )

        # --- Store result in episodic memory so agents can learn from past verifications ---
        # Non-critical: if this fails, the user still gets their verdict.
        if deps.claim_memory:
            try:
                parsed_claim = result.get("parsed_claim")
                agent_evidence = result.get("agent_evidence", {})
                deps.claim_memory.store_claim(
                    request_id=request_id,
                    claim_text=request.claim,
                    ticker=parsed_claim.ticker if parsed_claim else None,
                    metric=getattr(parsed_claim, "metric", None) if parsed_claim else None,
                    agent_type=agent_evidence.get("agent"),
                    verdict=final_response.get("verdict", ""),
                    confidence=final_response.get("confidence", 0),
                    retrieved_value=agent_evidence.get("retrieved_value"),
                    summary=final_response.get("summary"),
                    tools_called=agent_evidence.get("tools_called", []),
                    key_finding=agent_evidence.get("reasoning", "")[:200],
                )
            except Exception as e:
                logger.warning(f"Claim memory store failed (non-critical): {e}")

        return final_response

    # --- Exception handling for the main try (the entire pipeline invocation) ---

    except GuardrailViolation as gv:  # matches: main try around graph.invoke()
        # Input guardrail caught something (prompt injection, toxic content).
        # Log it in audit trail and re-raise — FastAPI returns the error to the user.
        audit.log_event(
            event_type="guardrail_violation",
            request_id=request_id,
            data={
                "violation_type": gv.violation_type,
                "details": gv.details,
            },
        )
        raise
    except AuditPersistenceError as ape:
        # The verdict exists but could not be recorded. Serving it would break
        # the guarantee that every released verdict is auditable.
        logger.error(f"Audit persistence failed (request: {request_id}): {ape}")
        raise HTTPException(
            status_code=503,
            detail="Verification could not be recorded; no verdict was issued.",
        )
    except HTTPException:  # matches: main try around graph.invoke()
        # Already a proper HTTP error (e.g., the 500 above). Just re-raise.
        raise
    except Exception as e:  # matches: main try around graph.invoke()
        # LangGraph wraps node exceptions in its own exception chain.
        # Walk the __cause__ chain to find if a GuardrailViolation is buried inside.
        cause = e
        while cause is not None:
            if isinstance(cause, GuardrailViolation):
                audit.log_event(
                    event_type="guardrail_violation",
                    request_id=request_id,
                    data={
                        "violation_type": cause.violation_type,
                        "details": cause.details,
                    },
                )
                raise cause
            cause = cause.__cause__
        # Truly unexpected error — log full stack trace and return 500
        logger.error(
            f"Verification failed (request: {request_id}): {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )
    finally:
        # Guardrail rejections and errors never commit, so nothing else
        # releases their buffered events.
        finalizer.release()


def _handle_hitl_interrupt(
    request_id: str,
    claim_text: str,
    result: dict,
    finalizer: ExecutionFinalizer,
) -> dict:
    """
    Called when the graph pauses at the HITL checkpoint (confidence < 0.70).

    Packages up everything the human reviewer needs (agent reasoning, tools called,
    why HITL was triggered) and saves a PENDING record in the audit trail.
    The graph stays paused — a human resumes it via POST /review/{request_id}.
    """
    agent_evidence = result.get("agent_evidence", {})
    hitl_triggers = result.get("hitl_triggers", [])  # e.g., ["confidence_below_threshold"]

    pending_response = build_pending_response(
        request_id,
        claim_text,
        result,
        preliminary_analysis=build_preliminary_analysis(result, agent_evidence),
    )

    # Same finalizer as every other terminal path — this record is PENDING, not
    # absent, so a reviewer can find the claim waiting for them.
    finalizer.finish(
        status="pending_review",
        state=result,
        final_response=pending_response,
    )

    logger.info(
        f"HITL interrupt: claim paused for review (request: {request_id}, "
        f"triggers: {hitl_triggers})"
    )

    return pending_response


@router.post("/verify-stream")
def verify_claim_stream(request: VerifyClaimRequest):
    """Verify a financial claim with streaming progress updates via SSE."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "anonymous"
    start_time = datetime.utcnow()

    audit = get_audit_logger()
    initial_state = begin_request(
        audit,
        request_id=request_id,
        claim_text=request.claim,
        user_id=user_id,
        started_at=start_time,
        memory_context=request.memory_context,
    )

    config = {
        "configurable": {"thread_id": request_id},
        "callbacks": [AuditCallbackHandler(audit, request_id)],
    }

    finalizer = ExecutionFinalizer(
        audit=audit,
        request_id=request_id,
        claim_text=request.claim,
        started_at=start_time,
    )

    def event_generator():
        try:
            final_result = {}
            for event in deps.verification_graph.stream(
                initial_state, config, stream_mode="updates"
            ):
                for node_name, updates in event.items():
                    yield f"data: {json.dumps({'type': 'progress', 'node': node_name, 'request_id': request_id})}\n\n"
                    final_result.update(updates)

            final_response = final_result.get("final_response")
            if final_response:
                # Commit before emitting the verdict. A client that has seen
                # "complete" believes the run is on record; if the write fails
                # it must see an error instead, not a verdict with no audit row.
                finalizer.finish(
                    status="success",
                    state=final_result,
                    final_response=final_response,
                )
                store_completed_claim(
                    deps.claim_memory,
                    request_id=request_id,
                    claim_text=request.claim,
                    final_response=final_response,
                    state=final_result,
                )
                yield f"data: {json.dumps({'type': 'complete', 'response': final_response})}\n\n"

            elif final_result.get("hitl_required"):
                # HITL interrupt — graph paused before hitl_checkpoint.
                # Build a pending_review response so the UI can redirect.
                agent_evidence = final_result.get("agent_evidence", {})
                pending_response = build_pending_response(
                    request_id,
                    request.claim,
                    final_result,
                    preliminary_analysis=build_preliminary_analysis(
                        final_result, agent_evidence),
                )
                finalizer.finish(
                    status="pending_review",
                    state=final_result,
                    final_response=pending_response,
                )
                yield f"data: {json.dumps({'type': 'complete', 'response': pending_response})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except AuditPersistenceError as ape:
            logger.error(f"Audit persistence failed (request: {request_id}): {ape}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Verification could not be recorded; no verdict was issued.'})}\n\n"
        except GuardrailViolation as gv:
            audit.log_event(
                event_type="guardrail_violation",
                request_id=request_id,
                data={"violation_type": gv.violation_type, "details": gv.details},
            )
            yield f"data: {json.dumps({'type': 'guardrail', 'response': {'error_code': gv.violation_type, 'error_message': str(gv)}})}\n\n"
        except Exception as e:
            # Check if a GuardrailViolation is wrapped inside
            cause = e
            while cause is not None:
                if isinstance(cause, GuardrailViolation):
                    audit.log_event(
                        event_type="guardrail_violation",
                        request_id=request_id,
                        data={"violation_type": cause.violation_type, "details": cause.details},
                    )
                    yield f"data: {json.dumps({'type': 'guardrail', 'response': {'error_code': cause.violation_type, 'error_message': str(cause)}})}\n\n"
                    return
                cause = cause.__cause__
            logger.error(f"Streaming failed (request: {request_id}): {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Guardrail rejections and errors never commit, so nothing else
            # releases their buffered events.
            finalizer.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
