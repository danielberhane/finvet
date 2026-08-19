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
from ...utils.exceptions import GuardrailViolation
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

    # Record the raw input in the PostgreSQL audit trail (compliance requirement)
    audit = get_audit_logger()
    audit.log_event(
        event_type="input_received",
        request_id=request_id,
        data={
            "claim_raw": request.claim,
            "user_id": user_id,
            "timestamp": start_time.isoformat(),
        },
    )

    try:
        # Build the initial state dict that flows through every node in the graph.
        # Each node reads what it needs and writes its own fields.
        initial_state = {
            "claim_raw": request.claim,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp_received": start_time.isoformat(),
            "execution_start_time": start_time.isoformat(),
            "audit_trail": [],
            "total_tokens_used": 0,
            "memory_context": request.memory_context,  # Prior result from /memory-check (if user chose "Verify With Context")
        }

        # If the user chose "Verify With Context" from /memory-check,
        # record that decision in the audit trail
        if request.memory_context:
            audit.log_event(
                event_type="memory_context_injected",
                request_id=request_id,
                data={
                    "user_decision": "with_context",
                    "prior_request_id": request.memory_context.get("request_id"),
                    "prior_similarity": request.memory_context.get("similarity"),
                },
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
            return _handle_hitl_interrupt(request_id, request.claim, result, start_time, audit)

        # --- Normal completion: pipeline finished with a verdict ---
        final_response = result.get("final_response")

        if not final_response:
            logger.error(f"No final response generated (request: {request_id})")
            raise HTTPException(
                status_code=500,
                detail="Verification completed but no response was generated"
            )

        end_time = datetime.utcnow()
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Map agent_type key ("sec"/"news"/"market") to display name for audit
        agents_run = []
        agent_type = result.get("agent_type")
        if agent_type:
            agents_run.append({"sec": "SEC", "news": "News", "market": "Market"}.get(agent_type, agent_type))

        # Save the completed execution to PostgreSQL audit trail.
        # This creates the audit_executions row with verdict, confidence,
        # execution time, and the SHA-256 execution_hash for tamper detection.
        audit.commit_execution(
            request_id=request_id,
            claim_text=request.claim,
            verdict=final_response.get("verdict"),
            confidence=final_response.get("confidence", 0.0),
            agents_run=agents_run,
            execution_time_ms=execution_time_ms,
            final_response=final_response,
            data_sources=final_response.get("metadata", {}).get("data_sources"),
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


def _handle_hitl_interrupt(
    request_id: str,
    claim_text: str,
    result: dict,
    start_time: datetime,
    audit,
) -> dict:
    """
    Called when the graph pauses at the HITL checkpoint (confidence < 0.70).

    Packages up everything the human reviewer needs (agent reasoning, tools called,
    why HITL was triggered) and saves a PENDING record in the audit trail.
    The graph stays paused — a human resumes it via POST /review/{request_id}.
    """
    agent_evidence = result.get("agent_evidence", {})
    hitl_triggers = result.get("hitl_triggers", [])  # e.g., ["confidence_below_threshold"]

    agent_type = agent_evidence.get("agent", "unknown")
    preliminary_analysis = build_preliminary_analysis(result, agent_evidence)

    # Build the response sent back to the user/UI showing the claim needs review
    pending_response = {
        "status": "pending_review",
        "request_id": request_id,
        "claim": claim_text,
        "verdict": "PENDING",
        "confidence": 0.0,
        "confidence_label": "PENDING",
        "summary": "This claim requires human review before a verdict can be provided.",
        "explanation": agent_evidence.get("reasoning", ""),
        "sources": [],
        "disclosures": ["Human review required for this claim"],
        "hitl_triggers": hitl_triggers,
        "preliminary_analysis": preliminary_analysis,
        "metadata": {
            "hitl_required": True,
            "hitl_triggers": hitl_triggers,
            "agent": agent_type,
            "tools_called": agent_evidence.get("tools_called", []),
        },
    }

    end_time = datetime.utcnow()
    execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

    # Map agent key to display name for audit record
    agents_run = []
    if agent_type and agent_type != "unknown":
        agents_run.append({"sec": "SEC", "news": "News", "market": "Market"}.get(agent_type, agent_type))

    # Save as PENDING in audit trail — this claim is waiting for human review
    audit.commit_execution(
        request_id=request_id,
        claim_text=claim_text,
        verdict="PENDING",
        confidence=0.0,
        agents_run=agents_run,
        execution_time_ms=execution_time_ms,
        final_response=pending_response,
        data_sources=pending_response.get("metadata", {}).get("data_sources"),
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
    audit.log_event(
        event_type="input_received",
        request_id=request_id,
        data={"claim_raw": request.claim, "user_id": user_id},
    )

    initial_state = {
        "claim_raw": request.claim,
        "user_id": user_id,
        "request_id": request_id,
        "timestamp_received": start_time.isoformat(),
        "execution_start_time": start_time.isoformat(),
        "audit_trail": [],
        "total_tokens_used": 0,
        "memory_context": request.memory_context,
    }

    config = {
        "configurable": {"thread_id": request_id},
        "callbacks": [AuditCallbackHandler(audit, request_id)],
    }

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
                # Store in episodic memory (non-critical)
                if deps.claim_memory:
                    try:
                        parsed_claim = final_result.get("parsed_claim")
                        agent_evidence = final_result.get("agent_evidence", {})
                        deps.claim_memory.store_claim(
                            request_id=request_id,
                            claim_text=request.claim,
                            verdict=final_response.get("verdict", ""),
                            confidence=final_response.get("confidence", 0),
                            ticker=parsed_claim.ticker if parsed_claim else None,
                            agent_type=agent_evidence.get("agent"),
                        )
                    except Exception:
                        pass

                yield f"data: {json.dumps({'type': 'complete', 'response': final_response})}\n\n"

            elif final_result.get("hitl_required"):
                # HITL interrupt — graph paused before hitl_checkpoint.
                # Build a pending_review response so the UI can redirect.
                agent_evidence = final_result.get("agent_evidence", {})
                hitl_triggers = final_result.get("hitl_triggers", [])
                pending_response = {
                    "status": "pending_review",
                    "request_id": request_id,
                    "claim": request.claim,
                    "verdict": "PENDING",
                    "confidence": 0.0,
                    "summary": "This claim requires human review.",
                    "explanation": agent_evidence.get("reasoning", ""),
                    "hitl_triggers": hitl_triggers,
                    "preliminary_analysis": build_preliminary_analysis(final_result, agent_evidence),
                    "metadata": {
                        "hitl_required": True,
                        "hitl_triggers": hitl_triggers,
                        "agent": agent_evidence.get("agent"),
                        "tools_called": agent_evidence.get("tools_called", []),
                    },
                }
                end_time = datetime.utcnow()
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
                agents_run = []
                agent_type = agent_evidence.get("agent")
                if agent_type:
                    agents_run.append({"sec": "SEC", "news": "News", "market": "Market"}.get(agent_type, agent_type))
                audit.commit_execution(
                    request_id=request_id,
                    claim_text=request.claim,
                    verdict="PENDING",
                    confidence=0.0,
                    agents_run=agents_run,
                    execution_time_ms=execution_time_ms,
                    final_response=pending_response,
                )
                yield f"data: {json.dumps({'type': 'complete', 'response': pending_response})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
