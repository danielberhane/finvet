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
    resolve_memory_context,
)
from ...utils.helpers import build_preliminary_analysis
from ...utils.logging import get_logger
from .. import deps
from ..models import VerifyClaimRequest

logger = get_logger(__name__)  # Terminal/console logging for debugging

router = APIRouter()

# A progress event is sent for every node of every run, to a browser. Only
# named, bounded values go in it.
_PROGRESS_STRING_LIMIT = 200


def _clip(value):
    """Bound a string that turns out to be prose."""
    if isinstance(value, str) and len(value) > _PROGRESS_STRING_LIMIT:
        return value[:_PROGRESS_STRING_LIMIT] + "…"
    return value


def _progress_detail(node_name: str, updates: dict) -> dict:
    """What a node produced, for the client to show while the run continues.

    The graph stream already carries each node's state delta and the route
    discarded it, yielding only the node's name. The pipeline therefore
    computed the parse, the resolved period and the deterministic comparison
    and told the client none of it — which is why the UI inferred the claim
    type from keywords in the raw text and captioned a spinner with a guess.

    An allow-list rather than the delta itself. `agent_evidence` carries
    `provenance` and `tool_calls_detail`, which hold whole filing excerpts;
    those belong in the reviewed final response, not on every step. Names and
    counts here, never bodies.
    """
    updates = updates or {}
    detail: dict = {}

    if node_name == "claim_parser":
        from ...graph.nodes.response_generator import _parsed_claim_view

        parsed = _parsed_claim_view(updates.get("parsed_claim"))
        if parsed:
            detail = {k: parsed.get(k) for k in
                      ("claim_type", "ticker", "metric", "operator", "value",
                       "period")}

    elif node_name == "period_resolver":
        period = updates.get("canonical_period")
        if period is not None:
            assumptions = getattr(period, "assumptions", None) or []
            detail = {
                "start": getattr(period, "start_date", None),
                "end": getattr(period, "end_date", None),
                "assumption": _clip(assumptions[0]) if assumptions else None,
            }

    elif node_name in ("sec_agent", "market_agent", "news_agent"):
        evidence = updates.get("agent_evidence") or {}
        observation = evidence.get("trusted_observation") or {}
        detail = {
            "agent": evidence.get("agent"),
            "tools": list(evidence.get("tools_called") or []),
            "retrieved_value": evidence.get("retrieved_value"),
            "magnitude_difference_percent": evidence.get(
                "magnitude_difference_percent"),
            "llm_original_verdict": evidence.get("llm_original_verdict"),
            "override_applied": evidence.get("override_applied"),
            "limitation": evidence.get("limitation"),
            "temporal_status": evidence.get("temporal_status"),
            "concept": observation.get("concept"),
            "period_end": observation.get("period_end"),
            "observed_at": observation.get("observed_at"),
            "rag_chunks": len(updates.get("rag_chunks_retrieved") or []),
            "a2a_status": (updates.get("corroboration_result") or {}).get(
                "status"),
        }

    elif node_name == "consensus":
        detail = {
            "verdict": updates.get("verdict"),
            "confidence": updates.get("confidence"),
            "confidence_label": updates.get("confidence_label"),
        }

    elif node_name == "output_guardrails":
        detail = {
            "hitl_required": updates.get("hitl_required"),
            "hitl_triggers": list(updates.get("hitl_triggers") or []),
        }

    # A node nobody has taught this function about says nothing, rather than
    # leaking whatever its delta happens to contain.
    return {k: _clip(v) for k, v in detail.items() if v is not None or k in
            ("value", "retrieved_value", "override_applied", "hitl_required")}


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
            memory_context=resolve_memory_context(
                deps.claim_memory, request.memory_context_request_id),
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

        # One coordinator decides what happened, records it, and hands back
        # the body to return. The route no longer labels outcomes itself --
        # every completed run used to be committed as "success", so a parser
        # rejection was stored as a successful verification.
        body = finalizer.finish_pipeline_outcome(
            result,
            result.get("final_response"),
            claim_memory=deps.claim_memory,
            preliminary_analysis=build_preliminary_analysis(
                result, result.get("agent_evidence", {})),
        )

        logger.info(
            f"Verification terminalized (request: {request_id}, "
            f"status: {body.get('status')}, verdict: {body.get('verdict')})"
        )
        return body

    # --- Exception handling for the main try (the entire pipeline invocation) ---

    except GuardrailViolation as gv:  # matches: main try around graph.invoke()
        # A refusal is a terminal outcome and belongs on the record: without
        # an execution row, the audit trail cannot show what was declined.
        audit.log_event(
            event_type="guardrail_violation",
            request_id=request_id,
            data={"violation_type": gv.violation_type, "details": gv.details},
        )
        finalizer.finish_guardrail(gv.violation_type, str(gv))
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
                finalizer.finish_guardrail(cause.violation_type, str(cause))
                raise cause
            cause = cause.__cause__
        # Truly unexpected error — log full stack trace and return 500
        logger.error(
            f"Verification failed (request: {request_id}): {str(e)}",
            exc_info=True
        )
        try:
            finalizer.finish_error(str(e))
        except AuditPersistenceError:
            # Auditing an audit failure would recurse. The 500 below still
            # tells the client the run did not succeed.
            logger.error(f"Could not record the failure of {request_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )
    finally:
        # Guardrail rejections and errors never commit, so nothing else
        # releases their buffered events.
        finalizer.release()


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
        memory_context=resolve_memory_context(
            deps.claim_memory, request.memory_context_request_id),
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
                    # `updates` was merged and discarded, so the client saw a
                    # node name and nothing else. _progress_detail sends the
                    # allow-listed part of what the node actually produced.
                    yield (
                        "data: "
                        + json.dumps({
                            "type": "progress",
                            "node": node_name,
                            "request_id": request_id,
                            "detail": _progress_detail(node_name, updates),
                        })
                        + "\n\n"
                    )
                    final_result.update(updates)

            # Same coordinator as /verify: it derives the status, records
            # the run, and returns the body to emit. Committing first is what
            # makes "on record before the client is told" true -- a client
            # that has seen a terminal event believes the run is stored.
            body = finalizer.finish_pipeline_outcome(
                final_result,
                final_result.get("final_response"),
                claim_memory=deps.claim_memory,
                preliminary_analysis=build_preliminary_analysis(
                    final_result, final_result.get("agent_evidence", {})),
            )

            if body.get("status") == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': body['explanation'], 'response': body})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'complete', 'response': body})}\n\n"

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
            body = finalizer.finish_guardrail(gv.violation_type, str(gv))
            yield f"data: {json.dumps({'type': 'guardrail', 'response': body})}\n\n"
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
                    body = finalizer.finish_guardrail(cause.violation_type,
                                                      str(cause))
                    yield f"data: {json.dumps({'type': 'guardrail', 'response': body})}\n\n"
                    return
                cause = cause.__cause__
            logger.error(f"Streaming failed (request: {request_id}): {e}")
            try:
                body = finalizer.finish_error(str(e))
            except AuditPersistenceError:
                # Auditing an audit failure would recurse.
                logger.error(f"Could not record the failure of {request_id}")
                body = {"status": "error", "explanation": str(e)}
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'response': body})}\n\n"
        finally:
            # Guardrail rejections and errors never commit, so nothing else
            # releases their buffered events.
            finalizer.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
