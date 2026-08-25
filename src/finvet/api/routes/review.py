"""HITL review endpoints."""

from fastapi import APIRouter, HTTPException

from ...audit import get_audit_logger
from ...audit.callbacks import AuditCallbackHandler
from ...utils.logging import get_logger
from .. import deps
from ..models import HITLReviewRequest

logger = get_logger(__name__)

router = APIRouter()


@router.get("/reviews")
async def get_pending_reviews():
    """List all claims pending HITL review."""
    audit = get_audit_logger()
    return audit.get_pending_reviews()


@router.post("/review/{request_id}")
def submit_hitl_review(request_id: str, review: HITLReviewRequest):
    """Submit a human review decision and resume the paused verification graph.

    The graph was interrupted at hitl_checkpoint during verification. The
    decision is injected into the checkpointed state and the graph resumes, so
    the reviewed verdict is produced by the same pipeline that produced the
    pending one -- not recomputed from the request body.

    Ordering matters. The pending row is claimed *before* anything is logged or
    resumed, so a request for an unknown id, or a second reviewer arriving after
    the first, is turned away before it can write anything.
    """
    audit = get_audit_logger()

    # Step 1 of the transition: take the row, or find out why we cannot.
    claim = audit.claim_pending_review(request_id)
    if claim == "missing":
        raise HTTPException(
            status_code=404,
            detail=f"No claim awaiting review for request {request_id}")
    if claim == "conflict":
        raise HTTPException(
            status_code=409,
            detail=f"Request {request_id} is already reviewed or being reviewed")

    logger.info(f"HITL review claimed for {request_id}: {review.decision}")
    audit.log_event(
        event_type=f"hitl_{review.decision}",
        request_id=request_id,
        data={
            "decision": review.decision,
            "override_verdict": review.override_verdict,
            "reviewer_notes": review.reviewer_notes,
        },
    )

    # Same callback configuration as the initial verification, so the resumed
    # nodes reach the event stream instead of vanishing from the timeline.
    config = {
        "configurable": {"thread_id": request_id},
        "callbacks": [AuditCallbackHandler(audit, request_id)],
    }

    try:
        deps.verification_graph.update_state(
            config,
            {
                "hitl_decision": review.decision,
                "hitl_override_verdict": review.override_verdict,
                "hitl_reviewer_notes": review.reviewer_notes,
            },
        )
        result = deps.verification_graph.invoke(None, config)
    except Exception as e:
        # No fallback verdict. Computing one from the request body meant a
        # reviewer's override became the answer without any pipeline having
        # produced it -- and for a checkpoint that no longer exists, there is
        # nothing to review. MemorySaver checkpoints do not survive a restart,
        # which is the common way to reach this.
        logger.warning(f"Could not resume graph for {request_id}: {e}")
        audit.release_review_claim(request_id)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "checkpoint_unavailable",
                "message": (
                    "The paused verification is no longer resumable; pending "
                    "reviews do not survive an API restart. Re-run the claim."
                ),
            },
        )

    final_response = result.get("final_response") or {}
    final_verdict = final_response.get("verdict", "ERROR")
    final_confidence = final_response.get("confidence", 0.0)

    # One transaction: verdict, confidence, response, data sources, event count
    # and a recomputed checksum. Updating only the first two left the stored
    # response and digest describing the pending-era run.
    finalized = audit.finalize_review(
        request_id,
        verdict=final_verdict,
        confidence=final_confidence,
        final_response=final_response,
        data_sources=(final_response.get("metadata") or {}).get("data_sources"),
        events=audit.get_events(request_id) or [],
        review_decision=review.decision,
        reviewer_notes=review.reviewer_notes,
    )
    if not finalized:
        logger.error(f"Review finalization failed for {request_id}")
        raise HTTPException(
            status_code=500,
            detail="Review completed but could not be recorded.")

    logger.info(
        f"HITL review completed (request: {request_id}, "
        f"decision: {review.decision}, verdict: {final_verdict})"
    )

    response = {
        "status": "reviewed",
        "request_id": request_id,
        "decision": review.decision,
        "verdict": final_verdict,
        "confidence": final_confidence,
        "reviewer_notes": review.reviewer_notes,
    }
    if final_response:
        response["final_response"] = final_response
    return response
