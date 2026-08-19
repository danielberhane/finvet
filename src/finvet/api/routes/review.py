"""HITL review endpoints."""

from fastapi import APIRouter, HTTPException

from ...audit import get_audit_logger
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
    """
    Submit a human review decision and resume the paused verification graph.

    The graph was interrupted at hitl_checkpoint during /verify. This endpoint
    injects the human decision into the graph state and resumes execution.
    """
    logger.info(f"HITL review submitted for {request_id}: {review.decision}")

    if review.decision not in ("approve", "override", "reject"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision: {review.decision}. Must be 'approve', 'override', or 'reject'"
        )

    if review.decision == "override" and review.override_verdict not in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid override_verdict: {review.override_verdict}"
        )

    audit = get_audit_logger()
    audit.log_event(
        event_type=f"hitl_{review.decision}",
        request_id=request_id,
        data={
            "decision": review.decision,
            "override_verdict": review.override_verdict,
            "reviewer_notes": review.reviewer_notes,
        },
    )

    config = {"configurable": {"thread_id": request_id}}

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

        final_response = result.get("final_response", {})
        final_verdict = final_response.get("verdict", "ERROR")
        final_confidence = final_response.get("confidence", 0.0)

    except Exception as e:
        logger.warning(
            f"Could not resume graph for {request_id} ({e}). "
            f"Falling back to direct verdict computation."
        )
        final_response = None

        if review.decision == "approve":
            execution = audit.get_execution(request_id)
            if execution and execution.get("full_trace"):
                prelim = execution["full_trace"].get("final_response", {}).get("preliminary_analysis", {})
                final_verdict = prelim.get("verdict", "NOT_ENOUGH_INFO")
                final_confidence = prelim.get("confidence", 0.5)
            else:
                final_verdict = "NOT_ENOUGH_INFO"
                final_confidence = 0.5
        elif review.decision == "override":
            final_verdict = review.override_verdict
            final_confidence = 0.95
        else:
            final_verdict = "REJECTED"
            final_confidence = 1.0

    audit.update_execution_verdict(request_id, final_verdict, final_confidence)

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
