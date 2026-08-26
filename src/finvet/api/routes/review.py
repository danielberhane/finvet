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

    Two phases, divided by the call to `invoke`:

        PENDING --claim--> REVIEWING --finalize--> <terminal verdict>
                               |
                               +-- failure before invoke --> PENDING
                               |
                               +-- invoke began, no durable outcome -->
                                             REVIEW_FINALIZATION_FAILED

    The divide is the *call*, not its return. Before it nothing of the graph
    has run, so the claim is safely reversible. From it onward the checkpoint
    may have advanced, and returning the row to PENDING would let a second
    reviewer resume a partially-executed graph.
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
    if claim != "claimed":
        # A storage failure. The row is not missing; the store is unreachable,
        # and a 404 would tell the reviewer their claim does not exist.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "audit_unavailable",
                "message": "The audit store is unavailable; no review was started.",
            },
        )

    try:
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

        # Same callback configuration as the initial verification, so the
        # resumed nodes reach the event stream instead of vanishing.
        config = {
            "configurable": {"thread_id": request_id},
            "callbacks": [AuditCallbackHandler(audit, request_id)],
        }

        # -- Phase A: reversible. Nothing of the graph has run yet. ----------
        try:
            deps.verification_graph.update_state(
                config,
                {
                    "hitl_decision": review.decision,
                    "hitl_override_verdict": review.override_verdict,
                    "hitl_reviewer_notes": review.reviewer_notes,
                },
            )
        except Exception as e:
            # No fallback verdict. Computing one from the request body meant a
            # reviewer's override became the answer without any pipeline having
            # produced it -- and for a checkpoint that no longer exists, there
            # is nothing to review. MemorySaver checkpoints do not survive a
            # restart, which is the common way to reach this. Nothing has run,
            # so returning the row to PENDING is safe and leaves it retryable.
            logger.warning(f"Could not resume graph for {request_id}: {e}")
            audit.release_review_claim(request_id)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "checkpoint_unavailable",
                    "message": (
                        "The paused verification is no longer resumable; "
                        "pending reviews do not survive an API restart. "
                        "Re-run the claim."
                    ),
                },
            )

        # -- Phase B: irreversible from the call itself, not from its return.
        #
        # A flag set after invoke() returns cannot protect anything: the
        # dangerous case is exactly the one where invoke advances the
        # checkpoint and then raises, so the flag is never set and the handler
        # releases a row whose graph has already moved. From here on the claim
        # is never returned to PENDING; a row with no durable outcome goes to
        # the explicit recovery state instead.
        try:
            result = deps.verification_graph.invoke(None, config)
        except Exception as e:
            logger.error(f"Resume failed after entering the graph for "
                         f"{request_id}: {e}")
            _mark_unfinalized(audit, request_id, review,
                              error_type="resume_failed",
                              checkpoint_has_final_state=False)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "review_finalization_failed",
                    "message": (
                        "The review was started but could not be completed. "
                        "It is recorded for reconciliation; retry finalization."
                    ),
                },
            )

        final_response = result.get("final_response") if isinstance(result, dict) else None
        if not isinstance(final_response, dict) or not final_response.get("verdict"):
            # No verdict was produced. The previous code defaulted to the
            # string "ERROR", finalized it, and returned status="reviewed" --
            # a completed review that no pipeline had produced.
            logger.error(f"Resumed graph produced no reviewed result for "
                         f"{request_id}")
            _mark_unfinalized(audit, request_id, review,
                              error_type="review_result_missing",
                              checkpoint_has_final_state=False)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "review_result_missing",
                    "message": "The resumed graph produced no reviewed result.",
                },
            )

        final_verdict = final_response["verdict"]
        final_confidence = final_response.get("confidence", 0.0)

        # The envelope is hashed, so it may not be assembled from a read that
        # failed. `get_events` collapses a database outage to [], which would
        # commit an empty trail whose checksum then verifies -- integrity
        # reporting "verified" over nothing. Read strictly and reconcile with
        # the buffer, and treat a failure here like any other post-invoke
        # failure: recoverable, never released as reviewed.
        try:
            reconciled_events = audit.events_for_finalization(request_id)
        except Exception as e:
            logger.error(f"Could not read the audit trail to finalize "
                         f"{request_id}: {e}")
            _mark_unfinalized(audit, request_id, review,
                              error_type="audit_read_failed",
                              checkpoint_has_final_state=True)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "review_finalization_failed",
                    "message": (
                        "The review completed but its audit trail could not "
                        "be read to record it. It is recorded for "
                        "reconciliation; retry finalization."
                    ),
                },
            )

        # One transaction: verdict, confidence, response, data sources, event
        # count and a recomputed checksum.
        finalized = audit.finalize_review(
            request_id,
            verdict=final_verdict,
            confidence=final_confidence,
            final_response=final_response,
            data_sources=(final_response.get("metadata") or {}).get("data_sources"),
            events=reconciled_events,
            review_decision=review.decision,
            reviewer_notes=review.reviewer_notes,
        )
        if not finalized:
            # The graph produced a real result; only the write failed. The
            # checkpoint still holds that result, so reconciliation can use it.
            logger.error(f"Review finalization failed for {request_id}")
            _mark_unfinalized(audit, request_id, review,
                              error_type="audit_persistence_failed",
                              checkpoint_has_final_state=True)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "review_finalization_failed",
                    "message": (
                        "The review completed but could not be recorded. "
                        "It is recorded for reconciliation; retry finalization."
                    ),
                },
            )

        logger.info(
            f"HITL review completed (request: {request_id}, "
            f"decision: {review.decision}, verdict: {final_verdict})"
        )

        return {
            "status": "reviewed",
            "request_id": request_id,
            "decision": review.decision,
            "verdict": final_verdict,
            "confidence": final_confidence,
            "reviewer_notes": review.reviewer_notes,
            "final_response": final_response,
        }
    finally:
        # Every exit, including every failure above. Without this the events
        # buffered under this request are retained for the process lifetime.
        audit.discard_buffer(request_id)


@router.post("/review/{request_id}/reconcile")
def reconcile_review(request_id: str):
    """Retry the audit write for a review that ran but was never recorded.

    Reads the outcome the checkpoint already holds and persists it. It never
    calls `update_state` or `invoke`: the human decided once and the graph ran
    once, so re-entering it would either repeat side effects or produce a
    second, different answer for the same decision. This is a write of an
    outcome that already exists, not another attempt to produce one.
    """
    audit = get_audit_logger()

    claim = audit.claim_review_finalization(request_id)
    if claim == "missing":
        raise HTTPException(
            status_code=404,
            detail=f"No review awaiting reconciliation for request {request_id}")
    if claim == "conflict":
        raise HTTPException(
            status_code=409,
            detail=(f"Request {request_id} is not awaiting reconciliation, or "
                    f"another operator is retrying it"))
    if claim != "claimed":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "audit_unavailable",
                "message": "The audit store is unavailable; nothing was retried.",
            },
        )

    restore = True
    try:
        recovery = ((audit.get_execution(request_id) or {})
                    .get("full_trace") or {}).get("review_recovery") or {}

        config = {
            "configurable": {"thread_id": request_id},
            "callbacks": [AuditCallbackHandler(audit, request_id)],
        }

        try:
            snapshot = deps.verification_graph.get_state(config)
            values = getattr(snapshot, "values", None) or {}
        except Exception as e:
            logger.warning(f"No checkpoint to reconcile for {request_id}: {e}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "checkpoint_unavailable",
                    "message": (
                        "The reviewed result is no longer in memory; "
                        "checkpoints do not survive an API restart."
                    ),
                },
            )

        final_response = values.get("final_response") if isinstance(values, dict) else None
        if not isinstance(final_response, dict) or not final_response.get("verdict"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "review_result_missing",
                    "message": (
                        "The checkpoint holds no reviewed result to record; "
                        "the claim must be re-run."
                    ),
                },
            )

        finalized = audit.finalize_review(
            request_id,
            verdict=final_response["verdict"],
            confidence=final_response.get("confidence", 0.0),
            final_response=final_response,
            data_sources=(final_response.get("metadata") or {}).get("data_sources"),
            events=audit.get_events(request_id) or [],
            # The reviewer decided once; that decision was persisted with the
            # recovery marker and is reused rather than asked for again.
            review_decision=recovery.get("review_decision", "approve"),
            reviewer_notes=recovery.get("reviewer_notes"),
        )
        if not finalized:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "review_finalization_failed",
                    "message": "The audit row could not be written; retry later.",
                },
            )

        restore = False
        logger.info(f"Review reconciled for {request_id}: "
                    f"{final_response['verdict']}")
        return {
            "status": "reviewed",
            "request_id": request_id,
            "decision": recovery.get("review_decision"),
            "verdict": final_response["verdict"],
            "confidence": final_response.get("confidence", 0.0),
            "reviewer_notes": recovery.get("reviewer_notes"),
            "final_response": final_response,
            "reconciled": True,
        }
    finally:
        if restore:
            # Claiming moved the row out of the recovery state. A retry that
            # did not finalize must put it back, or it sits in REVIEWING with
            # nobody working on it and nothing to draw an operator's eye.
            audit.restore_review_finalization_failed(request_id)
        audit.discard_buffer(request_id)


def _mark_unfinalized(audit, request_id: str, review: HITLReviewRequest, *,
                      error_type: str, checkpoint_has_final_state: bool) -> None:
    """Record that the graph ran but no audited outcome exists.

    Best effort by necessity: if the store is down, the row stays REVIEWING and
    the operator report is what surfaces it. Never raises into the caller's
    error path, which already has a failure to report.
    """
    try:
        marked = audit.mark_review_finalization_failed(
            request_id,
            error_type=error_type,
            review_decision=review.decision,
            reviewer_notes=review.reviewer_notes,
            checkpoint_has_final_state=checkpoint_has_final_state,
        )
        if not marked:
            logger.error(
                f"Could not mark {request_id} for reconciliation; it remains "
                f"REVIEWING and will appear in the stuck-review report")
    except Exception as exc:
        logger.error(f"Could not mark {request_id} for reconciliation: {exc}")
