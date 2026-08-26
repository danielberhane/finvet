"""Audit trail and search history endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...audit import get_audit_logger
from ...audit.integrity import (
    CHECKSUM_ALGORITHM,
    CHECKSUM_SCOPE,
    compare_execution_projection,
    verify_execution_checksum,
)
from ..models import VerifyClaimRequest

router = APIRouter()


@router.get("/audit")
async def list_audit_executions(
    verdict: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    data_source: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List audit executions with optional filters, ordered by timestamp descending."""
    audit = get_audit_logger()
    items = audit.list_executions(
        verdict=verdict,
        agent=agent,
        date_from=date_from,
        date_to=date_to,
        data_source=data_source,
        limit=limit,
        offset=offset,
    )
    return {"total": len(items), "items": items}


@router.get("/audit/{request_id}")
async def get_audit_trail(request_id: str):
    """Retrieve full audit trail for a verification request."""
    audit = get_audit_logger()
    execution = audit.get_execution(request_id)

    if not execution:
        raise HTTPException(status_code=404, detail=f"No audit trail found for request {request_id}")

    events = audit.get_events(request_id)

    return {
        "request_id": request_id,
        "execution": execution,
        "events": events,
        "total_events": len(events),
        "integrity": _integrity_for(execution, events),
    }


# Verdict values that are review-lifecycle state rather than an outcome. A row
# in one of these is mid-transition: `claim_pending_review` writes REVIEWING
# with a single atomic UPDATE (that atomicity is what makes the reviewer race
# safe) and the review route logs its hitl_* event straight to the database.
# Both are intended, and neither is covered by the committed envelope, so the
# projection comparison cannot speak for such a row. The checksum still can.
NON_TERMINAL_REVIEW_VERDICTS = {
    "REVIEWING": "review_in_progress",
    "REVIEW_FINALIZATION_FAILED": "review_finalization_failed",
}


def _integrity_for(execution: dict, events: list) -> dict:
    """Recompute the stored checksum, then compare the whole displayed record.

    Verification happens here, once, server side. The UI used to decide an
    execution was "Verified" because the hash column was non-empty, which is
    not a check -- it reported the presence of a string.

    Order matters. The checksum runs before any lifecycle reasoning: it is the
    only step that detects a changed snapshot, and skipping it for a row under
    review would let a genuinely corrupted record report "not checked" instead
    of "failed". Lifecycle state excuses the projection comparison, never the
    checksum.

    `scope` is returned so the claim is legible rather than implied: this
    covers the stored snapshot only, and it cannot resist a privileged writer
    who updates the data and the checksum together.
    """
    base = {"algorithm": CHECKSUM_ALGORITHM, "scope": CHECKSUM_SCOPE}
    envelope = execution.get("full_trace")
    stored = execution.get("execution_hash")

    if not isinstance(envelope, dict) or not stored:
        return {**base, "status": "unavailable",
                "reason": "no_checksum_recorded", "mismatches": []}

    if not verify_execution_checksum(envelope, stored):
        return {**base, "status": "failed", "reason": "checksum_mismatch",
                "mismatches": ["execution_hash"]}

    reason = NON_TERMINAL_REVIEW_VERDICTS.get(execution.get("verdict"))
    if reason:
        return {**base, "status": "unavailable", "reason": reason,
                "mismatches": []}

    mismatches = compare_execution_projection(execution, events)
    if mismatches:
        return {**base, "status": "failed", "reason": "projection_mismatch",
                "mismatches": mismatches}

    return {**base, "status": "verified", "reason": None, "mismatches": []}


@router.post("/search-history")
async def search_history(request: VerifyClaimRequest):
    """Search for past verifications of the same/similar claim."""
    audit = get_audit_logger()
    past_verifications = audit.search_past_verifications(request.claim, days=7)

    return {
        "claim": request.claim,
        "found": len(past_verifications),
        "verifications": past_verifications,
    }
