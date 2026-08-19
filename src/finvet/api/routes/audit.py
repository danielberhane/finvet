"""Audit trail and search history endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...audit import get_audit_logger
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
    }


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
