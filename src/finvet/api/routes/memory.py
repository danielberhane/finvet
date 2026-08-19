"""Memory check and accept endpoints."""

from fastapi import APIRouter

from ...audit import get_audit_logger
from ...config.constants import MEMORY_CACHE_THRESHOLD
from ...utils.logging import get_logger
from .. import deps
from ..models import MemoryCheckRequest, MemoryAcceptRequest

logger = get_logger(__name__)

router = APIRouter()


@router.post("/memory-check")
def memory_check(request: MemoryCheckRequest):
    """Search for similar past verifications (pre-pipeline memory check)."""
    if not deps.claim_memory:
        return {"matches": []}
    try:
        matches = deps.claim_memory.search_similar(
            claim_text=request.claim,
            limit=3,
            threshold=MEMORY_CACHE_THRESHOLD,
        )
        return {"matches": matches}
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return {"matches": []}


@router.post("/memory-accept")
def memory_accept(request: MemoryAcceptRequest):
    """Log that the user accepted a cached verification result."""
    audit = get_audit_logger()
    audit.log_event(
        event_type="memory_cache_accepted",
        request_id=request.original_request_id,
        data={
            "claim_submitted": request.claim,
            "similarity": request.similarity,
            "user_decision": "accept",
        },
    )
    return {"status": "logged", "original_request_id": request.original_request_id}
