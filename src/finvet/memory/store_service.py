"""Claim memory backed by LangGraph PostgresStore."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore

from ..config.constants import MEMORY_SIMILAR_THRESHOLD
from ..utils.logging import get_logger

logger = get_logger(__name__)

NAMESPACE = ("claims",)


class ClaimMemoryItem(BaseModel):
    """Schema for writing verification episodes."""

    claim_text: str
    ticker: Optional[str] = None
    # Without this, recall cannot tell "Apple revenue FY2024" from "Apple net
    # income FY2024" beyond raw text similarity.
    metric: Optional[str] = None
    agent_type: Optional[str] = None
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    retrieved_value: Optional[float] = None
    summary: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    key_finding: Optional[str] = None
    verified_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ClaimMemoryMatch(BaseModel):
    """Schema for reading verification episodes."""

    claim: str
    verdict: str
    confidence: float
    similarity: float = Field(ge=0.0, le=1.0)
    agent: Optional[str] = None
    ticker: Optional[str] = None
    metric: Optional[str] = None
    retrieved_value: Optional[float] = None
    summary: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    key_finding: Optional[str] = None
    verified_at: Optional[str] = None
    request_id: str


class ClaimMemoryService:
    """Claim memory using LangGraph Store with Pydantic validation."""

    def __init__(self, store: BaseStore):
        self.store = store

    def store_claim(
        self,
        request_id: str,
        claim_text: str,
        ticker: Optional[str] = None,
        metric: Optional[str] = None,
        agent_type: Optional[str] = None,
        verdict: str = "",
        confidence: float = 0.0,
        retrieved_value: Optional[float] = None,
        summary: Optional[str] = None,
        tools_called: Optional[List[str]] = None,
        key_finding: Optional[str] = None,
    ) -> bool:
        """Store a completed verification episode."""
        if verdict in ("PENDING", "REJECTED", "ERROR", "UNKNOWN"):
            return False
        try:
            item = ClaimMemoryItem(
                claim_text=claim_text,
                ticker=ticker.upper() if ticker else None,
                metric=metric,
                agent_type=agent_type,
                verdict=verdict,
                confidence=confidence,
                retrieved_value=retrieved_value,
                summary=summary,
                tools_called=tools_called or [],
                key_finding=key_finding,
            )
            self.store.put(
                NAMESPACE,
                key=request_id,
                value=item.model_dump(),
                index=["claim_text"],
            )
            logger.info(f"Memory stored: {request_id} ({verdict})")
            return True
        except Exception as e:
            logger.warning(f"Memory store failed: {e}")
            return False

    def get_claim(self, request_id: str) -> Optional[dict]:
        """The stored episode for one request id, validated, or None.

        The verify route injects only what this returns. Reading the episode
        here rather than accepting it from the client is the whole point: the
        content is then something this system wrote, not something a caller
        supplied.
        """
        try:
            item = self.store.get(NAMESPACE, key=request_id)
        except Exception as e:
            logger.warning(f"Memory lookup failed for {request_id}: {e}")
            return None

        if item is None:
            return None

        try:
            validated = ClaimMemoryItem(**item.value)
        except Exception as e:
            logger.warning(f"Stored memory for {request_id} is malformed: {e}")
            return None

        return {
            "request_id": request_id,
            "claim": validated.claim_text,
            "verdict": validated.verdict,
            "confidence": validated.confidence,
            "summary": validated.summary,
            "verified_at": validated.verified_at,
        }

    def search_similar(
        self,
        claim_text: str,
        limit: int = 5,
        threshold: float = MEMORY_SIMILAR_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Search for similar past claims."""
        try:
            results = self.store.search(NAMESPACE, query=claim_text, limit=limit * 2)
            matches = []
            for item in results:
                if item.score >= threshold:
                    val = item.value
                    match = ClaimMemoryMatch(
                        claim=val["claim_text"],
                        verdict=val["verdict"],
                        confidence=val["confidence"],
                        similarity=round(item.score, 3),
                        agent=val.get("agent_type"),
                        ticker=val.get("ticker"),
                        metric=val.get("metric"),
                        retrieved_value=val.get("retrieved_value"),
                        summary=val.get("summary"),
                        tools_called=val.get("tools_called", []),
                        key_finding=val.get("key_finding"),
                        verified_at=val.get("verified_at"),
                        request_id=item.key,
                    )
                    matches.append(match.model_dump())
            return matches[:limit]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []
