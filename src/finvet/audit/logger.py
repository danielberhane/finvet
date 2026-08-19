"""Audit logger service for tracking all verification events."""

import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from .database import AuditDatabase


class AuditLogger:
    """Service for logging audit events during verification.

    Events are scoped by request_id to prevent cross-request interleaving
    when multiple verifications run concurrently.
    """

    def __init__(self):
        """Initialize audit logger."""
        self.db = AuditDatabase()
        self._events: Dict[str, List[Dict[str, Any]]] = {}  # request_id -> events
        self._lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        request_id: str,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> str:
        """Log an audit event.

        Args:
            event_type: Type of event (e.g., "claim_parsed", "sec_agent_completed")
            request_id: Request this event belongs to
            data: Event data
            parent_event_id: Optional parent event ID
            agent: Optional agent name

        Returns:
            Generated event_id
        """
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat()

        event = {
            "event_id": event_id,
            "request_id": request_id,
            "parent_event_id": parent_event_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "agent": agent,
            "data": data,
        }

        # Add to request-scoped in-memory buffer
        with self._lock:
            if request_id not in self._events:
                self._events[request_id] = []
            self._events[request_id].append(event)

        # Write to database
        self.db.log_event(
            event_id=event_id,
            request_id=request_id,
            event_type=event_type,
            data=data,
            parent_event_id=parent_event_id,
            agent=agent,
        )

        return event_id

    def commit_execution(
        self,
        request_id: str,
        claim_text: str,
        verdict: Optional[str],
        confidence: Optional[float],
        agents_run: List[str],
        execution_time_ms: int,
        final_response: Optional[Dict[str, Any]] = None,
        data_sources: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Commit full execution trace.

        Args:
            request_id: Request identifier
            claim_text: Claim text
            verdict: Final verdict
            confidence: Final confidence
            agents_run: List of agents that executed
            execution_time_ms: Total execution time
            final_response: Optional final API response dict
            data_sources: Optional data source provenance dict

        Returns:
            True if committed successfully
        """
        with self._lock:
            events = self._events.pop(request_id, [])

        success = self.db.commit_execution(
            request_id=request_id,
            claim_text=claim_text,
            verdict=verdict,
            confidence=confidence,
            agents_run=agents_run,
            events=events,
            execution_time_ms=execution_time_ms,
            final_response=final_response,
            data_sources=data_sources,
        )

        return success

    def get_execution(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve execution trace.

        Args:
            request_id: Request identifier

        Returns:
            Execution record or None
        """
        return self.db.get_execution(request_id)

    def get_events(self, request_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a request.

        Args:
            request_id: Request identifier

        Returns:
            List of events
        """
        return self.db.get_events(request_id)

    def update_execution_verdict(self, request_id: str, verdict: str, confidence: float) -> bool:
        """Update verdict after HITL review so it leaves the pending queue."""
        return self.db.update_execution_verdict(request_id, verdict, confidence)

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all executions pending HITL review.

        Returns:
            List of pending review records
        """
        return self.db.get_pending_reviews()

    def search_by_data_source(self, source_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find executions that used a specific data source (xbrl, rag, a2a).

        Args:
            source_type: Data source key to search for
            limit: Maximum results to return

        Returns:
            List of matching execution records
        """
        return self.db.search_by_data_source(source_type, limit)

    def search_past_verifications(self, claim_text: str, days: int = 7) -> List[Dict[str, Any]]:
        """Search for past verifications of same claim.

        Args:
            claim_text: Claim to search for
            days: Look back this many days

        Returns:
            List of matching executions
        """
        return self.db.search_by_claim_hash(claim_text, days)

    def list_executions(
        self,
        verdict: Optional[str] = None,
        agent: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        data_source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List audit executions with optional filters.

        Args:
            verdict: Filter by verdict (SUPPORTS, REFUTES, NOT_ENOUGH_INFO, PENDING, REJECTED)
            agent: Filter by agent used (sec, market, news)
            date_from: ISO timestamp lower bound (inclusive)
            date_to: ISO timestamp upper bound (inclusive)
            data_source: Filter by data source key (xbrl, rag, a2a)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of execution dicts ordered by timestamp descending
        """
        return self.db.list_executions(
            verdict=verdict,
            agent=agent,
            date_from=date_from,
            date_to=date_to,
            data_source=data_source,
            limit=limit,
            offset=offset,
        )


# Global singleton
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
