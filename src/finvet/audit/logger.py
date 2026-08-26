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
        self._last_write_ok = True

    def log_event(
        self,
        event_type: str,
        request_id: str,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None,
        agent: Optional[str] = None,
        buffer_for_execution: bool = True,
    ) -> str:
        """Log an audit event.

        Args:
            event_type: Type of event (e.g., "claim_parsed", "sec_agent_completed")
            request_id: Request this event belongs to
            data: Event data
            parent_event_id: Optional parent event ID
            agent: Optional agent name
            buffer_for_execution: Whether this event belongs to the run that is
                still in flight. False for anything recorded *after* that run
                was finalized -- accepting a cached result reuses the original
                request_id, and buffering it would leave events accumulating
                under a request nothing will ever commit again.

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
        if buffer_for_execution:
            with self._lock:
                if request_id not in self._events:
                    self._events[request_id] = []
                self._events[request_id].append(event)

        # Write to database, carrying the timestamp generated above. The row
        # and the buffered copy describe one event and must agree; the database
        # layer used to stamp its own, so the audit API's comparison of the
        # persisted trail against the hashed envelope mismatched every time.
        self._last_write_ok = self.db.log_event(
            event_id=event_id,
            request_id=request_id,
            event_type=event_type,
            data=data,
            timestamp=timestamp,
            parent_event_id=parent_event_id,
            agent=agent,
        )

        return event_id

    def log_event_persisted(self, **kwargs) -> bool:
        """Log an event and report whether the database write succeeded.

        For callers that must not claim an event was recorded when it was not.
        log_event returns an id whether or not the row landed, which is fine
        while a later commit_execution reconciles the buffer -- and wrong for a
        post-hoc event, where nothing else will.
        """
        self.log_event(**kwargs)
        return bool(self._last_write_ok)

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
        terminal_status: Optional[str] = None,
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
        # Copy, do not pop: a failed write must leave the buffer intact so the
        # caller can retry or discard deliberately. Popping first meant a
        # database outage destroyed the only in-memory copy of the trace.
        with self._lock:
            events = list(self._events.get(request_id, []))

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
            terminal_status=terminal_status,
        )

        if success:
            self.discard_buffer(request_id)

        return success

    def discard_buffer(self, request_id: str) -> None:
        """Release a request's buffered events.

        Called after a successful commit, and explicitly by a caller that has
        given up on persisting the run. Every terminal path must reach one or
        the other, or the buffer is retained for the process lifetime.
        """
        with self._lock:
            self._events.pop(request_id, None)

    def buffer_size(self, request_id: str) -> int:
        """Number of events currently buffered for a request."""
        with self._lock:
            return len(self._events.get(request_id, []))

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

    def events_for_finalization(self, request_id: str) -> List[Dict[str, Any]]:
        """Every event the run produced, for the envelope that gets hashed.

        Two failures the plain `get_events` cannot express, both of which used
        to end with a review finalized over an incomplete trail:

        1. The read itself fails. `get_events` returns [] and the caller cannot
           tell that from a run with no events, so the execution committed with
           an empty `full_trace.events` -- and the checksum, computed over that
           empty envelope, then verified. A wiped trail reported as intact.
        2. An event's immediate write failed. `log_event` is best effort and
           keeps a buffered copy; re-reading the database cannot see what never
           landed. `commit_execution` reconciles from the buffer for exactly
           this reason, and the review path did not.

        So the read is strict, and the buffer fills the gaps. Ordering matches
        the database's own (timestamp, event_id) contract so the envelope and
        the queried trail agree.
        """
        persisted = self.db.get_events_strict(request_id)

        with self._lock:
            buffered = list(self._events.get(request_id, []))

        merged = {
            event["event_id"]: event
            for event in list(buffered) + list(persisted)
            if event.get("event_id")
        }
        return sorted(merged.values(),
                      key=lambda e: (e.get("timestamp") or "",
                                     e.get("event_id") or ""))

    def claim_pending_review(self, request_id: str) -> str:
        """Take ownership of a pending review.

        "claimed", "conflict", "missing", or "unavailable" -- the last for a
        storage failure, which is not the same as no such row.
        """
        return self.db.claim_pending_review(request_id)

    def release_review_claim(self, request_id: str) -> bool:
        """Return a claimed row to PENDING after a *pre-invoke* failure only.

        Safe while the graph has not been entered. Once invoke has been called
        the checkpoint may have advanced, and releasing would let a second
        reviewer resume a partially-executed run; use
        mark_review_finalization_failed there instead.
        """
        return self.db.release_review_claim(request_id)

    def mark_review_finalization_failed(self, request_id: str, **kwargs) -> bool:
        """Move a claimed row to the explicit recovery state."""
        return self.db.mark_review_finalization_failed(request_id, **kwargs)

    def claim_review_finalization(self, request_id: str) -> str:
        """Take ownership of a stuck review so it can be retried."""
        return self.db.claim_review_finalization(request_id)

    def restore_review_finalization_failed(self, request_id: str) -> bool:
        """Put a failed retry back where an operator will find it."""
        return self.db.restore_review_finalization_failed(request_id)

    def finalize_review(self, request_id: str, **kwargs) -> bool:
        """Write the reviewed outcome, its events and its checksum atomically."""
        return self.db.finalize_review(request_id, **kwargs)

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
