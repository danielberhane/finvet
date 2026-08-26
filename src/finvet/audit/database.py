"""PostgreSQL database for audit trail storage using SQLAlchemy."""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.exc import IntegrityError
from ..config.database import get_db_session
from ..utils.logging import get_logger
from .integrity import (
    build_execution_envelope,
    compute_execution_checksum,
)
from .models import AuditEvent, AuditExecution

logger = get_logger(__name__)

# The verdict column doubles as review-lifecycle state. This marker means the
# graph was entered but produced no durable audited outcome, so the row is
# neither pending nor reviewed: it is waiting for an operator to retry
# finalization. Release B moves lifecycle state to its own column.
REVIEW_FINALIZATION_FAILED = "REVIEW_FINALIZATION_FAILED"

# The queue's view of each lifecycle state. Exposed as its own field so a
# client chooses an action from a value it can switch on, rather than parsing
# the verdict column and inheriting its double meaning.
REVIEW_STATUS_BY_VERDICT = {
    "PENDING": "pending",
    "REVIEWING": "in_review",
    REVIEW_FINALIZATION_FAILED: "finalization_failed",
}


class AuditDatabase:
    """PostgreSQL persistence for the audit trail.

    Append-only by convention, not by enforcement: nothing in the schema
    prevents an UPDATE. The execution checksum detects a row altered
    without recomputation; it does not prevent the alteration.
    """

    def __init__(self):
        """Initialize audit database. PostgreSQL connection is managed by get_db_session()."""

    def log_event(
        self,
        event_id: str,
        request_id: str,
        event_type: str,
        data: Dict[str, Any],
        timestamp: str,
        parent_event_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> bool:
        """Log a single audit event (append-only).

        Args:
            event_id: Unique event identifier
            request_id: Request this event belongs to
            event_type: Type of event (e.g., "claim_parsed", "sec_agent_completed")
            data: Event data as dictionary
            timestamp: When the event occurred, generated once by AuditLogger
                and persisted unchanged. Stamping the row here from a second
                clock gave the same logical event two different times -- one in
                the buffered copy that reaches `full_trace.events`, another in
                the row -- so the audit API's comparison of the two reported a
                mismatch on every legitimate execution.
            parent_event_id: Optional parent event ID
            agent: Optional agent name

        Returns:
            True if logged successfully
        """
        try:
            with get_db_session() as session:
                event = AuditEvent(
                    event_id=event_id,
                    request_id=request_id,
                    parent_event_id=parent_event_id,
                    event_type=event_type,
                    timestamp=timestamp,
                    agent=agent,
                    data=data,  # SQLAlchemy handles JSONB serialization
                )
                session.add(event)
                # Session auto-commits on successful context exit
            return True
        except IntegrityError:
            # Event already exists (duplicate), this is OK for idempotency
            return False
        except Exception as e:
            logger.error(f"Failed to log audit event {event_id}: {e}")
            return False

    @staticmethod
    def _insert_missing_events(session, request_id: str, events: list) -> None:
        """Insert buffered events the event table is missing, same transaction.

        log_event writes each event to the database as it happens, but that
        write is best effort -- a transient failure there used to leave the
        envelope describing events that no row records, so the execution and
        its queryable timeline disagreed. Reconciling inside the execution's
        own transaction makes them converge or fail together.
        """
        if not events:
            return

        buffered_ids = [e.get("event_id") for e in events if e.get("event_id")]
        if not buffered_ids:
            return

        existing = {
            row[0] for row in session.query(AuditEvent.event_id)
            .filter(AuditEvent.event_id.in_(buffered_ids)).all()
        }

        for event in events:
            event_id = event.get("event_id")
            if not event_id or event_id in existing:
                continue
            session.add(AuditEvent(
                event_id=event_id,
                request_id=event.get("request_id", request_id),
                parent_event_id=event.get("parent_event_id"),
                event_type=event.get("event_type", "unknown"),
                timestamp=event.get("timestamp", datetime.utcnow().isoformat()),
                agent=event.get("agent"),
                data=event.get("data") or {},
            ))

    def commit_execution(
        self,
        request_id: str,
        claim_text: str,
        verdict: Optional[str],
        confidence: Optional[float],
        agents_run: list,
        events: list,
        execution_time_ms: int,
        final_response: Optional[Dict[str, Any]] = None,
        data_sources: Optional[Dict[str, Any]] = None,
        terminal_status: Optional[str] = None,
    ) -> bool:
        """Commit full execution trace (append-only).

        Args:
            request_id: Unique request identifier
            claim_text: The claim text
            verdict: Final verdict
            confidence: Final confidence
            agents_run: List of agents that executed
            events: All audit events for this execution
            execution_time_ms: Total execution time
            final_response: Optional final API response dict (for HITL review lookups)

        Returns:
            True if committed successfully
        """
        timestamp = datetime.utcnow().isoformat()
        claim_hash = hashlib.sha256(claim_text.encode()).hexdigest()

        # Hash exactly what gets stored. The previous scheme hashed the event
        # list while the row also held the claim, verdict, confidence, final
        # response and data sources -- so any of those could change without
        # disturbing the digest.
        envelope = build_execution_envelope(
            request_id=request_id,
            claim_text=claim_text,
            terminal_status=terminal_status,
            verdict=verdict,
            confidence=confidence,
            agents_run=agents_run,
            events=events,
            final_response=final_response,
            data_sources=data_sources,
        )
        execution_hash = compute_execution_checksum(envelope)

        try:
            with get_db_session() as session:
                execution = AuditExecution(
                    request_id=request_id,
                    timestamp=timestamp,
                    claim_text=claim_text,
                    claim_hash=claim_hash,
                    verdict=verdict,
                    confidence=confidence,
                    agents_run=agents_run,  # SQLAlchemy handles JSONB
                    total_events=len(events),
                    execution_time_ms=execution_time_ms,
                    execution_hash=execution_hash,
                    # Stored verbatim: this is the object that was hashed, so
                    # verification can recompute from what it reads back.
                    full_trace=envelope,
                    data_sources=data_sources,
                )
                session.add(execution)
                self._insert_missing_events(session, request_id, events)
                # Session auto-commits on successful context exit
            return True
        except IntegrityError:
            # Execution already committed (duplicate request_id)
            return False
        except Exception as e:
            logger.error(f"Failed to commit execution {request_id}: {e}")
            return False

    @staticmethod
    def _execution_to_dict(execution: AuditExecution) -> Dict[str, Any]:
        """Convert an AuditExecution ORM object to a dictionary."""
        return {
            "execution_id": execution.execution_id,
            "request_id": execution.request_id,
            "timestamp": execution.timestamp,
            "claim_text": execution.claim_text,
            "claim_hash": execution.claim_hash,
            "verdict": execution.verdict,
            "confidence": execution.confidence,
            "agents_run": execution.agents_run,
            "total_events": execution.total_events,
            "execution_time_ms": execution.execution_time_ms,
            "execution_hash": execution.execution_hash,
            "full_trace": execution.full_trace,
            "data_sources": execution.data_sources,
            "created_at": execution.created_at,
        }

    def get_execution(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full execution trace by request_id.

        Args:
            request_id: Request identifier

        Returns:
            Execution record or None if not found
        """
        try:
            with get_db_session() as session:
                execution = session.query(AuditExecution).filter(
                    AuditExecution.request_id == request_id
                ).first()

                if execution:
                    return self._execution_to_dict(execution)
                return None
        except Exception as e:
            logger.error(f"Failed to get execution {request_id}: {e}")
            return None

    def get_events_strict(self, request_id: str) -> list:
        """Retrieve all events for a request, raising if the read fails.

        `get_events` swallows the exception and returns [], which is right for
        a caller rendering a page and wrong for the one caller that hashes the
        result: an unreadable trail became an empty envelope whose checksum
        then verified, reporting a wiped audit trail as an intact one.
        """
        with get_db_session() as session:
            # Same ordering contract as get_events: event_id breaks ties so
            # the envelope and the queried trail cannot disagree.
            events = session.query(AuditEvent).filter(
                AuditEvent.request_id == request_id
            ).order_by(AuditEvent.timestamp, AuditEvent.event_id).all()

            return [
                {
                    "event_id": event.event_id,
                    "request_id": event.request_id,
                    "parent_event_id": event.parent_event_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "agent": event.agent,
                    "data": event.data,
                    "created_at": event.created_at,
                }
                for event in events
            ]

    def get_events(self, request_id: str) -> list:
        """Retrieve all events for a request.

        Args:
            request_id: Request identifier

        Returns:
            List of event dictionaries
        """
        try:
            with get_db_session() as session:
                # event_id breaks ties. Two events can share a timestamp, and
                # ordering by timestamp alone leaves their order up to the
                # database -- enough to make the audit API's comparison against
                # the hashed envelope fail intermittently on correct data.
                events = session.query(AuditEvent).filter(
                    AuditEvent.request_id == request_id
                ).order_by(AuditEvent.timestamp, AuditEvent.event_id).all()

                return [
                    {
                        "event_id": event.event_id,
                        "request_id": event.request_id,
                        "parent_event_id": event.parent_event_id,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "agent": event.agent,
                        "data": event.data,
                        "created_at": event.created_at,
                    }
                    for event in events
                ]
        except Exception as e:
            logger.error(f"Failed to get events for {request_id}: {e}")
            return []

    def claim_pending_review(self, request_id: str) -> str:
        """Atomically take ownership of a pending review.

        Returns "claimed", "conflict", "missing", or "unavailable".

        "unavailable" is distinct on purpose: a database outage used to be
        reported as "missing", which the route turned into a 404 telling the
        reviewer their claim did not exist. It does exist; the store cannot be
        reached, and a 503 says so.

        The transition is a single conditional UPDATE. Reading the row and then
        writing it leaves a window where two reviewers both see PENDING and both
        proceed, and the previous code had no status check at all: a second
        submission simply overwrote the first, and a review for a request that
        never existed reported success.
        """
        try:
            with get_db_session() as session:
                changed = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == "PENDING")
                    .update({"verdict": "REVIEWING"}, synchronize_session=False)
                )
                if changed:
                    return "claimed"
                # Nothing changed: either the row is gone or somebody else has
                # it. One read distinguishes 404 from 409.
                exists = (
                    session.query(AuditExecution.request_id)
                    .filter(AuditExecution.request_id == request_id)
                    .first()
                )
                return "conflict" if exists else "missing"
        except Exception as e:
            logger.error(f"Failed to claim review for {request_id}: {e}")
            return "unavailable"

    def release_review_claim(self, request_id: str) -> bool:
        """Return a claimed row to PENDING after a failed resume.

        Without this a lost checkpoint would strand the claim in REVIEWING and
        no reviewer could pick it up again.
        """
        try:
            with get_db_session() as session:
                changed = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == "REVIEWING")
                    .update({"verdict": "PENDING"}, synchronize_session=False)
                )
                return bool(changed)
        except Exception as e:
            logger.error(f"Failed to release review claim for {request_id}: {e}")
            return False

    def mark_review_finalization_failed(
        self,
        request_id: str,
        *,
        error_type: str,
        review_decision: str,
        reviewer_notes: Optional[str],
        checkpoint_has_final_state: bool,
    ) -> bool:
        """REVIEWING -> REVIEW_FINALIZATION_FAILED, in one transaction.

        Reached once the graph has been entered but no durable reviewed outcome
        exists. Returning the row to PENDING is not an option there: the
        checkpoint may already have advanced, and a second reviewer resuming it
        would run a partially-executed graph. The row is instead moved to an
        explicit recovery state that an operator can find and a reconcile call
        can retry.

        Only safe metadata is recorded. The graph's verdict and confidence are
        deliberately absent: they were never durably audited, and copying them
        into the row would present an unaudited result as a stored outcome.
        """
        try:
            with get_db_session() as session:
                execution = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == "REVIEWING")
                    .first()
                )
                if execution is None:
                    logger.warning(
                        "mark_review_finalization_failed found no REVIEWING "
                        f"row for {request_id}")
                    return False

                envelope = dict(execution.full_trace or {})
                envelope["terminal_status"] = "review_finalization_failed"
                envelope["review_recovery"] = {
                    "error_type": error_type,
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "review_decision": review_decision,
                    "reviewer_notes": reviewer_notes,
                    # What recovery can actually rely on, not what it hopes for.
                    "checkpoint_has_final_state": bool(checkpoint_has_final_state),
                }

                execution.verdict = REVIEW_FINALIZATION_FAILED
                execution.full_trace = envelope
                # The envelope changed, so its digest must change with it.
                execution.execution_hash = compute_execution_checksum(envelope)
            return True
        except Exception as e:
            logger.error(
                f"Failed to mark review finalization failed for {request_id}: {e}")
            return False

    def claim_review_finalization(self, request_id: str) -> str:
        """Take ownership of a stuck review for reconciliation.

        REVIEW_FINALIZATION_FAILED -> REVIEWING as a single conditional UPDATE,
        so two operators retrying the same row cannot both proceed to
        finalize_review. Returns "claimed", "conflict", "missing" or
        "unavailable", matching claim_pending_review.
        """
        try:
            with get_db_session() as session:
                changed = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == REVIEW_FINALIZATION_FAILED)
                    .update({"verdict": "REVIEWING"}, synchronize_session=False)
                )
                if changed:
                    return "claimed"
                exists = (
                    session.query(AuditExecution.request_id)
                    .filter(AuditExecution.request_id == request_id)
                    .first()
                )
                return "conflict" if exists else "missing"
        except Exception as e:
            logger.error(
                f"Failed to claim review finalization for {request_id}: {e}")
            return "unavailable"

    def restore_review_finalization_failed(self, request_id: str) -> bool:
        """REVIEWING -> REVIEW_FINALIZATION_FAILED after a failed retry.

        A reconcile attempt that claims the row and then cannot finalize must
        put it back where an operator will find it again, rather than leaving
        it in REVIEWING with nobody working on it.
        """
        try:
            with get_db_session() as session:
                changed = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == "REVIEWING")
                    .update({"verdict": REVIEW_FINALIZATION_FAILED},
                            synchronize_session=False)
                )
                return bool(changed)
        except Exception as e:
            logger.error(
                f"Failed to restore finalization-failed state for {request_id}: {e}")
            return False

    def finalize_review(
        self,
        request_id: str,
        *,
        verdict: str,
        confidence: float,
        final_response: Optional[Dict[str, Any]],
        data_sources: Optional[Dict[str, Any]],
        events: list,
        review_decision: str,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """Write the reviewed outcome and its checksum in one transaction.

        The previous version updated verdict and confidence alone, leaving
        full_trace, the checksum and the event count describing the pending-era
        run -- so the header could say REFUTES while the stored response still
        said PENDING.

        The new envelope records the review decision and the checksum it
        supersedes, so the pre-review state remains identifiable rather than
        being silently overwritten.
        """
        try:
            with get_db_session() as session:
                execution = (
                    session.query(AuditExecution)
                    .filter(AuditExecution.request_id == request_id,
                            AuditExecution.verdict == "REVIEWING")
                    .first()
                )
                if execution is None:
                    logger.warning(
                        f"finalize_review found no REVIEWING row for {request_id}")
                    return False

                envelope = build_execution_envelope(
                    request_id=request_id,
                    claim_text=execution.claim_text,
                    terminal_status="reviewed",
                    verdict=verdict,
                    confidence=confidence,
                    agents_run=execution.agents_run or [],
                    events=events,
                    final_response=final_response,
                    data_sources=data_sources,
                )
                envelope["review"] = {
                    "decision": review_decision,
                    "reviewer_notes": reviewer_notes,
                    "supersedes_checksum": execution.execution_hash,
                }

                execution.verdict = verdict
                execution.confidence = confidence
                execution.full_trace = envelope
                execution.execution_hash = compute_execution_checksum(envelope)
                execution.total_events = len(events)
                if data_sources is not None:
                    execution.data_sources = data_sources
                self._insert_missing_events(session, request_id, events)
            return True
        except Exception as e:
            logger.error(f"Failed to finalize review for {request_id}: {e}")
            return False

    def get_pending_reviews(self) -> list:
        """Retrieve all executions pending HITL review.

        Returns:
            List of pending execution records with UI-friendly fields
        """
        try:
            with get_db_session() as session:
                # REVIEWING is included deliberately. A claim is moved there
                # while its reviewer works; if that process dies before the
                # claim is released, filtering on PENDING alone would drop the
                # row from every reviewer's queue permanently. Showing it is
                # safe -- claim_pending_review still serialises access, so a
                # second reviewer gets a conflict rather than a duplicate.
                # REVIEW_FINALIZATION_FAILED belongs here too: its graph ran
                # but the audit write did not land, and the queue is the only
                # place a reviewer would ever look for it.
                executions = session.query(AuditExecution).filter(
                    AuditExecution.verdict.in_(
                        ("PENDING", "REVIEWING", REVIEW_FINALIZATION_FAILED))
                ).order_by(AuditExecution.timestamp.desc()).all()

                results = []
                for execution in executions:
                    full_trace = execution.full_trace or {}
                    final_resp = full_trace.get("final_response", {})

                    results.append({
                        "request_id": execution.request_id,
                        "claim": execution.claim_text,
                        "timestamp": execution.timestamp,
                        # Machine-readable, so the UI picks an action rather
                        # than decoding a verdict string that doubles as
                        # lifecycle state.
                        "review_status": REVIEW_STATUS_BY_VERDICT.get(
                            execution.verdict, "pending"),
                        "preliminary_analysis": final_resp.get("preliminary_analysis", {}),
                        "hitl_triggers": final_resp.get("hitl_triggers",
                                         final_resp.get("metadata", {}).get("hitl_triggers", [])),
                    })
                return results
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

    def search_by_claim_hash(self, claim_text: str, days: int = 7) -> list:
        """Search for past verifications of same/similar claim.

        Args:
            claim_text: Claim to search for
            days: Look back this many days

        Returns:
            List of matching execution records
        """
        claim_hash = hashlib.sha256(claim_text.encode()).hexdigest()
        cutoff_date = datetime.utcnow()
        from datetime import timedelta
        cutoff_date = (cutoff_date - timedelta(days=days)).isoformat()

        try:
            with get_db_session() as session:
                executions = session.query(AuditExecution).filter(
                    AuditExecution.claim_hash == claim_hash,
                    AuditExecution.timestamp >= cutoff_date
                ).order_by(AuditExecution.timestamp.desc()).all()

                return [self._execution_to_dict(e) for e in executions]
        except Exception as e:
            logger.error(f"Failed to search by claim hash: {e}")
            return []

    def search_by_data_source(self, source_type: str, limit: int = 50) -> list:
        """Find executions that used a specific data source (xbrl, rag, a2a).

        Args:
            source_type: Data source key to search for
            limit: Maximum results to return

        Returns:
            List of matching execution records
        """
        try:
            with get_db_session() as session:
                executions = session.query(AuditExecution).filter(
                    AuditExecution.data_sources.has_key(source_type)
                ).order_by(AuditExecution.timestamp.desc()).limit(limit).all()

                return [self._execution_to_dict(e) for e in executions]
        except Exception as e:
            logger.error(f"Failed to search by data source {source_type}: {e}")
            return []

    def list_executions(
        self,
        verdict: Optional[str] = None,
        agent: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        data_source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """List audit executions with optional filters.

        Args:
            verdict: Filter by verdict (SUPPORTS, REFUTES, NOT_ENOUGH_INFO, PENDING, REJECTED)
            agent: Filter by agent used (sec, market, news)
            date_from: ISO timestamp lower bound (inclusive)
            date_to: ISO timestamp upper bound (inclusive)
            data_source: Filter by data source key present (xbrl, rag, a2a)
            limit: Maximum results (default 50)
            offset: Pagination offset

        Returns:
            List of execution dicts ordered by timestamp descending
        """
        try:
            with get_db_session() as session:
                query = session.query(AuditExecution)
                if verdict:
                    query = query.filter(AuditExecution.verdict == verdict)
                if agent:
                    # DB stores agents as ['SEC'], ['Market'], ['News'] — map from lowercase UI values
                    _agent_stored = {"sec": "SEC", "market": "Market", "news": "News"}
                    stored_agent = _agent_stored.get(agent.lower(), agent)
                    query = query.filter(AuditExecution.agents_run.contains([stored_agent]))
                if date_from:
                    query = query.filter(AuditExecution.timestamp >= date_from)
                if date_to:
                    query = query.filter(AuditExecution.timestamp <= date_to)
                if data_source:
                    query = query.filter(AuditExecution.data_sources.has_key(data_source))
                executions = (
                    query
                    .order_by(AuditExecution.timestamp.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [self._execution_to_dict(e) for e in executions]
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return []
