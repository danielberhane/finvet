"""One event, one timestamp.

`AuditLogger.log_event` stamped its buffered copy and then called
`AuditDatabase.log_event`, which stamped the row again from its own clock. The
same logical event therefore carried two different times: one inside
`full_trace.events` (via the buffer that `commit_execution` persists) and one in
the `audit_events` row.

Nothing surfaced it while integrity only recomputed the envelope's own
checksum. The moment the audit API compares the queried event rows against the
envelope's events -- which is what makes "the displayed trail is the trail that
was hashed" true -- every legitimate execution reports a `timestamp` mismatch
and integrity reads `failed`. A correctness check that fires on correct data is
worse than none, because it trains a reader to ignore it.

The contract: the timestamp is generated once, in the logger, and persisted
unchanged.
"""

from unittest.mock import MagicMock, patch

from finvet.audit.logger import AuditLogger


def _logger_with_recording_db():
    """A real AuditLogger whose database layer records what it was handed.

    The defect lives in the hand-off between the two layers, so the test has to
    exercise the real `AuditLogger.log_event` rather than assemble an event
    dict itself.
    """
    logger = AuditLogger.__new__(AuditLogger)
    import threading

    logger._events = {}
    logger._lock = threading.Lock()
    logger._last_write_ok = True
    logger.db = MagicMock()
    logger.db.log_event.return_value = True
    return logger


class TestOneEventHasOneTimestamp:

    def test_the_persisted_timestamp_is_the_buffered_timestamp(self):
        logger = _logger_with_recording_db()

        logger.log_event(event_type="input_received", request_id="req_ts",
                         data={"claim_raw": "c"})

        buffered = logger._events["req_ts"][0]["timestamp"]
        persisted = logger.db.log_event.call_args.kwargs.get("timestamp")

        assert persisted is not None, (
            "AuditDatabase.log_event was not given a timestamp, so it stamps "
            "the row from its own clock and the two copies diverge")
        assert persisted == buffered

    def test_the_event_id_also_matches(self):
        """Sanity: the two copies describe the same event."""
        logger = _logger_with_recording_db()
        logger.log_event(event_type="input_received", request_id="req_ts",
                         data={})

        assert (logger.db.log_event.call_args.kwargs["event_id"]
                == logger._events["req_ts"][0]["event_id"])

    def test_an_unbuffered_event_still_carries_one_timestamp(self):
        """buffer_for_execution=False skips the buffer, not the contract."""
        logger = _logger_with_recording_db()
        logger.log_event(event_type="memory_cache_accepted",
                         request_id="req_ts", data={},
                         buffer_for_execution=False)

        assert logger.db.log_event.call_args.kwargs.get("timestamp")
        assert "req_ts" not in logger._events


class TestPersistedRowsMatchTheEnvelope:
    """The projection the audit API compares must be equal for a clean run."""

    def test_projected_persisted_events_equal_the_envelope_events(self):
        from finvet.audit.integrity import project_events
        from finvet.audit.integrity import build_execution_envelope

        logger = _logger_with_recording_db()
        for event_type in ("input_received", "claim_parsed", "sec_completed"):
            logger.log_event(event_type=event_type, request_id="req_ts",
                             data={"k": event_type})

        buffered = logger._events["req_ts"]

        # What the row-writing layer was actually told to store, shaped like a
        # get_events() result.
        persisted = [
            {
                "event_id": call.kwargs["event_id"],
                "request_id": call.kwargs["request_id"],
                "parent_event_id": call.kwargs.get("parent_event_id"),
                "event_type": call.kwargs["event_type"],
                "timestamp": call.kwargs["timestamp"],
                "agent": call.kwargs.get("agent"),
                "data": call.kwargs["data"],
                "created_at": "storage-metadata-not-in-the-envelope",
            }
            for call in logger.db.log_event.call_args_list
        ]

        envelope = build_execution_envelope(
            request_id="req_ts", claim_text="c", terminal_status="success",
            verdict="SUPPORTS", confidence=0.9, agents_run=["SEC"],
            events=buffered, final_response={"verdict": "SUPPORTS"},
            data_sources=None)

        assert (project_events(persisted)
                == project_events(envelope["events"]))


class TestDeterministicEventOrdering:
    """Two events can share a timestamp; the order must still be stable.

    Ordering by timestamp alone leaves same-millisecond events in whatever
    order the database returns them, so the projection comparison could fail
    intermittently on data that is entirely correct.
    """

    def test_get_events_orders_by_timestamp_then_event_id(self):
        from finvet.audit.database import AuditDatabase

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: False
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        db = AuditDatabase.__new__(AuditDatabase)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            db.get_events("req_ts")

        order_by = session.query.return_value.filter.return_value.order_by
        clauses = [str(c) for c in order_by.call_args.args]
        assert any("timestamp" in c for c in clauses), clauses
        assert any("event_id" in c for c in clauses), (
            f"ordering is not deterministic for same-timestamp events: {clauses}")

    def test_same_timestamp_events_have_a_total_order(self):
        """The tiebreaker must actually decide, not merely be present."""
        from finvet.audit.integrity import project_events

        same = "2026-08-25T12:00:00"
        rows = [
            {"event_id": "evt_b", "request_id": "r", "parent_event_id": None,
             "event_type": "x", "timestamp": same, "agent": None, "data": {}},
            {"event_id": "evt_a", "request_id": "r", "parent_event_id": None,
             "event_type": "x", "timestamp": same, "agent": None, "data": {}},
        ]
        ordered = sorted(rows, key=lambda e: (e["timestamp"], e["event_id"]))
        assert [e["event_id"] for e in ordered] == ["evt_a", "evt_b"]
        assert (project_events(ordered)
                != project_events(rows)), (
            "the two orderings must be distinguishable, or the tiebreaker is "
            "not doing anything")
