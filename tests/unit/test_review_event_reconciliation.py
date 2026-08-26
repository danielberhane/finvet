"""A review may not be finalized from an event list that failed to load.

`submit_hitl_review` finalized with `events=audit.get_events(request_id) or []`.
That read goes to `AuditDatabase.get_events`, which catches bare `Exception` and
returns `[]`. So a database read failure at finalization was indistinguishable
from a run that genuinely produced no events, and the two were handled the same
way: the execution committed with `full_trace.events = []`.

The checksum is then computed over that empty envelope, so it is internally
consistent and `/audit/{id}` reports integrity **`verified`** -- a wiped trail
presented as an intact one. For a system whose thesis is auditability that is
worse than an outright failure, because nothing downstream can tell.

There is a second half. `log_event` writes each row best-effort and keeps a
buffered copy; `commit_execution` (the verify path) reconciles from that buffer,
which is why the same defect does not exist there. The review path re-read the
database instead, so any event whose immediate write had failed was silently
absent from the envelope even when the read itself succeeded.

Finalization therefore reads strictly -- a failure raises rather than
disappearing -- and merges the persisted rows with the buffer by `event_id`, so
the envelope describes every event the run produced.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from finvet.api.models import HITLReviewRequest
from finvet.api.routes import review as review_route


def _event(event_id, timestamp, event_type="node_completed"):
    return {"event_id": event_id, "request_id": "req_x", "timestamp": timestamp,
            "event_type": event_type, "parent_event_id": None, "agent": None,
            "data": {}}


class TestTheStrictReadDoesNotInventAnEmptyTrail:
    """`get_events` returning [] on failure is right for a read-only caller
    rendering a page, and wrong for the one caller that hashes the result."""

    def _logger(self, *, db_events=None, db_error=None, buffered=()):
        from finvet.audit.logger import AuditLogger

        logger = AuditLogger.__new__(AuditLogger)
        logger.db = MagicMock()
        if db_error is not None:
            logger.db.get_events_strict.side_effect = db_error
        else:
            logger.db.get_events_strict.return_value = list(db_events or [])
        import threading
        logger._lock = threading.Lock()
        logger._events = {"req_x": list(buffered)}
        return logger

    def test_a_read_failure_raises_rather_than_returning_nothing(self):
        logger = self._logger(db_error=RuntimeError("connection reset"))

        with pytest.raises(Exception):
            logger.events_for_finalization("req_x")

    def test_a_genuinely_empty_run_is_still_empty(self):
        """The control. Distinguishing failure from emptiness only matters if
        emptiness still works."""
        logger = self._logger(db_events=[])

        assert logger.events_for_finalization("req_x") == []

    def test_persisted_events_come_back(self):
        logger = self._logger(db_events=[_event("evt_a", "2026-08-26T00:00:01")])

        assert [e["event_id"] for e in
                logger.events_for_finalization("req_x")] == ["evt_a"]


class TestTheBufferFillsWhatThePersistedTrailIsMissing:

    def _logger(self, db_events, buffered):
        from finvet.audit.logger import AuditLogger
        import threading

        logger = AuditLogger.__new__(AuditLogger)
        logger.db = MagicMock()
        logger.db.get_events_strict.return_value = list(db_events)
        logger._lock = threading.Lock()
        logger._events = {"req_x": list(buffered)}
        return logger

    def test_an_event_whose_write_failed_is_still_in_the_envelope(self):
        """The defect's second half: the row never landed, so a re-read cannot
        see it, and the envelope described a run that was missing a step."""
        persisted = [_event("evt_a", "2026-08-26T00:00:01"),
                     _event("evt_b", "2026-08-26T00:00:02")]
        buffered = persisted + [_event("evt_c", "2026-08-26T00:00:03")]

        ids = [e["event_id"] for e in
               self._logger(persisted, buffered).events_for_finalization("req_x")]

        assert ids == ["evt_a", "evt_b", "evt_c"]

    def test_events_present_in_both_appear_once(self):
        events = [_event("evt_a", "2026-08-26T00:00:01"),
                  _event("evt_b", "2026-08-26T00:00:02")]

        ids = [e["event_id"] for e in
               self._logger(events, events).events_for_finalization("req_x")]

        assert ids == ["evt_a", "evt_b"]

    def test_the_order_matches_the_database_contract(self):
        """(timestamp, event_id) -- the same tiebreak `get_events` applies, so
        the envelope and the queried trail cannot disagree about order."""
        persisted = [_event("evt_b", "2026-08-26T00:00:01")]
        buffered = [_event("evt_a", "2026-08-26T00:00:01"),
                    _event("evt_c", "2026-08-26T00:00:00")]

        ids = [e["event_id"] for e in
               self._logger(persisted, buffered).events_for_finalization("req_x")]

        assert ids == ["evt_c", "evt_a", "evt_b"]

    def test_an_empty_buffer_changes_nothing(self):
        persisted = [_event("evt_a", "2026-08-26T00:00:01")]

        ids = [e["event_id"] for e in
               self._logger(persisted, []).events_for_finalization("req_x")]

        assert ids == ["evt_a"]


class TestTheRouteRefusesToFinalizeOnAFailedRead:
    """Driven through the route, which is what actually releases a verdict."""

    @pytest.fixture
    def audit(self):
        logger = MagicMock()
        logger.claim_pending_review.return_value = "claimed"
        logger.finalize_review.return_value = True
        logger.mark_review_finalization_failed.return_value = True
        logger.events_for_finalization.return_value = []
        return logger

    @pytest.fixture
    def graph(self):
        g = MagicMock()
        g.invoke.return_value = {
            "final_response": {"verdict": "REFUTES", "confidence": 0.95,
                               "metadata": {"data_sources": {}}}}
        return g

    @pytest.fixture(autouse=True)
    def _wire(self, monkeypatch, audit, graph):
        monkeypatch.setattr(review_route, "get_audit_logger", lambda: audit)
        monkeypatch.setattr(review_route.deps, "verification_graph", graph)
        yield

    def _submit(self):
        return review_route.submit_hitl_review(
            "req_pending", HITLReviewRequest(decision="approve"))

    def test_a_failed_event_read_does_not_report_reviewed(self, audit):
        audit.events_for_finalization.side_effect = RuntimeError("db down")

        with pytest.raises(HTTPException) as excinfo:
            self._submit()

        assert excinfo.value.status_code == 503

    def test_a_failed_event_read_never_finalizes(self, audit):
        """The whole point: an empty envelope must not be written and then
        reported as integrity-verified."""
        audit.events_for_finalization.side_effect = RuntimeError("db down")

        with pytest.raises(HTTPException):
            self._submit()

        audit.finalize_review.assert_not_called()

    def test_a_failed_event_read_marks_the_row_recoverable(self, audit):
        """invoke() has already run, so the claim may not go back to PENDING;
        the checkpoint still holds the result reconciliation needs."""
        audit.events_for_finalization.side_effect = RuntimeError("db down")

        with pytest.raises(HTTPException):
            self._submit()

        audit.mark_review_finalization_failed.assert_called_once()
        audit.release_review_claim.assert_not_called()

    def test_the_envelope_carries_the_reconciled_events(self, audit):
        audit.events_for_finalization.return_value = [
            _event("evt_a", "2026-08-26T00:00:01")]

        self._submit()

        assert audit.finalize_review.call_args.kwargs["events"] == [
            _event("evt_a", "2026-08-26T00:00:01")]

    def test_a_healthy_review_still_completes(self, audit):
        result = self._submit()

        assert result["status"] == "reviewed"


class TestRecoveryUsesTheSameStrictView:
    """The recovery path is where an incomplete envelope actually matters.

    `submit_hitl_review` was fixed to read strictly and merge the buffer.
    `reconcile_review` — the path a failed finalization *lands in* — still read
    `get_events() or []`, so the defect survived exactly where it does harm: a
    review whose first write failed is reconciled from a lossy read, and an
    event whose immediate write also failed is absent from the recovered
    envelope. The primary path was closed and the recovery path left open.

    The buffer is also discarded in `finally` on every exit, including the
    failure that leads to recovery, so by reconciliation time the in-memory
    copy is gone. It is now retained while a review is still recoverable.
    """

    @pytest.fixture
    def audit(self):
        logger = MagicMock()
        logger.get_review_recovery.return_value = {
            "review_decision": "approve", "reviewer_notes": None}
        logger.claim_review_finalization.return_value = "claimed"
        logger.get_execution.return_value = {
            "full_trace": {"review_recovery": {
                "review_decision": "approve", "reviewer_notes": None}}}
        logger.finalize_review.return_value = True
        logger.events_for_finalization.return_value = [
            _event("evt_a", "2026-08-26T00:00:01")]
        logger.get_events.return_value = []
        return logger

    @pytest.fixture(autouse=True)
    def _wire(self, monkeypatch, audit):
        monkeypatch.setattr(review_route, "get_audit_logger", lambda: audit)
        graph = MagicMock()
        graph.get_state.return_value = MagicMock(values={
            "final_response": {"verdict": "REFUTES", "confidence": 0.9,
                               "metadata": {}}})
        monkeypatch.setattr(review_route.deps, "verification_graph", graph)
        yield

    def test_reconciliation_reads_the_merged_view(self, audit):
        review_route.reconcile_review("req_recover")

        audit.events_for_finalization.assert_called_once_with("req_recover")
        assert audit.finalize_review.call_args.kwargs["events"] == [
            _event("evt_a", "2026-08-26T00:00:01")]

    def test_reconciliation_never_uses_the_lossy_read(self, audit):
        review_route.reconcile_review("req_recover")

        audit.get_events.assert_not_called()

    def test_a_failed_read_does_not_finalize_a_recovery(self, audit):
        audit.events_for_finalization.side_effect = RuntimeError("db down")

        with pytest.raises(HTTPException) as excinfo:
            review_route.reconcile_review("req_recover")

        assert excinfo.value.status_code == 503
        audit.finalize_review.assert_not_called()

    def test_the_buffer_survives_a_failed_finalization(self, audit):
        """It is the only copy of an event whose write failed, and
        reconciliation is what still needs it."""
        audit.finalize_review.return_value = False

        with pytest.raises(HTTPException):
            review_route.submit_hitl_review(
                "req_pending", HITLReviewRequest(decision="approve"))

        audit.discard_buffer.assert_not_called()

    def test_the_buffer_is_released_after_durable_success(self, audit):
        review_route.reconcile_review("req_recover")

        audit.discard_buffer.assert_called_once_with("req_recover")
