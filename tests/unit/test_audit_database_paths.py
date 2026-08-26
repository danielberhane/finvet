"""The audit store's failure and query paths.

Every method here has a happy path exercised elsewhere and an error path that
was not exercised at all. That asymmetry matters more in this module than in
most: when the audit store is failing is exactly when its behaviour decides
whether a wrong answer is *recorded* as wrong or simply lost.

The contracts being pinned:

- A read that fails returns the empty answer, never a partial one. A caller
  cannot tell a half-populated list from a complete one.
- A conditional transition that fails returns `False` or `"unavailable"` —
  never `True`, and never `"missing"`, which the review route turns into a 404
  telling a reviewer their claim does not exist when in fact the store is down.
- A write is either committed whole or reported failed. `commit_execution`
  swallowing an error while returning `True` is the one outcome that would let
  an unaudited verdict reach a user.

Sessions are mocked because the point is what the method does when the database
misbehaves, which a live Postgres will not do on request. The real transitions
run against Postgres in `tests/integration/test_review_race.py`.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from finvet.audit.database import REVIEW_FINALIZATION_FAILED, AuditDatabase


def _db():
    return AuditDatabase.__new__(AuditDatabase)


def _session(**behaviour):
    """A session context manager whose query chain is configurable."""
    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = lambda s, *a: False
    for attr, value in behaviour.items():
        setattr(session, attr, value)
    return session


def _broken_session(exc=RuntimeError("connection reset")):
    def boom():
        raise exc
    return boom


class TestReadsFailToTheEmptyAnswer:
    """A partial result is indistinguishable from a complete one."""

    def test_get_execution_returns_none_when_the_store_is_down(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().get_execution("req_x") is None

    def test_get_execution_returns_none_for_an_unknown_id(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = None
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().get_execution("req_missing") is None

    def test_get_events_returns_an_empty_list_when_the_store_is_down(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().get_events("req_x") == []

    def test_get_pending_reviews_returns_an_empty_list_on_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().get_pending_reviews() == []

    def test_search_by_claim_hash_returns_an_empty_list_on_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().search_by_claim_hash("a claim", days=7) == []

    def test_search_by_data_source_returns_an_empty_list_on_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().search_by_data_source("xbrl") == []

    def test_list_executions_returns_an_empty_list_on_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().list_executions() == []


class TestWritesReportFailureRatherThanSwallowIt:

    def test_log_event_reports_a_failed_write(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().log_event(
                event_id="evt_1", request_id="req_x", event_type="t",
                data={}, timestamp="2026-08-26T00:00:00") is False

    def test_commit_execution_reports_a_failed_write(self):
        """The one outcome that would let an unaudited verdict reach a user."""
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().commit_execution(
                request_id="req_x", claim_text="c", verdict="SUPPORTS",
                confidence=0.9, agents_run=["SEC"], events=[],
                execution_time_ms=1) is False

    def test_a_duplicate_execution_is_not_an_error(self):
        """Re-committing the same request_id hits the unique constraint. That
        means the row is already there, which is the desired end state."""
        session = _session()
        session.add.side_effect = IntegrityError("dup", None, Exception())
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().commit_execution(
                request_id="req_x", claim_text="c", verdict="SUPPORTS",
                confidence=0.9, agents_run=[], events=[],
                execution_time_ms=1) is False


class TestLifecycleTransitionsFailClosed:
    """A transition that cannot be attempted must not report success."""

    def test_claiming_a_review_reports_unavailable_not_missing(self):
        """"missing" becomes a 404 telling the reviewer their claim does not
        exist. During an outage it does exist and the store is unreachable."""
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().claim_pending_review("req_x") == "unavailable"

    def test_releasing_a_claim_reports_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().release_review_claim("req_x") is False

    def test_marking_finalization_failed_reports_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().mark_review_finalization_failed(
                "req_x", error_type="audit_persistence_failed",
                review_decision="approve", reviewer_notes=None,
                checkpoint_has_final_state=True) is False

    def test_claiming_a_finalization_reports_unavailable(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().claim_review_finalization("req_x") == "unavailable"

    def test_restoring_the_recovery_state_reports_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().restore_review_finalization_failed("req_x") is False

    def test_finalizing_a_review_reports_failure(self):
        with patch("finvet.audit.database.get_db_session", _broken_session()):
            assert _db().finalize_review(
                "req_x", verdict="SUPPORTS", confidence=0.9,
                final_response={}, data_sources=None, events=[],
                review_decision="approve") is False


class TestTransitionsRefuseTheWrongStartingState:
    """Each conditional UPDATE is guarded on the state it may leave."""

    def _no_rows_changed(self):
        session = _session()
        query = session.query.return_value.filter.return_value
        query.update.return_value = 0
        query.first.return_value = None
        return session, query

    def test_marking_refuses_a_row_that_is_not_being_reviewed(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = None
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().mark_review_finalization_failed(
                "req_x", error_type="x", review_decision="approve",
                reviewer_notes=None, checkpoint_has_final_state=False) is False

    def test_finalizing_refuses_a_row_that_is_not_being_reviewed(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = None
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().finalize_review(
                "req_x", verdict="SUPPORTS", confidence=0.9,
                final_response={}, data_sources=None, events=[],
                review_decision="approve") is False

    def test_claiming_a_finalization_that_is_not_stuck_conflicts(self):
        session, query = self._no_rows_changed()
        query.first.return_value = ("req_x",)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().claim_review_finalization("req_x") == "conflict"

    def test_claiming_a_finalization_for_an_unknown_row_is_missing(self):
        session, _ = self._no_rows_changed()
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().claim_review_finalization("req_x") == "missing"

    def test_restoring_reports_false_when_no_row_changed(self):
        session, _ = self._no_rows_changed()
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert _db().restore_review_finalization_failed("req_x") is False

    @pytest.mark.parametrize("method,expected", [
        ("claim_review_finalization", REVIEW_FINALIZATION_FAILED),
        ("restore_review_finalization_failed", "REVIEWING"),
    ])
    def test_each_transition_names_the_state_it_leaves(self, method, expected):
        """Assert the predicate, not just the value written. Deleting the
        guard leaves the UPDATE looking correct while letting any row through."""
        session, _ = self._no_rows_changed()
        with patch("finvet.audit.database.get_db_session", lambda: session):
            getattr(_db(), method)("req_x")

        # The FIRST filter is the guarded UPDATE. claim_review_finalization
        # makes a second, unguarded read afterwards to tell 404 from 409, and
        # asserting on that one would pass with the guard deleted.
        first_filter = session.query.return_value.filter.call_args_list[0]
        criteria = " ".join(
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in first_filter.args)
        assert expected in criteria, criteria


class TestEventReconciliation:
    """`_insert_missing_events` converges the envelope and the event table."""

    def test_nothing_to_do_for_an_empty_event_list(self):
        session = _session()
        AuditDatabase._insert_missing_events(session, "req_x", [])
        session.add.assert_not_called()

    def test_events_without_ids_are_skipped(self):
        session = _session()
        AuditDatabase._insert_missing_events(
            session, "req_x", [{"event_type": "t"}])
        session.add.assert_not_called()

    def test_an_event_the_table_already_has_is_not_reinserted(self):
        session = _session()
        session.query.return_value.filter.return_value.all.return_value = [("evt_1",)]
        AuditDatabase._insert_missing_events(
            session, "req_x", [{"event_id": "evt_1", "event_type": "t"}])
        session.add.assert_not_called()

    def test_a_missing_event_is_inserted_with_its_own_timestamp(self):
        """Not a fresh one: the buffered copy and the row describe one event,
        and the audit API compares them."""
        session = _session()
        session.query.return_value.filter.return_value.all.return_value = []
        AuditDatabase._insert_missing_events(session, "req_x", [{
            "event_id": "evt_9", "event_type": "t",
            "timestamp": "2026-08-26T00:00:00", "data": {"k": "v"}}])

        session.add.assert_called_once()
        added = session.add.call_args.args[0]
        assert added.event_id == "evt_9"
        assert added.timestamp == "2026-08-26T00:00:00"
