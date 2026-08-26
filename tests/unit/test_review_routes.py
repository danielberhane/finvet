"""Identity, concurrency and audit consistency for POST /review/{request_id}.

Three defects lived here. Every resume failure fell into a fallback that
computed a verdict from the request body, so a review for an id that was never
verified returned status="reviewed" having persisted nothing. Nothing checked
that the claim was still pending, so two reviewers could both succeed and the
last write won. And finalization updated verdict and confidence alone, leaving
the stored response, checksum and event count describing the pending-era run.

The route now claims the row atomically before doing anything else, resumes
with the same callbacks as the initial verification, and finalizes the whole
envelope in one transaction. These tests drive the real handler.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from finvet.api.models import HITLReviewRequest
from finvet.api.routes import review as review_route


def _review(**kwargs):
    kwargs.setdefault("decision", "approve")
    return HITLReviewRequest(**kwargs)


@pytest.fixture
def audit():
    logger = MagicMock()
    logger.claim_pending_review.return_value = "claimed"
    logger.finalize_review.return_value = True
    logger.get_events.return_value = []
    return logger


@pytest.fixture
def graph():
    g = MagicMock()
    g.invoke.return_value = {
        "final_response": {"verdict": "REFUTES", "confidence": 0.95,
                           "metadata": {"data_sources": {"xbrl": {"used": True}}}},
    }
    return g


@pytest.fixture(autouse=True)
def _wire(monkeypatch, audit, graph):
    monkeypatch.setattr(review_route, "get_audit_logger", lambda: audit)
    monkeypatch.setattr(review_route.deps, "verification_graph", graph)
    yield


def _submit(request_id="req_pending", **kwargs):
    return review_route.submit_hitl_review(request_id, _review(**kwargs))


def _status(excinfo):
    return getattr(excinfo.value, "status_code", None)


# ---------------------------------------------------------------------------
# Step 1 — the contract rejects malformed decisions before anything happens
# ---------------------------------------------------------------------------

class TestRequestValidation:

    def test_unknown_decision_is_rejected(self):
        with pytest.raises(ValidationError):
            HITLReviewRequest(decision="maybe")

    def test_override_requires_a_verdict(self):
        with pytest.raises(ValidationError):
            HITLReviewRequest(decision="override")

    def test_verdict_without_override_is_rejected(self):
        """Silently discarding the reviewer's verdict is worse than refusing."""
        with pytest.raises(ValidationError):
            HITLReviewRequest(decision="approve", override_verdict="REFUTES")

    def test_notes_are_bounded(self):
        with pytest.raises(ValidationError):
            HITLReviewRequest(decision="approve", reviewer_notes="x" * 2001)


# ---------------------------------------------------------------------------
# Steps 2-3 — identity and concurrency
# ---------------------------------------------------------------------------

class TestIdentityAndConcurrency:

    def test_unknown_request_returns_404(self, audit):
        audit.claim_pending_review.return_value = "missing"
        with pytest.raises(HTTPException) as excinfo:
            _submit("req_does_not_exist", decision="override",
                    override_verdict="REFUTES")
        assert _status(excinfo) == 404

    def test_already_reviewed_returns_409(self, audit):
        audit.claim_pending_review.return_value = "conflict"
        with pytest.raises(HTTPException) as excinfo:
            _submit("req_done")
        assert _status(excinfo) == 409

    def test_second_review_is_rejected_after_first_claim(self, audit):
        """Two reviewers, one pending row. The second must not overwrite."""
        audit.claim_pending_review.side_effect = ["claimed", "conflict"]

        first = _submit()
        assert first["status"] == "reviewed"

        with pytest.raises(HTTPException) as excinfo:
            _submit()
        assert _status(excinfo) == 409

    def test_nothing_is_written_before_the_claim_succeeds(self, audit):
        """The old order logged an audit event, then discovered the request did
        not exist -- leaving an orphan event under an id with no execution."""
        audit.claim_pending_review.return_value = "missing"
        with pytest.raises(HTTPException):
            _submit("req_does_not_exist")

        assert audit.log_event.call_count == 0
        assert audit.finalize_review.call_count == 0


# ---------------------------------------------------------------------------
# Step 3 — no fabricated verdicts
# ---------------------------------------------------------------------------

class TestNoDirectVerdictFallback:

    def test_lost_checkpoint_returns_409_not_a_verdict(self, audit, graph):
        """MemorySaver checkpoints do not survive a restart. There is nothing
        to review, so the reviewer's own override must not become the answer."""
        graph.update_state.side_effect = Exception("no checkpoint for thread")

        with pytest.raises(HTTPException) as excinfo:
            _submit(decision="override", override_verdict="REFUTES")

        assert _status(excinfo) == 409
        assert excinfo.value.detail["error"] == "checkpoint_unavailable"
        assert audit.finalize_review.call_count == 0

    def test_lost_checkpoint_releases_the_claim(self, audit, graph):
        """Otherwise the row is stranded in REVIEWING and nobody can pick it up."""
        graph.update_state.side_effect = Exception("no checkpoint for thread")

        with pytest.raises(HTTPException):
            _submit()

        audit.release_review_claim.assert_called_once_with("req_pending")

    def test_resume_error_after_invoke_begins_never_releases_the_claim(
            self, audit, graph):
        """This test previously asserted the opposite, and the opposite is
        unsafe: once invoke has been entered the checkpoint may have advanced,
        so returning the row to PENDING would let a second reviewer resume a
        partially-executed graph. The row goes to the recovery state instead.

        The pre-invoke case still releases -- see
        test_lost_checkpoint_releases_the_claim above, which covers the
        update_state failure that release_review_claim exists for.
        """
        graph.invoke.side_effect = RuntimeError("node exploded")

        with pytest.raises(HTTPException) as excinfo:
            _submit()

        assert _status(excinfo) == 503
        audit.release_review_claim.assert_not_called()
        audit.mark_review_finalization_failed.assert_called_once()


# ---------------------------------------------------------------------------
# Steps 4-5 — the resumed run reaches the audit trail, whole
# ---------------------------------------------------------------------------

class TestResumeIsAudited:

    def test_resume_attaches_callbacks(self, graph):
        """Without them the resumed nodes never enter the event stream, so the
        timeline stops at the checkpoint."""
        from finvet.audit.callbacks import AuditCallbackHandler

        _submit()

        config = graph.invoke.call_args.args[1]
        assert any(isinstance(cb, AuditCallbackHandler)
                   for cb in config.get("callbacks", []))
        assert config["configurable"]["thread_id"] == "req_pending"

    def test_finalization_carries_the_whole_envelope(self, audit):
        """verdict and confidence alone left the stored response and checksum
        describing the pending-era run."""
        _submit(decision="override", override_verdict="REFUTES")

        kwargs = audit.finalize_review.call_args.kwargs
        assert kwargs["verdict"] == "REFUTES"
        assert kwargs["confidence"] == 0.95
        assert kwargs["final_response"]["verdict"] == "REFUTES"
        assert kwargs["data_sources"] == {"xbrl": {"used": True}}
        assert kwargs["review_decision"] == "override"

    def test_failed_finalization_is_not_reported_as_success(self, audit):
        """503, not 500: the verdict exists and only the write failed, so the
        row is recorded for reconciliation rather than lost."""
        audit.finalize_review.return_value = False
        with pytest.raises(HTTPException) as excinfo:
            _submit()
        assert _status(excinfo) == 503
        assert excinfo.value.detail["error"] == "review_finalization_failed"

    @pytest.mark.parametrize("decision,verdict", [
        ("approve", None), ("reject", None), ("override", "REFUTES"),
    ])
    def test_every_decision_reaches_finalization(self, audit, decision, verdict):
        out = _submit(decision=decision, override_verdict=verdict)
        assert out["status"] == "reviewed"
        assert audit.finalize_review.call_args.kwargs["review_decision"] == decision


class TestAtomicTransitionsAtTheDatabaseLayer:
    """The three transitions themselves, not the route's use of them.

    claim_pending_review is a single conditional UPDATE. Reading the row and
    then writing it would leave a window where two reviewers both see PENDING
    and both proceed, which is the race the route's 409 depends on being closed
    one layer down.
    """

    def _db(self, *, rows_changed, row_exists=True):
        from finvet.audit.database import AuditDatabase

        session = MagicMock()
        query = session.query.return_value.filter.return_value
        query.update.return_value = rows_changed
        query.first.return_value = ("req_x",) if row_exists else None
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: False

        db = AuditDatabase.__new__(AuditDatabase)
        return db, session, query

    @staticmethod
    def _filter_sql(session):
        """The filter's SQL with values inlined.

        str() on a SQLAlchemy expression renders values as bind parameters
        (`:verdict_1`), which hides the very literal these tests exist to
        check. literal_binds compiles them in.
        """
        parts = []
        for criterion in session.query.return_value.filter.call_args.args:
            parts.append(str(criterion.compile(
                compile_kwargs={"literal_binds": True})))
        return " ".join(parts)

    def test_claim_succeeds_when_a_pending_row_changes(self):
        db, session, _ = self._db(rows_changed=1)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert db.claim_pending_review("req_x") == "claimed"

    def test_claim_conflicts_when_the_row_exists_but_is_not_pending(self):
        db, session, _ = self._db(rows_changed=0, row_exists=True)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert db.claim_pending_review("req_x") == "conflict"

    def test_claim_is_missing_when_no_such_row(self):
        db, session, _ = self._db(rows_changed=0, row_exists=False)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert db.claim_pending_review("req_x") == "missing"

    def test_claim_is_one_conditional_update_not_read_then_write(self):
        """The update must be attempted first; a read-then-write ordering is
        the race this exists to close."""
        db, session, query = self._db(rows_changed=1)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            db.claim_pending_review("req_x")

        query.update.assert_called_once()
        assert query.update.call_args.args[0] == {"verdict": "REVIEWING"}
        query.first.assert_not_called()   # no read needed on the happy path

    def test_claim_is_conditional_on_the_row_still_being_pending(self):
        """The condition is the whole point, so assert it explicitly.

        Asserting only that the UPDATE sets REVIEWING passes even if the
        `verdict == PENDING` predicate is deleted -- which would let a second
        reviewer seize a row the first already holds. Verified by removing the
        predicate: without this test the suite stayed green.
        """
        db, session, _ = self._db(rows_changed=1)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            db.claim_pending_review("req_x")

        criteria = self._filter_sql(session)
        assert "request_id" in criteria
        assert "PENDING" in criteria, (
            f"claim is not conditional on PENDING; filtered on: {criteria}")

    def test_release_is_conditional_on_the_row_being_reviewing(self):
        db, session, _ = self._db(rows_changed=1)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            db.release_review_claim("req_x")

        criteria = self._filter_sql(session)
        assert "REVIEWING" in criteria, (
            f"release is not conditional on REVIEWING; filtered on: {criteria}")

    def test_release_returns_a_claimed_row_to_pending(self):
        db, session, query = self._db(rows_changed=1)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert db.release_review_claim("req_x") is True
        assert query.update.call_args.args[0] == {"verdict": "PENDING"}

    def test_finalize_refuses_a_row_that_is_not_reviewing(self):
        """Guards against finalizing a claim this caller never held."""
        from finvet.audit.database import AuditDatabase

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: False

        db = AuditDatabase.__new__(AuditDatabase)
        with patch("finvet.audit.database.get_db_session", lambda: session):
            assert db.finalize_review(
                "req_x", verdict="REFUTES", confidence=0.9,
                final_response={}, data_sources=None, events=[],
                review_decision="override") is False

    def test_finalize_rewrites_the_envelope_and_its_checksum(self):
        """verdict and confidence alone left the stored response and digest
        describing the pending-era run."""
        from finvet.audit.database import AuditDatabase
        from finvet.audit.integrity import verify_execution_checksum

        row = MagicMock()
        row.claim_text = "Apple was fined EUR 500 million"
        row.agents_run = ["News"]
        row.execution_hash = "old-digest"

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = row
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: False

        db = AuditDatabase.__new__(AuditDatabase)
        with patch("finvet.audit.database.get_db_session", lambda: session), \
             patch.object(AuditDatabase, "_insert_missing_events", lambda *a: None):
            ok = db.finalize_review(
                "req_x", verdict="REFUTES", confidence=0.95,
                final_response={"verdict": "REFUTES"},
                data_sources={"xbrl": {"used": True}},
                events=[{"event_id": "evt_1"}],
                review_decision="override", reviewer_notes="disagreed")

        assert ok is True
        assert row.verdict == "REFUTES"
        assert row.total_events == 1
        assert row.full_trace["review"]["decision"] == "override"
        assert row.full_trace["review"]["supersedes_checksum"] == "old-digest"
        assert row.execution_hash != "old-digest"
        assert verify_execution_checksum(row.full_trace, row.execution_hash)
