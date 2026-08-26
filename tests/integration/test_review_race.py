"""Two reviewers, one pending execution, a real database.

Spec invariant 6: only one reviewer can claim a pending execution. The unit
tests mock the session, so they verify that the UPDATE carries a PENDING
predicate but not that Postgres actually serialises two concurrent claims.
This runs both claims against the real database from separate threads.

Opt in with -m integration; skipped when Postgres is unreachable.
"""

import threading
import uuid

import pytest

pytestmark = pytest.mark.integration


def _db_or_skip():
    try:
        from sqlalchemy import text

        from finvet.audit.database import AuditDatabase
        from finvet.config.database import get_db_session

        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        return AuditDatabase()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres unavailable: {exc}")


@pytest.fixture
def pending_execution():
    """A committed PENDING execution, removed afterwards."""
    from sqlalchemy import text

    from finvet.config.database import get_db_session

    db = _db_or_skip()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    assert db.commit_execution(
        request_id=request_id,
        claim_text="race-test claim",
        verdict="PENDING",
        confidence=0.0,
        agents_run=["SEC"],
        events=[],
        execution_time_ms=1,
        final_response={"status": "pending_review", "verdict": "PENDING"},
        terminal_status="pending_review",
    )
    yield db, request_id

    with get_db_session() as session:
        session.execute(
            text("DELETE FROM audit_executions WHERE request_id = :r"),
            {"r": request_id})
        session.execute(
            text("DELETE FROM audit_events WHERE request_id = :r"),
            {"r": request_id})


class TestOnlyOneReviewerWins:

    def test_two_concurrent_claims_produce_one_winner(self, pending_execution):
        db, request_id = pending_execution
        results = []
        barrier = threading.Barrier(2)

        def claim():
            barrier.wait()          # maximise overlap
            results.append(db.claim_pending_review(request_id))

        threads = [threading.Thread(target=claim) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["claimed", "conflict"], (
            f"both sessions saw {results}; the claim is not serialised")

    def test_a_claimed_row_cannot_be_claimed_again(self, pending_execution):
        db, request_id = pending_execution
        assert db.claim_pending_review(request_id) == "claimed"
        assert db.claim_pending_review(request_id) == "conflict"

    def test_unknown_id_is_missing_not_conflict(self, pending_execution):
        db, _ = pending_execution
        assert db.claim_pending_review("req_ffffffffffff") == "missing"

    def test_a_pre_invoke_failure_can_release_the_claim(self, pending_execution):
        """`release_review_claim` is still required, for exactly one case: a
        failure before the graph is entered, where nothing has run and the row
        is safely retryable.

        The route no longer calls this after invoke has begun -- see
        tests/unit/test_review_lifecycle.py -- but the transition itself must
        keep working, or a lost checkpoint would strand every claim.
        """
        db, request_id = pending_execution
        assert db.claim_pending_review(request_id) == "claimed"
        assert db.release_review_claim(request_id) is True
        assert db.claim_pending_review(request_id) == "claimed"


class TestStuckClaimsAreRecoverable:
    """A row left in REVIEWING by a crashed reviewer must not vanish.

    Spec invariant 4 requires an explicit, recoverable finalization-failure
    state; a claim that disappears from every reviewer's queue is neither.
    """

    def test_a_reviewing_row_is_still_listed(self, pending_execution):
        db, request_id = pending_execution
        assert db.claim_pending_review(request_id) == "claimed"

        listed = {row.get("request_id") for row in db.get_pending_reviews()}
        assert request_id in listed, (
            "a claimed row vanished from the review queue with no way back")

    def test_storage_failure_is_not_reported_as_missing(self, monkeypatch,
                                                        pending_execution):
        """A database outage must not look like 'no such request', which the
        route turns into a 404."""
        from finvet.audit import database as dbmod

        db, request_id = pending_execution

        def boom():
            raise RuntimeError("connection reset")

        monkeypatch.setattr(dbmod, "get_db_session", boom)
        assert db.claim_pending_review(request_id) != "missing"


class TestTheRecoveryStateAgainstRealPostgres:
    """The transitions a stuck review depends on, serialised by the database.

    Unit tests assert the route calls these; only the database can show that
    two operators retrying the same row cannot both finalize it.
    """

    def _claimed(self, db, request_id):
        assert db.claim_pending_review(request_id) == "claimed"
        return request_id

    def test_marking_moves_a_claimed_row_to_the_recovery_state(
            self, pending_execution):
        from finvet.audit.database import REVIEW_FINALIZATION_FAILED

        db, request_id = pending_execution
        self._claimed(db, request_id)

        assert db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes="n",
            checkpoint_has_final_state=True) is True

        row = db.get_execution(request_id)
        assert row["verdict"] == REVIEW_FINALIZATION_FAILED
        assert row["full_trace"]["terminal_status"] == "review_finalization_failed"

    def test_marking_refuses_a_row_that_is_not_being_reviewed(
            self, pending_execution):
        """PENDING is not a state the recovery marker may be written over."""
        db, request_id = pending_execution

        assert db.mark_review_finalization_failed(
            request_id, error_type="x", review_decision="approve",
            reviewer_notes=None, checkpoint_has_final_state=False) is False

    def test_the_recovery_descriptor_holds_no_unaudited_verdict(
            self, pending_execution):
        db, request_id = pending_execution
        self._claimed(db, request_id)
        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="override", reviewer_notes="notes",
            checkpoint_has_final_state=True)

        recovery = db.get_execution(request_id)["full_trace"]["review_recovery"]
        assert recovery["review_decision"] == "override"
        assert recovery["checkpoint_has_final_state"] is True
        assert "verdict" not in recovery
        assert "confidence" not in recovery

    def test_the_descriptor_does_not_claim_a_final_state_it_lacks(
            self, pending_execution):
        db, request_id = pending_execution
        self._claimed(db, request_id)
        db.mark_review_finalization_failed(
            request_id, error_type="review_result_missing",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=False)

        recovery = db.get_execution(request_id)["full_trace"]["review_recovery"]
        assert recovery["checkpoint_has_final_state"] is False

    def test_the_checksum_is_recomputed_when_the_envelope_changes(
            self, pending_execution):
        from finvet.audit.integrity import (
            compute_execution_checksum, verify_execution_checksum)

        db, request_id = pending_execution
        self._claimed(db, request_id)
        before = db.get_execution(request_id)["execution_hash"]

        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=True)

        row = db.get_execution(request_id)
        assert row["execution_hash"] != before, (
            "the envelope changed but its digest did not, so integrity would "
            "report a mismatch on a row nothing tampered with")
        assert verify_execution_checksum(row["full_trace"],
                                         row["execution_hash"]) is True
        assert compute_execution_checksum(row["full_trace"]) == row["execution_hash"]

    def test_only_one_of_two_concurrent_retries_claims_the_row(
            self, pending_execution):
        import threading

        db, request_id = pending_execution
        self._claimed(db, request_id)
        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=True)

        results = []
        barrier = threading.Barrier(2)

        def retry():
            barrier.wait()
            results.append(db.claim_review_finalization(request_id))

        threads = [threading.Thread(target=retry) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["claimed", "conflict"], (
            f"both operators saw {results}; the same review could be "
            f"finalized twice")

    def test_a_failed_retry_restores_the_recovery_state(self, pending_execution):
        from finvet.audit.database import REVIEW_FINALIZATION_FAILED

        db, request_id = pending_execution
        self._claimed(db, request_id)
        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=True)

        assert db.claim_review_finalization(request_id) == "claimed"
        assert db.restore_review_finalization_failed(request_id) is True
        assert db.get_execution(request_id)["verdict"] == REVIEW_FINALIZATION_FAILED
        # And it can be claimed again afterwards.
        assert db.claim_review_finalization(request_id) == "claimed"

    def test_claiming_a_row_that_is_not_stuck_conflicts(self, pending_execution):
        db, request_id = pending_execution
        assert db.claim_review_finalization(request_id) == "conflict"

    def test_an_unknown_row_is_missing_not_conflict(self, pending_execution):
        db, _ = pending_execution
        assert db.claim_review_finalization("req_ffffffffffff") == "missing"


class TestTheQueueSurfacesEveryUnfinishedReview:

    def test_a_stuck_review_appears_with_its_status(self, pending_execution):
        db, request_id = pending_execution
        assert db.claim_pending_review(request_id) == "claimed"
        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=True)

        listed = {row["request_id"]: row for row in db.get_pending_reviews()}

        assert request_id in listed, (
            "a review awaiting reconciliation vanished from the only queue a "
            "reviewer looks at")
        assert listed[request_id]["review_status"] == "finalization_failed"

    def test_a_claimed_row_reports_in_review(self, pending_execution):
        db, request_id = pending_execution
        db.claim_pending_review(request_id)

        listed = {row["request_id"]: row for row in db.get_pending_reviews()}
        assert listed[request_id]["review_status"] == "in_review"

    def test_an_unclaimed_row_reports_pending(self, pending_execution):
        db, request_id = pending_execution

        listed = {row["request_id"]: row for row in db.get_pending_reviews()}
        assert listed[request_id]["review_status"] == "pending"


class TestTheOperatorReportSeesStuckRows:

    def test_a_row_awaiting_reconciliation_is_reported(self, pending_execution):
        from finvet.audit.operator_report import build_operator_report

        db, request_id = pending_execution
        db.claim_pending_review(request_id)
        db.mark_review_finalization_failed(
            request_id, error_type="audit_persistence_failed",
            review_decision="approve", reviewer_notes=None,
            checkpoint_has_final_state=True)

        report = build_operator_report()

        assert request_id in {r["request_id"]
                              for r in report.awaiting_reconciliation}
        assert report.is_clean is False

    def test_a_freshly_claimed_row_is_not_yet_called_stuck(self,
                                                           pending_execution):
        """It is someone's work in progress until the threshold passes."""
        from finvet.audit.operator_report import build_operator_report

        db, request_id = pending_execution
        db.claim_pending_review(request_id)

        report = build_operator_report(stale_after_minutes=60)

        assert request_id not in {r["request_id"] for r in report.stuck_reviews}

    def test_an_old_claimed_row_is_reported_as_stuck(self, pending_execution):
        from finvet.audit.operator_report import build_operator_report

        db, request_id = pending_execution
        db.claim_pending_review(request_id)

        # Threshold of zero: everything already claimed is past it.
        report = build_operator_report(stale_after_minutes=0)

        assert request_id in {r["request_id"] for r in report.stuck_reviews}

    def test_the_report_writes_nothing(self, pending_execution):
        from finvet.audit.operator_report import build_operator_report

        db, request_id = pending_execution
        db.claim_pending_review(request_id)
        before = db.get_execution(request_id)

        build_operator_report(stale_after_minutes=0)

        assert db.get_execution(request_id) == before
