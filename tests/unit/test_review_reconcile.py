"""Retrying a review that ran but was never recorded.

A row reaches `REVIEW_FINALIZATION_FAILED` when the graph was entered and no
durable audited outcome followed. The checkpoint may still hold the reviewed
result. Reconciliation reads it and writes the audit row.

What reconciliation must never do is run the graph again. `update_state` and
`invoke` are both forbidden here: the review already happened, and re-entering
the graph would either duplicate side effects or produce a second, different
answer for one human decision. Reconciliation is a write of an outcome that
already exists, not a second attempt at producing one.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from finvet.api.routes import review as review_route

FINAL = {"verdict": "REFUTES", "confidence": 0.95,
         "metadata": {"data_sources": {"xbrl": {"used": True}}}}


@pytest.fixture
def audit():
    logger = MagicMock()
    logger.claim_review_finalization.return_value = "claimed"
    logger.finalize_review.return_value = True
    logger.restore_review_finalization_failed.return_value = True
    logger.get_events.return_value = []
    logger.get_execution.return_value = {
        "request_id": "req_stuck",
        "full_trace": {"review_recovery": {"review_decision": "approve",
                                           "reviewer_notes": "looked fine"}},
    }
    return logger


@pytest.fixture
def graph():
    g = MagicMock()
    snapshot = MagicMock()
    snapshot.values = {"final_response": FINAL}
    g.get_state.return_value = snapshot
    return g


@pytest.fixture(autouse=True)
def _wire(monkeypatch, audit, graph):
    monkeypatch.setattr(review_route, "get_audit_logger", lambda: audit)
    monkeypatch.setattr(review_route.deps, "verification_graph", graph)
    yield


def _reconcile(request_id="req_stuck"):
    return review_route.reconcile_review(request_id)


def _status(excinfo):
    return getattr(excinfo.value, "status_code", None)


class TestTheGraphIsNeverRunAgain:
    """The single most important property of this endpoint."""

    def test_reconciliation_does_not_invoke(self, graph):
        _reconcile()
        graph.invoke.assert_not_called()

    def test_reconciliation_does_not_update_state(self, graph):
        _reconcile()
        graph.update_state.assert_not_called()

    def test_it_reads_the_existing_checkpoint(self, graph):
        _reconcile()
        graph.get_state.assert_called_once()
        config = graph.get_state.call_args.args[0]
        assert config["configurable"]["thread_id"] == "req_stuck"

    def test_neither_is_called_even_when_the_checkpoint_is_empty(self, graph):
        graph.get_state.return_value = MagicMock(values={})

        with pytest.raises(HTTPException):
            _reconcile()

        graph.invoke.assert_not_called()
        graph.update_state.assert_not_called()


class TestOnlyAStuckRowCanBeReconciled:

    @pytest.mark.parametrize("claim_result,expected", [
        ("missing", 404),
        ("conflict", 409),
        ("unavailable", 503),
    ])
    def test_claim_outcomes_are_distinct(self, audit, claim_result, expected):
        audit.claim_review_finalization.return_value = claim_result

        with pytest.raises(HTTPException) as excinfo:
            _reconcile()

        assert _status(excinfo) == expected

    def test_a_failed_claim_finalizes_nothing(self, audit, graph):
        audit.claim_review_finalization.return_value = "conflict"

        with pytest.raises(HTTPException):
            _reconcile()

        assert audit.finalize_review.call_count == 0
        graph.get_state.assert_not_called()

    def test_two_concurrent_retries_produce_one_winner(self, audit):
        """The claim is a single conditional UPDATE, so the second caller sees
        a conflict and cannot finalize the same review twice."""
        audit.claim_review_finalization.side_effect = ["claimed", "conflict"]

        assert _reconcile()["status"] == "reviewed"
        with pytest.raises(HTTPException) as excinfo:
            _reconcile()

        assert _status(excinfo) == 409
        assert audit.finalize_review.call_count == 1


class TestTheCheckpointResultIsValidated:

    @pytest.mark.parametrize("values", [
        {},
        {"final_response": None},
        {"final_response": {}},
        {"final_response": {"confidence": 0.9}},
        {"final_response": {"verdict": None}},
        {"final_response": "not a dict"},
    ], ids=["empty", "none", "empty-dict", "no-verdict", "null-verdict",
            "not-a-dict"])
    def test_an_invalid_checkpoint_result_is_not_finalized(self, audit, graph,
                                                           values):
        graph.get_state.return_value = MagicMock(values=values)

        with pytest.raises(HTTPException):
            _reconcile()

        assert audit.finalize_review.call_count == 0

    def test_a_missing_checkpoint_is_reported_not_fabricated(self, audit, graph):
        graph.get_state.side_effect = Exception("no checkpoint for thread")

        with pytest.raises(HTTPException) as excinfo:
            _reconcile()

        assert _status(excinfo) in (409, 503)
        assert audit.finalize_review.call_count == 0


class TestAFailedRetryStaysDiscoverable:
    """Claiming moves the row out of the recovery state. If the retry then
    fails, it must go back -- otherwise it sits in REVIEWING with nobody
    working on it and no operator report reason to look."""

    def test_an_invalid_checkpoint_restores_the_recovery_state(self, audit,
                                                               graph):
        graph.get_state.return_value = MagicMock(values={})

        with pytest.raises(HTTPException):
            _reconcile()

        audit.restore_review_finalization_failed.assert_called_once_with(
            "req_stuck")

    def test_a_failed_write_restores_the_recovery_state(self, audit):
        audit.finalize_review.return_value = False

        with pytest.raises(HTTPException):
            _reconcile()

        audit.restore_review_finalization_failed.assert_called_once_with(
            "req_stuck")

    def test_a_successful_retry_does_not_restore_it(self, audit):
        _reconcile()
        audit.restore_review_finalization_failed.assert_not_called()


class TestSuccessfulReconciliation:

    def test_reviewed_is_returned_only_after_a_durable_write(self, audit):
        out = _reconcile()
        assert out["status"] == "reviewed"
        assert out["verdict"] == "REFUTES"

    def test_it_finalizes_with_the_checkpoint_result(self, audit):
        _reconcile()
        kwargs = audit.finalize_review.call_args.kwargs
        assert kwargs["verdict"] == "REFUTES"
        assert kwargs["final_response"] == FINAL

    def test_it_reuses_the_persisted_review_decision(self, audit):
        """The reviewer decided once. Reconciliation must not invent a new
        decision or ask for one."""
        _reconcile()
        kwargs = audit.finalize_review.call_args.kwargs
        assert kwargs["review_decision"] == "approve"
        assert kwargs["reviewer_notes"] == "looked fine"

    def test_the_buffer_is_drained(self, audit):
        _reconcile()
        audit.discard_buffer.assert_called_once_with("req_stuck")

    @pytest.mark.parametrize("break_it", [
        lambda a, g: setattr(g.get_state, "side_effect", Exception("x")),
        lambda a, g: setattr(g.get_state, "return_value", MagicMock(values={})),
        lambda a, g: setattr(a.finalize_review, "return_value", False),
    ], ids=["no-checkpoint", "invalid-result", "failed-write"])
    def test_every_failure_path_drains_the_buffer(self, audit, graph, break_it):
        break_it(audit, graph)

        with pytest.raises(HTTPException):
            _reconcile()

        audit.discard_buffer.assert_called_once_with("req_stuck")
