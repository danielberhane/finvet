"""The review lifecycle, and the one boundary that cannot be crossed twice.

`submit_hitl_review` used to wrap `update_state()` and `invoke()` in a single
`try` whose handler always called `release_review_claim()`. That made every
resume failure look identical to a lost checkpoint, and returned the row to
`PENDING` even when the graph had already begun running. A second reviewer
could then resume a checkpoint the first had partially advanced.

The boundary is **the call to `invoke`, not its return**. A flag set after
`invoke()` returns cannot protect anything: the dangerous case is precisely the
one where `invoke` mutates the checkpoint and then raises, so the flag is never
set. Phase state is therefore committed before the call.

State transitions:

    PENDING --claim--> REVIEWING --finalize--> <terminal verdict>
                           |
                           +--pre-invoke failure--> PENDING   (retryable)
                           |
                           +--invoke began, no durable outcome-->
                                          REVIEW_FINALIZATION_FAILED
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

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
    logger.mark_review_finalization_failed.return_value = True
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


class TestClaimOutcomesAreDistinct:
    """A storage outage is not a missing row."""

    @pytest.mark.parametrize("claim_result,expected", [
        ("missing", 404),
        ("conflict", 409),
        ("unavailable", 503),
    ])
    def test_each_claim_outcome_has_its_own_status(self, audit, claim_result,
                                                   expected):
        audit.claim_pending_review.return_value = claim_result

        with pytest.raises(HTTPException) as excinfo:
            _submit()

        assert _status(excinfo) == expected

    def test_a_failed_claim_writes_nothing(self, audit, graph):
        audit.claim_pending_review.return_value = "unavailable"

        with pytest.raises(HTTPException):
            _submit()

        assert audit.finalize_review.call_count == 0
        assert audit.mark_review_finalization_failed.call_count == 0
        graph.invoke.assert_not_called()


class TestPreInvokeFailuresRelease:
    """Before the graph runs, the claim is safely reversible."""

    def test_a_lost_checkpoint_releases_the_claim(self, audit, graph):
        """update_state raising means nothing has run. Returning the row to
        PENDING is correct and necessary -- otherwise it strands in REVIEWING."""
        graph.update_state.side_effect = Exception("no checkpoint for thread")

        with pytest.raises(HTTPException) as excinfo:
            _submit()

        assert _status(excinfo) == 409
        assert excinfo.value.detail["error"] == "checkpoint_unavailable"
        audit.release_review_claim.assert_called_once_with("req_pending")
        graph.invoke.assert_not_called()

    def test_a_pre_invoke_failure_does_not_mark_finalization_failed(self,
                                                                    audit, graph):
        """The row is retryable, not stuck: it must not be pushed into the
        recovery state it does not need."""
        graph.update_state.side_effect = Exception("no checkpoint")

        with pytest.raises(HTTPException):
            _submit()

        assert audit.mark_review_finalization_failed.call_count == 0


class TestOnceInvokeBeginsTheClaimIsNeverReleased:
    """Stop condition 9. The row must never return to PENDING after the graph
    has been entered, or it can be resumed a second time."""

    def test_an_invoke_exception_does_not_release_the_claim(self, audit, graph):
        graph.invoke.side_effect = RuntimeError("node exploded")

        with pytest.raises(HTTPException):
            _submit()

        audit.release_review_claim.assert_not_called()

    def test_an_invoke_exception_marks_the_row_recoverable(self, audit, graph):
        graph.invoke.side_effect = RuntimeError("node exploded")

        with pytest.raises(HTTPException):
            _submit()

        audit.mark_review_finalization_failed.assert_called_once()

    def test_a_partial_run_that_raises_is_still_irreversible(self, audit, graph):
        """The case a post-return flag cannot catch: invoke mutates the
        checkpoint and then raises, so it never returns."""
        def mutate_then_fail(*args, **kwargs):
            graph.update_state(  # a write that really happened
                {"configurable": {"thread_id": "req_pending"}},
                {"partial": True})
            raise RuntimeError("failed after advancing two nodes")

        graph.invoke.side_effect = mutate_then_fail

        with pytest.raises(HTTPException):
            _submit()

        audit.release_review_claim.assert_not_called()

    def test_the_graph_is_invoked_at_most_once(self, audit, graph):
        _submit()
        assert graph.invoke.call_count == 1


class TestMissingFinalResponseIsNeverReviewed:
    """Stop condition 8. `verdict` used to default to the string "ERROR",
    which was then finalized and returned as a completed review."""

    @pytest.mark.parametrize("result", [
        {},
        {"final_response": None},
        {"final_response": {}},
        {"final_response": {"confidence": 0.9}},        # no verdict
        {"final_response": {"verdict": None}},
        {"final_response": "not a dict"},
    ], ids=["empty", "none", "empty-dict", "no-verdict", "null-verdict",
            "not-a-dict"])
    def test_an_invalid_result_is_not_reported_as_reviewed(self, audit, graph,
                                                           result):
        graph.invoke.return_value = result

        with pytest.raises(HTTPException) as excinfo:
            _submit()

        assert _status(excinfo) in (500, 503)
        assert audit.finalize_review.call_count == 0

    def test_no_error_verdict_is_ever_fabricated(self, audit, graph):
        graph.invoke.return_value = {}

        with pytest.raises(HTTPException):
            _submit()

        for call in audit.finalize_review.call_args_list:
            assert call.kwargs.get("verdict") != "ERROR"

    def test_a_missing_result_marks_the_row_recoverable(self, audit, graph):
        graph.invoke.return_value = {}

        with pytest.raises(HTTPException):
            _submit()

        audit.mark_review_finalization_failed.assert_called_once()
        kwargs = audit.mark_review_finalization_failed.call_args.kwargs
        assert kwargs["checkpoint_has_final_state"] is False, (
            "no valid final_response was produced, so the recovery descriptor "
            "must not claim the checkpoint holds one")


class TestFinalizationFailureIsRecoverable:

    def test_a_failed_finalization_is_not_reported_as_success(self, audit):
        audit.finalize_review.return_value = False

        with pytest.raises(HTTPException) as excinfo:
            _submit()

        assert _status(excinfo) in (500, 503)

    def test_a_failed_finalization_marks_the_row_recoverable(self, audit):
        audit.finalize_review.return_value = False

        with pytest.raises(HTTPException):
            _submit()

        kwargs = audit.mark_review_finalization_failed.call_args.kwargs
        assert kwargs["checkpoint_has_final_state"] is True, (
            "the graph produced a valid final_response; recovery can use it")

    def test_a_failed_finalization_never_releases_the_claim(self, audit):
        audit.finalize_review.return_value = False

        with pytest.raises(HTTPException):
            _submit()

        audit.release_review_claim.assert_not_called()

    def test_reviewed_is_returned_only_after_durable_finalization(self, audit):
        audit.finalize_review.return_value = True
        assert _submit()["status"] == "reviewed"


class TestTheBufferIsDrainedOnlyWhenNothingStillNeedsIt:
    """This class previously asserted the buffer drained on *every* exit, and
    that invariant was the defect.

    `log_event` writes best effort and keeps a buffered copy, so the buffer
    holds the only record of an event whose immediate write failed.
    Reconciliation is what needs that copy — and the paths that lead to
    reconciliation are precisely the failures the old rule drained on. By the
    time `/reconcile` ran, the recovered envelope could no longer be made
    whole.

    The rule now follows what happens to the row:

      * durable success ....................... drain; nothing else will run
      * back to PENDING (pre-invoke failure) .. drain; the review retries whole
      * REVIEW_FINALIZATION_FAILED ............ retain; /reconcile needs it
    """

    def test_success_drains_the_buffer(self, audit):
        _submit()
        audit.discard_buffer.assert_called_once_with("req_pending")

    def test_a_pre_invoke_failure_drains_it(self, audit, graph):
        """The row goes back to PENDING and the whole review can be retried
        from scratch, re-logging its events. Nothing needs the old copy."""
        graph.update_state.side_effect = Exception("x")

        with pytest.raises(HTTPException):
            _submit()

        audit.discard_buffer.assert_called_once_with("req_pending")

    @pytest.mark.parametrize("break_it", [
        lambda a, g: setattr(g.invoke, "side_effect", Exception("x")),
        lambda a, g: setattr(g.invoke, "return_value", {}),
        lambda a, g: setattr(a.finalize_review, "return_value", False),
    ], ids=["invoke", "no-result", "finalize"])
    def test_a_recoverable_failure_keeps_it(self, audit, graph, break_it):
        """Each of these leaves the row in REVIEW_FINALIZATION_FAILED, so
        reconciliation still has to assemble a complete envelope."""
        break_it(audit, graph)

        with pytest.raises(HTTPException):
            _submit()

        audit.discard_buffer.assert_not_called()
