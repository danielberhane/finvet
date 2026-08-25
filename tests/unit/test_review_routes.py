"""Identity and concurrency contract for POST /review/{request_id}.

The route resumes a paused graph, and every resume failure falls into a
fallback that computes a verdict directly. Because the fallback does not first
establish that the request exists or is still pending, a review can be accepted
for an id that was never verified, and two reviewers can both "succeed" with
conflicting verdicts.

These tests call the real handler; they do not assert against a hand-built
response shape.
"""

from unittest.mock import MagicMock

import pytest

from finvet.api.models import HITLReviewRequest
from finvet.api.routes import review as review_route


@pytest.fixture
def audit():
    logger = MagicMock()
    logger.update_execution_verdict.return_value = False
    return logger


@pytest.fixture(autouse=True)
def _wire(monkeypatch, audit):
    monkeypatch.setattr(review_route, "get_audit_logger", lambda: audit)
    yield


def _graph_without_checkpoint():
    """A resume against an unknown thread_id: no checkpoint exists."""
    graph = MagicMock()
    graph.update_state.side_effect = Exception("no checkpoint for thread")
    return graph


class TestReviewIdentity:

    def test_unknown_request_id_is_rejected(self, monkeypatch, audit):
        """A review for a request that was never verified must not succeed.

        Today the resume raises, the fallback computes REFUTES from the
        submitted override, update_execution_verdict returns False, that return
        is discarded, and the caller receives status="reviewed".
        """
        audit.get_execution.return_value = None
        monkeypatch.setattr(review_route.deps, "verification_graph",
                            _graph_without_checkpoint())

        with pytest.raises(Exception) as excinfo:
            review_route.submit_hitl_review(
                "req_does_not_exist",
                HITLReviewRequest(decision="override",
                                  override_verdict="REFUTES"),
            )

        assert getattr(excinfo.value, "status_code", None) == 404

    def test_already_reviewed_request_is_rejected(self, monkeypatch, audit):
        """Second review of a settled claim must conflict, not overwrite.

        Without a status predicate on the update, two reviewers submitting
        opposite verdicts both report success and the last commit wins.
        """
        audit.get_execution.return_value = {"verdict": "SUPPORTS"}
        monkeypatch.setattr(review_route.deps, "verification_graph",
                            _graph_without_checkpoint())

        with pytest.raises(Exception) as excinfo:
            review_route.submit_hitl_review(
                "req_already_done",
                HITLReviewRequest(decision="approve"),
            )

        assert getattr(excinfo.value, "status_code", None) == 409

    def test_pending_request_is_accepted(self, monkeypatch, audit):
        """The happy path must keep working once the guards are added."""
        audit.get_execution.return_value = {"verdict": "PENDING"}
        audit.update_execution_verdict.return_value = True
        graph = MagicMock()
        graph.invoke.return_value = {
            "final_response": {"verdict": "REFUTES", "confidence": 0.95}}
        monkeypatch.setattr(review_route.deps, "verification_graph", graph)

        out = review_route.submit_hitl_review(
            "req_pending",
            HITLReviewRequest(decision="override", override_verdict="REFUTES"),
        )

        assert out["status"] == "reviewed"
        assert out["verdict"] == "REFUTES"


class TestReviewPersistence:

    def test_failed_persistence_is_not_reported_as_success(self, monkeypatch, audit):
        """update_execution_verdict returns a boolean that is currently ignored.

        A review that did not persist must not be reported as reviewed.
        """
        audit.get_execution.return_value = {"verdict": "PENDING"}
        audit.update_execution_verdict.return_value = False
        graph = MagicMock()
        graph.invoke.return_value = {
            "final_response": {"verdict": "REFUTES", "confidence": 0.95}}
        monkeypatch.setattr(review_route.deps, "verification_graph", graph)

        with pytest.raises(Exception) as excinfo:
            review_route.submit_hitl_review(
                "req_pending",
                HITLReviewRequest(decision="override",
                                  override_verdict="REFUTES"),
            )

        assert getattr(excinfo.value, "status_code", None) == 500
