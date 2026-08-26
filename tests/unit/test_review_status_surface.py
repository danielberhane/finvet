"""What the queue tells a reviewer about a row's state.

`get_pending_reviews` returned PENDING and REVIEWING rows with no field
distinguishing them, so the UI rendered a decision form for every one. A row in
`REVIEW_FINALIZATION_FAILED` needs a different action entirely -- its decision
was already made and its graph already ran; what it needs is a retry of the
audit write, not a second opinion.

Machine-readable, not inferred from prose: the UI must not have to guess from a
verdict string that doubles as lifecycle state.
"""

from unittest.mock import MagicMock, patch

import pytest

from finvet.audit.database import REVIEW_FINALIZATION_FAILED, AuditDatabase


def _rows(*verdicts):
    rows = []
    for index, verdict in enumerate(verdicts):
        row = MagicMock()
        row.request_id = f"req_{index}"
        row.claim_text = "a claim"
        row.timestamp = "2026-08-26T10:00:00"
        row.verdict = verdict
        row.full_trace = {"final_response": {}}
        rows.append(row)
    return rows


def _pending_reviews(rows):
    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = lambda s, *a: False
    (session.query.return_value.filter.return_value
     .order_by.return_value.all.return_value) = rows

    db = AuditDatabase.__new__(AuditDatabase)
    with patch("finvet.audit.database.get_db_session", lambda: session):
        return db.get_pending_reviews(), session


class TestTheQueueExposesReviewStatus:

    @pytest.mark.parametrize("verdict,expected", [
        ("PENDING", "pending"),
        ("REVIEWING", "in_review"),
        (REVIEW_FINALIZATION_FAILED, "finalization_failed"),
    ])
    def test_each_lifecycle_state_has_a_machine_readable_status(self, verdict,
                                                                expected):
        results, _ = _pending_reviews(_rows(verdict))
        assert results[0]["review_status"] == expected

    def test_finalization_failed_rows_are_listed(self):
        """Otherwise a stuck review is invisible to the only queue a reviewer
        looks at."""
        results, session = _pending_reviews(_rows(REVIEW_FINALIZATION_FAILED))

        assert len(results) == 1
        criteria = " ".join(
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in session.query.return_value.filter.call_args.args
        )
        assert REVIEW_FINALIZATION_FAILED in criteria, (
            f"stuck reviews are not queried: {criteria}")

    def test_pending_and_reviewing_are_still_listed(self):
        criteria_rows, session = _pending_reviews(_rows("PENDING", "REVIEWING"))
        assert len(criteria_rows) == 2
        criteria = " ".join(
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in session.query.return_value.filter.call_args.args
        )
        assert "PENDING" in criteria and "REVIEWING" in criteria

    def test_status_does_not_depend_on_reading_the_verdict_string(self):
        """The UI gets a field, not a convention it has to decode."""
        results, _ = _pending_reviews(_rows(REVIEW_FINALIZATION_FAILED))
        assert "review_status" in results[0]
        assert results[0]["review_status"] != REVIEW_FINALIZATION_FAILED


class TestTheUiOffersTheRightAction:
    """The decision form and the retry action are mutually exclusive."""

    def test_the_view_and_its_client_import_cleanly(self):
        """`ui/` files use absolute imports that only resolve with `ui/` on
        sys.path, which is what `streamlit run ui/app.py` provides. Importing
        the module the same way catches a broken import that a source grep
        would not."""
        import importlib
        import sys
        from pathlib import Path

        ui_dir = str(Path("ui").resolve())
        added = ui_dir not in sys.path
        if added:
            sys.path.insert(0, ui_dir)
        try:
            client = importlib.import_module("api_client")
            assert hasattr(client, "reconcile_review")
            importlib.import_module("views.review_detail")
        finally:
            if added:
                sys.path.remove(ui_dir)

    def test_the_review_detail_view_branches_on_review_status(self):
        from pathlib import Path

        source = Path("ui/views/review_detail.py").read_text()
        assert "review_status" in source, (
            "the review detail view does not read review_status, so it renders "
            "a decision form for a row whose decision was already made")
        assert "finalization_failed" in source
        assert "reconcile" in source
