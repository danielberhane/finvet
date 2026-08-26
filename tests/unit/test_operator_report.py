"""What an operator needs to see, and nothing it may change.

Three ways a run becomes invisible:

1. Events exist under a request id with no execution row -- a run that was
   never terminalized. Historical, from before terminalization was shared.
2. A row stuck in `REVIEWING` because the process marking it died. It is not
   pending, not reviewed, and no reconcile endpoint will take it.
3. A row in `REVIEW_FINALIZATION_FAILED`, waiting for someone to retry it.

An orphan inventory alone finds only the first. The other two are execution
rows, so they never appear in an events-without-executions query -- which is
why this report separates all three.

The report only reads. Repairing audit data automatically is how a
reconstructed guess becomes indistinguishable from a recorded fact.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from finvet.audit import operator_report as report_module
from finvet.audit.operator_report import (
    STALE_REVIEW_MINUTES,
    build_operator_report,
    stale_cutoff,
)


def _session(*, orphans=(), stuck=(), awaiting=()):
    """A session whose three queries return the given rows, in call order."""
    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = lambda s, *a: False

    session.query.return_value.filter.return_value.limit.return_value.all.side_effect = [
        [(r,) for r in orphans],
        list(stuck),
        list(awaiting),
    ]
    return session


def _row(request_id, timestamp, claim="a claim"):
    row = MagicMock()
    row.request_id = request_id
    row.timestamp = timestamp
    row.claim_text = claim
    return row


@pytest.fixture
def patched(monkeypatch):
    def _install(session):
        monkeypatch.setattr(report_module, "get_db_session", lambda: session)
        return session
    return _install


class TestTheThreeCategoriesAreSeparate:

    def test_orphan_events_are_reported(self, patched):
        patched(_session(orphans=["req_orphan1", "req_orphan2"]))

        result = build_operator_report()

        assert result.orphan_event_count == 2
        assert "req_orphan1" in result.orphan_event_request_ids

    def test_stuck_reviews_are_reported(self, patched):
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        patched(_session(stuck=[_row("req_stuck", old)]))

        result = build_operator_report()

        assert result.stuck_review_count == 1
        assert result.stuck_reviews[0]["request_id"] == "req_stuck"

    def test_rows_awaiting_reconciliation_are_reported(self, patched):
        patched(_session(awaiting=[_row("req_failed", "2026-08-25T10:00:00")]))

        result = build_operator_report()

        assert result.awaiting_reconciliation_count == 1
        assert result.awaiting_reconciliation[0]["request_id"] == "req_failed"

    def test_a_stuck_review_is_not_counted_as_an_orphan(self, patched):
        """The distinction the orphan inventory alone could not make: a stuck
        review has an execution row, so it is invisible to that query."""
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        patched(_session(orphans=[], stuck=[_row("req_stuck", old)]))

        result = build_operator_report()

        assert result.orphan_event_count == 0
        assert result.stuck_review_count == 1

    def test_an_empty_system_reports_nothing_wrong(self, patched):
        patched(_session())

        result = build_operator_report()

        assert result.orphan_event_count == 0
        assert result.stuck_review_count == 0
        assert result.awaiting_reconciliation_count == 0
        assert result.is_clean


class TestTheStalenessThresholdIsDocumented:

    def test_the_threshold_has_a_named_default(self):
        assert isinstance(STALE_REVIEW_MINUTES, int)
        assert STALE_REVIEW_MINUTES > 0

    def test_the_cutoff_is_the_threshold_before_now(self):
        cutoff = datetime.fromisoformat(stale_cutoff(60))
        delta = datetime.now(timezone.utc) - cutoff
        assert timedelta(minutes=59) < delta < timedelta(minutes=61)

    def test_the_report_states_the_threshold_it_used(self, patched):
        patched(_session())
        assert build_operator_report(stale_after_minutes=45).stale_after_minutes == 45

    def test_the_query_selects_reviews_older_than_the_cutoff(self, patched):
        """The predicate itself, not the arithmetic beside it.

        Asserting only that a fresh timestamp sorts after the cutoff passes
        even when the query's comparison is reversed -- verified by flipping
        `<` to `>`, which left the rest of this file green while the report
        listed every *active* review as stuck and hid every abandoned one.
        """
        session = patched(_session())
        build_operator_report()

        stuck_call = session.query.return_value.filter.call_args_list[1]
        criteria = " ".join(
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in stuck_call.args
        )
        assert "REVIEWING" in criteria, criteria
        assert "timestamp <" in criteria.replace("  ", " "), (
            f"stuck reviews are not selected as older than the cutoff: {criteria}")

    def test_a_fresh_review_is_below_the_cutoff(self):
        """Sanity on the cutoff arithmetic itself."""
        fresh = datetime.now(timezone.utc).isoformat()
        assert fresh > stale_cutoff(STALE_REVIEW_MINUTES)

    def test_an_abandoned_review_is_above_the_cutoff(self):
        old = (datetime.now(timezone.utc)
               - timedelta(minutes=STALE_REVIEW_MINUTES + 5)).isoformat()
        assert old < stale_cutoff(STALE_REVIEW_MINUTES)


class TestTheReportOnlyReads:
    """Automatic repair is how a guess becomes indistinguishable from a fact."""

    def test_nothing_is_written(self, patched):
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        session = patched(_session(orphans=["req_o"], stuck=[_row("req_s", old)],
                                   awaiting=[_row("req_f", old)]))

        build_operator_report()

        session.add.assert_not_called()
        session.delete.assert_not_called()
        session.merge.assert_not_called()
        session.add_all.assert_not_called()
        session.execute.assert_not_called()

    def test_no_query_is_turned_into_an_update(self, patched):
        session = patched(_session(orphans=["req_o"]))

        build_operator_report()

        assert session.query.return_value.filter.return_value.update.call_count == 0
        assert session.query.return_value.filter.return_value.delete.call_count == 0

    def test_the_module_exposes_no_repair_entry_point(self):
        """There is no --apply because there is nothing to apply."""
        exported = [name for name in dir(report_module)
                    if not name.startswith("_")]
        for forbidden in ("reconcile", "repair", "apply", "reconstruct",
                          "fix", "delete"):
            assert not any(forbidden in name.lower() for name in exported), (
                f"the read-only report exposes {forbidden!r}: {exported}")


class TestTheReportIsSerialisable:

    def test_it_renders_as_json(self, patched):
        import json

        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        patched(_session(orphans=["req_o"], stuck=[_row("req_s", old)]))

        payload = json.loads(json.dumps(build_operator_report().as_dict()))

        assert payload["orphan_event_count"] == 1
        assert payload["stuck_review_count"] == 1
        assert "generated_at" in payload
        assert payload["stale_after_minutes"] == STALE_REVIEW_MINUTES

    def test_a_storage_failure_is_reported_not_silently_empty(self, monkeypatch):
        """An unreachable database must not look like a clean system."""
        def boom():
            raise RuntimeError("connection reset")

        monkeypatch.setattr(report_module, "get_db_session", boom)

        result = build_operator_report()

        assert result.error is not None
        assert result.is_clean is False
