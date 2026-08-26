"""Integrity over the whole displayed record, in the right order.

Two defects this pins down.

**Order.** Checking review-lifecycle state before recomputing the checksum lets
a genuinely corrupted `REVIEWING` row report `unavailable` -- "not checked" --
when the checksum would have proved tampering. Lifecycle state is a reason to
skip the *projection* comparison, never a reason to skip the checksum. The
checksum is cheap, self-contained, and the only part of this that detects a
changed snapshot; it runs first, always.

**Scope.** The row stores claim, verdict, confidence, agents_run and
data_sources denormalized beside `full_trace`, and the event rows are queried
separately. A checksum over the envelope alone leaves all of that free to drift
while the API reports `verified`.

Status contract:

- `verified`     -- checksum passes and the projection matches.
- `failed`       -- checksum mismatch, or projection mismatch.
- `unavailable`  -- no checksum recorded, or a checksum that passes on a row
                    whose lifecycle state the envelope cannot cover.
"""

import copy
import math

import pytest

from finvet.api.routes.audit import _integrity_for
from finvet.audit.integrity import (
    build_execution_envelope,
    compute_execution_checksum,
)

EVENTS = [{
    "event_id": "evt_1", "request_id": "req_1", "parent_event_id": None,
    "event_type": "input_received", "timestamp": "2026-08-25T12:00:00",
    "agent": None, "data": {"claim_raw": "Revenue was 100"},
}]


def _record(*, verdict="SUPPORTS", terminal_status="success", events=EVENTS):
    envelope = build_execution_envelope(
        request_id="req_1", claim_text="Revenue was 100",
        terminal_status=terminal_status, verdict=verdict, confidence=0.9,
        agents_run=["SEC"], events=events,
        final_response={"status": "success", "verdict": verdict},
        data_sources={"xbrl": {"used": True}})
    return {
        "request_id": "req_1", "claim_text": "Revenue was 100",
        "verdict": verdict, "confidence": 0.9, "agents_run": ["SEC"],
        "data_sources": {"xbrl": {"used": True}},
        "full_trace": envelope,
        "execution_hash": compute_execution_checksum(envelope),
    }


@pytest.fixture
def terminal_record():
    return _record()


@pytest.fixture
def reviewing_record():
    """A pending row a reviewer has claimed.

    `claim_pending_review` writes REVIEWING to the verdict column with a single
    atomic UPDATE and `submit_hitl_review` logs its hitl_* event straight to the
    database. Both are intended, and neither is covered by the committed
    envelope.
    """
    record = _record(verdict="PENDING", terminal_status="pending_review",
                     events=[])
    record["verdict"] = "REVIEWING"      # the column, not the envelope
    return record


REVIEW_EVENT = {
    "event_id": "evt_9", "request_id": "req_1", "parent_event_id": None,
    "event_type": "hitl_approve", "timestamp": "2026-08-25T13:00:00",
    "agent": None, "data": {"decision": "approve"},
}


class TestTheControl:
    """Without these, every assertion below could pass vacuously."""

    def test_a_valid_terminal_execution_verifies(self, terminal_record):
        result = _integrity_for(terminal_record, EVENTS)
        assert result["status"] == "verified"
        assert result["mismatches"] == []

    def test_storage_metadata_does_not_create_a_false_mismatch(self,
                                                               terminal_record):
        """get_events() returns created_at; the envelope never held it."""
        rows = [{**EVENTS[0], "created_at": "2026-08-25T12:00:01"}]
        assert _integrity_for(terminal_record, rows)["status"] == "verified"


class TestChecksumIsCheckedBeforeLifecycleState:
    """The ordering defect, stated directly."""

    def test_a_corrupted_reviewing_envelope_fails_not_unavailable(
            self, reviewing_record):
        """A REVIEWING row whose envelope was edited must not hide behind
        'not checked'. Lifecycle state excuses the projection comparison, not
        the checksum."""
        reviewing_record["full_trace"]["claim"] = "Revenue was 999"

        result = _integrity_for(reviewing_record, [REVIEW_EVENT])

        assert result["status"] == "failed", (
            "a corrupted record under review reported "
            f"{result['status']!r} -- the checksum was never recomputed")
        assert result["reason"] == "checksum_mismatch"

    def test_a_tampered_stored_checksum_on_a_reviewing_row_fails(
            self, reviewing_record):
        reviewing_record["execution_hash"] = "0" * 64

        result = _integrity_for(reviewing_record, [REVIEW_EVENT])

        assert result["status"] == "failed"
        assert result["reason"] == "checksum_mismatch"

    def test_a_corrupted_finalization_failed_row_also_fails(self):
        record = _record(verdict="PENDING",
                         terminal_status="review_finalization_failed",
                         events=[])
        record["verdict"] = "REVIEW_FINALIZATION_FAILED"
        record["full_trace"]["confidence"] = 0.1

        assert _integrity_for(record, [])["status"] == "failed"


class TestLifecycleStateIsReportedNotChecked:
    """Once the checksum passes, an uncoverable divergence is 'not checked'."""

    @pytest.mark.parametrize("verdict,reason", [
        ("REVIEWING", "review_in_progress"),
        ("REVIEW_FINALIZATION_FAILED", "review_finalization_failed"),
    ])
    def test_intact_non_terminal_rows_are_unavailable(self, verdict, reason):
        record = _record(verdict="PENDING", terminal_status="pending_review",
                         events=[])
        record["verdict"] = verdict

        result = _integrity_for(record, [REVIEW_EVENT])

        assert result["status"] == "unavailable"
        assert result["reason"] == reason

    def test_an_unclaimed_pending_row_still_verifies(self):
        """PENDING is not a divergence: nothing has been written past it."""
        record = _record(verdict="PENDING", terminal_status="pending_review",
                         events=[])
        assert _integrity_for(record, [])["status"] == "verified"

    def test_missing_checksum_is_unavailable(self, terminal_record):
        terminal_record["execution_hash"] = None
        result = _integrity_for(terminal_record, EVENTS)
        assert result["status"] == "unavailable"
        assert result["reason"] == "no_checksum_recorded"


class TestProjectionMismatchWithAValidChecksum:
    """The envelope verifies; the row beside it does not agree."""

    @pytest.mark.parametrize("field,value", [
        ("claim_text", "Revenue was 999"),
        ("verdict", "REFUTES"),
        ("confidence", 0.1),
        ("agents_run", ["NEWS"]),
        ("data_sources", {"news": {"used": True}}),
    ])
    def test_a_changed_row_column_fails(self, terminal_record, field, value):
        altered = copy.deepcopy(terminal_record)
        altered[field] = value

        result = _integrity_for(altered, EVENTS)

        assert result["status"] == "failed"
        assert result["reason"] == "projection_mismatch"

    @pytest.mark.parametrize("mutate", [
        lambda e: [{**e[0], "event_type": "tampered"}],
        lambda e: [{**e[0], "timestamp": "2020-01-01T00:00:00"}],
        lambda e: [{**e[0], "data": {"claim_raw": "something else"}}],
        lambda e: [],
        lambda e: e + [{**e[0], "event_id": "evt_2"}],
    ], ids=["type", "timestamp", "data", "deleted", "inserted"])
    def test_a_changed_event_row_fails(self, terminal_record, mutate):
        result = _integrity_for(terminal_record, mutate(EVENTS))
        assert result["status"] == "failed"
        assert result["reason"] == "projection_mismatch"
        assert "events" in result["mismatches"]


class TestChecksumCanonicalization:
    """Serialization must refuse what it cannot represent faithfully."""

    def test_nan_is_rejected(self):
        from finvet.audit.integrity import compute_execution_checksum

        with pytest.raises((ValueError, TypeError)):
            compute_execution_checksum({"confidence": float("nan")})

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_infinity_is_rejected(self, value):
        from finvet.audit.integrity import compute_execution_checksum

        with pytest.raises((ValueError, TypeError)):
            compute_execution_checksum({"confidence": value})

    def test_an_unsupported_type_is_rejected_not_stringified(self):
        """default=str turned anything unknown into its repr, so two different
        objects could hash the same and a changed object could hash unchanged."""
        from finvet.audit.integrity import compute_execution_checksum

        class Opaque:
            pass

        with pytest.raises(TypeError):
            compute_execution_checksum({"thing": Opaque()})

    def test_supported_scalars_survive(self):
        from datetime import datetime
        from decimal import Decimal

        from finvet.audit.integrity import compute_execution_checksum

        digest = compute_execution_checksum({
            "when": datetime(2026, 8, 25, 12, 0, 0),
            "amount": Decimal("391035000000.00"),
            "nested": {"list": [1, 2.5, True, None, "x"]},
        })
        assert len(digest) == 64

    def test_finite_floats_are_accepted(self):
        from finvet.audit.integrity import compute_execution_checksum

        assert math.isfinite(0.9)
        assert len(compute_execution_checksum({"confidence": 0.9})) == 64
