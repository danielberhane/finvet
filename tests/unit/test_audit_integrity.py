"""What the stored execution checksum actually covers.

The old hash was computed over the buffered events alone, while the row it sat
next to also stored the claim, the verdict, the confidence, the final response
and the data sources. Changing any of those left the hash unchanged, and the
audit UI printed a green "Verified" badge whenever the hash string was
non-empty -- it never recomputed anything.

These tests pin the replacement: one canonical envelope, hashed whole, stored
verbatim as what was hashed, and verified by recomputation rather than by
presence.
"""

import pytest

from finvet.audit.integrity import (
    build_execution_envelope,
    compute_execution_checksum,
    verify_execution_checksum,
)


def _envelope(**overrides):
    base = dict(
        request_id="req_abc123",
        claim_text="Apple FY2024 revenue was $391 billion",
        terminal_status="success",
        verdict="SUPPORTS",
        confidence=0.93,
        agents_run=["SEC"],
        events=[{"event_id": "evt_1", "event_type": "input_received",
                 "data": {"claim_raw": "Apple FY2024 revenue was $391 billion"}}],
        final_response={"verdict": "SUPPORTS", "confidence": 0.93,
                        "summary": "Filed revenue matches the claim."},
        data_sources={"xbrl": {"used": True, "concept": "Revenues"}},
    )
    base.update(overrides)
    return build_execution_envelope(**base)


class TestEnvelopeShape:

    def test_envelope_is_versioned(self):
        """A stored hash is only interpretable against a known layout."""
        assert _envelope()["schema_version"] == 2

    def test_envelope_carries_every_field_the_row_stores(self):
        envelope = _envelope()
        assert set(envelope) == {
            "schema_version", "request_id", "claim", "terminal_status",
            "verdict", "confidence", "agents_run", "events", "final_response",
            "data_sources", "llm_config",
        }


class TestChecksumIsStable:

    def test_same_envelope_same_checksum(self):
        assert compute_execution_checksum(_envelope()) == \
            compute_execution_checksum(_envelope())

    def test_key_order_does_not_change_the_checksum(self):
        """Canonicalisation must not depend on dict insertion order."""
        envelope = _envelope()
        reordered = dict(reversed(list(envelope.items())))
        assert compute_execution_checksum(reordered) == \
            compute_execution_checksum(envelope)

    def test_verify_accepts_its_own_checksum(self):
        envelope = _envelope()
        assert verify_execution_checksum(envelope,
                                         compute_execution_checksum(envelope))

    def test_verify_rejects_a_missing_checksum(self):
        assert verify_execution_checksum(_envelope(), None) is False
        assert verify_execution_checksum(_envelope(), "") is False


class TestEveryFieldIsCovered:
    """The defect that mattered: fields stored beside the hash but not in it.

    Each case changes one field and requires verification to fail. A field that
    can be edited without breaking the checksum is a field the checksum does
    not protect, whatever the UI says about it.
    """

    @pytest.mark.parametrize("field,mutated", [
        ("claim_text", "Apple FY2024 revenue was $1 billion"),
        ("terminal_status", "guardrail_blocked"),
        ("verdict", "REFUTES"),
        ("confidence", 0.10),
        ("agents_run", ["News"]),
        ("events", [{"event_id": "evt_1", "event_type": "input_received",
                     "data": {"claim_raw": "something else entirely"}}]),
        ("final_response", {"verdict": "REFUTES", "confidence": 0.93,
                            "summary": "Filed revenue matches the claim."}),
        ("data_sources", {"xbrl": {"used": False}}),
        ("request_id", "req_tampered"),
    ])
    def test_mutating_any_field_fails_verification(self, field, mutated):
        original = _envelope()
        checksum = compute_execution_checksum(original)

        assert verify_execution_checksum(_envelope(**{field: mutated}),
                                         checksum) is False, \
            f"{field} can be changed without invalidating the checksum"

    def test_nested_event_data_is_covered(self):
        """Not just the top level: the events list is hashed by content."""
        original = _envelope()
        checksum = compute_execution_checksum(original)

        tampered = _envelope()
        tampered["events"][0]["data"]["claim_raw"] = "a different claim"

        assert verify_execution_checksum(tampered, checksum) is False


class TestApiVerifiesByRecomputation:
    """Step 4: the API decides, the UI renders. Driven through the real route."""

    def _route(self, execution):
        import asyncio
        from unittest.mock import MagicMock, patch

        from finvet.api.routes import audit as audit_route

        logger = MagicMock()
        logger.get_execution.return_value = execution
        # The event rows the envelope describes. Returning [] here would be a
        # genuine divergence -- the envelope claiming events no row records --
        # which integrity is now right to report.
        logger.get_events.return_value = list(
            (execution.get("full_trace") or {}).get("events") or [])
        with patch.object(audit_route, "get_audit_logger", lambda: logger):
            return asyncio.run(audit_route.get_audit_trail("req_abc123"))

    def _committed(self):
        """A row as `get_execution` returns it.

        The denormalized columns belong here: integrity now compares them
        against the envelope, so a fixture carrying only `full_trace` would
        describe a row that production never produces.
        """
        envelope = _envelope()
        return {"request_id": "req_abc123",
                "claim_text": envelope["claim"],
                "verdict": envelope["verdict"],
                "confidence": envelope["confidence"],
                "agents_run": envelope["agents_run"],
                "data_sources": envelope["data_sources"],
                "full_trace": envelope,
                "execution_hash": compute_execution_checksum(envelope)}

    def test_intact_record_verifies(self):
        out = self._route(self._committed())
        assert out["integrity"]["status"] == "verified"
        assert out["integrity"]["algorithm"] == "sha256"
        assert out["integrity"]["scope"] == "audit_execution.full_trace"
        assert out["integrity"]["mismatches"] == []

    def test_altered_verdict_fails_verification(self):
        """The defect this replaces: the verdict lived beside the hash, not in
        it, so it could be edited freely."""
        record = self._committed()
        record["full_trace"]["verdict"] = "REFUTES"

        assert self._route(record)["integrity"]["status"] == "failed"

    def test_record_without_a_checksum_is_not_verified(self):
        record = self._committed()
        record["execution_hash"] = ""

        assert self._route(record)["integrity"]["status"] == "unavailable"


class TestPostHocEventsAreNotBuffered:
    """Step 6: an event recorded after the run was finalized must not join a
    buffer that nothing will commit again.

    /memory-accept reuses the original request_id, so buffering its event would
    accumulate entries under a request whose execution is already written.
    """

    def _logger(self):
        from unittest.mock import MagicMock

        from finvet.audit.logger import AuditLogger

        logger = AuditLogger.__new__(AuditLogger)
        logger.db = MagicMock()
        logger.db.log_event.return_value = True
        logger._events = {}
        import threading
        logger._lock = threading.Lock()
        logger._last_write_ok = True
        return logger

    def test_default_still_buffers(self):
        logger = self._logger()
        logger.log_event(event_type="input_received", request_id="req_1", data={})
        assert logger.buffer_size("req_1") == 1

    def test_post_hoc_event_is_not_buffered(self):
        logger = self._logger()
        logger.log_event(event_type="memory_cache_accepted", request_id="req_1",
                         data={}, buffer_for_execution=False)
        assert logger.buffer_size("req_1") == 0

    def test_post_hoc_event_still_reaches_the_database(self):
        logger = self._logger()
        logger.log_event(event_type="memory_cache_accepted", request_id="req_1",
                         data={}, buffer_for_execution=False)
        assert logger.db.log_event.call_count == 1

    def test_persisted_variant_reports_a_failed_write(self):
        logger = self._logger()
        logger.db.log_event.return_value = False
        assert logger.log_event_persisted(
            event_type="memory_cache_accepted", request_id="req_1",
            data={}, buffer_for_execution=False) is False


class TestWhatIsWrittenIsWhatVerifies:
    """The round trip, driven through the real commit path.

    Everything above tests the envelope helpers in isolation. This one runs
    AuditDatabase.commit_execution, takes the object it actually hands to the
    session, pushes it through a JSON round trip (what JSONB does on the way
    back out), and asks the real /audit route to verify it.

    A divergence between what gets hashed and what gets stored would pass every
    isolated test and fail here -- which is the failure mode that put the
    original defect in production.
    """

    def _commit_and_capture(self):
        from unittest.mock import MagicMock, patch

        from finvet.audit.database import AuditDatabase

        added = []

        class _Session:
            def add(self, obj):
                added.append(obj)

            def query(self, *args, **kwargs):
                q = MagicMock()
                q.filter.return_value.all.return_value = []
                return q

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        db = AuditDatabase.__new__(AuditDatabase)
        with patch("finvet.audit.database.get_db_session", lambda: _Session()):
            committed = db.commit_execution(
                request_id="req_live",
                claim_text="Apple FY2024 revenue was $391 billion",
                verdict="SUPPORTS", confidence=0.93, agents_run=["SEC"],
                events=[{"event_id": "evt_1", "request_id": "req_live",
                         "event_type": "input_received",
                         "timestamp": "2026-08-25T00:00:00", "agent": None,
                         "data": {"claim_raw": "Apple FY2024 revenue..."}}],
                execution_time_ms=1234,
                final_response={"verdict": "SUPPORTS", "confidence": 0.93},
                data_sources={"xbrl": {"used": True}},
                terminal_status="success",
            )
        assert committed is True
        return added

    def _verify_via_route(self, execution):
        import asyncio
        from unittest.mock import MagicMock, patch

        from finvet.api.routes import audit as audit_route

        logger = MagicMock()
        logger.get_execution.return_value = execution
        logger.get_events.return_value = list(
            (execution.get("full_trace") or {}).get("events") or [])
        with patch.object(audit_route, "get_audit_logger", lambda: logger):
            return asyncio.run(
                audit_route.get_audit_trail("req_live"))["integrity"]["status"]

    def _readback(self, row):
        """What Postgres returns for a row: a JSONB round trip plus the
        denormalized columns integrity compares against the envelope."""
        import json
        return {"request_id": row.request_id,
                "claim_text": row.claim_text,
                "verdict": row.verdict,
                "confidence": row.confidence,
                "agents_run": row.agents_run,
                "data_sources": row.data_sources,
                "llm_config": row.llm_config,
                "full_trace": json.loads(json.dumps(row.full_trace)),
                "execution_hash": row.execution_hash}

    def test_committed_record_verifies_after_a_jsonb_round_trip(self):
        added = self._commit_and_capture()
        row = next(o for o in added if hasattr(o, "full_trace"))

        assert self._verify_via_route(self._readback(row)) == "verified"

    def test_a_field_altered_in_storage_fails(self):
        added = self._commit_and_capture()
        row = next(o for o in added if hasattr(o, "full_trace"))

        record = self._readback(row)
        record["full_trace"]["verdict"] = "REFUTES"

        assert self._verify_via_route(record) == "failed"

    def test_missing_events_are_inserted_in_the_same_transaction(self):
        """Step 3: the envelope and the queryable timeline converge or fail
        together, rather than the envelope describing events no row records."""
        added = self._commit_and_capture()
        events = [o for o in added if hasattr(o, "event_type")]

        assert len(events) == 1
        assert events[0].event_id == "evt_1"
