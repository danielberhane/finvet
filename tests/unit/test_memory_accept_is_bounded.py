"""`/memory-accept` writes to the audit trail; it must be as bounded as `/verify`.

The endpoint persists an `audit_events` row from three request fields. None was
validated: the identifier had no format, the claim text no length, the
similarity no range -- while `/verify` holds the same kind of identifier to
`^req_[0-9a-f]{12}$` precisely because it crosses into the record. And unlike
its sibling `/memory-check`, it had no feature gate, so with claim memory
disabled (the default) a caller could still write rows for a feature that was
switched off. `audit_events.request_id` carries no foreign key, so nothing
downstream would have rejected them.
"""

import pytest
from pydantic import ValidationError

from finvet.api.models import MemoryAcceptRequest


def _valid(**over):
    body = {"original_request_id": "req_4efc3fdede22",
            "claim": "Apple's total revenue was $391 billion in fiscal year 2024",
            "similarity": 0.93}
    body.update(over)
    return body


class TestTheRequestIsBounded:

    def test_a_well_formed_request_is_accepted(self):
        assert MemoryAcceptRequest(**_valid()).similarity == 0.93

    @pytest.mark.parametrize("bad_id", [
        "not-a-request-id",
        "req_XYZ",
        "req_4efc3fdede22 OR 1=1",
        "",
        "req_4efc3fdede2",       # eleven hex digits
        "req_4efc3fdede222",     # thirteen
    ])
    def test_a_free_form_identifier_is_rejected(self, bad_id):
        """It becomes `audit_events.request_id`, which has no foreign key."""
        with pytest.raises(ValidationError):
            MemoryAcceptRequest(**_valid(original_request_id=bad_id))

    def test_an_unbounded_claim_is_rejected(self):
        """The text lands in the audit row's JSONB payload."""
        with pytest.raises(ValidationError):
            MemoryAcceptRequest(**_valid(claim="x" * 2001))

    def test_a_too_short_claim_is_rejected(self):
        with pytest.raises(ValidationError):
            MemoryAcceptRequest(**_valid(claim="short"))

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 42.0])
    def test_similarity_outside_zero_to_one_is_rejected(self, bad):
        with pytest.raises(ValidationError):
            MemoryAcceptRequest(**_valid(similarity=bad))

    def test_unknown_fields_are_rejected(self):
        """Same rule as the verify request: extra keys are refused, not ignored."""
        with pytest.raises(ValidationError):
            MemoryAcceptRequest(**_valid(user_decision="accept"))


class TestTheEndpointIsGatedOnTheFeature:

    def test_it_refuses_when_claim_memory_is_disabled(self, monkeypatch):
        """The shipped default. No cached result exists, so there is no
        acceptance to record."""
        from fastapi import HTTPException
        from finvet.api import deps
        from finvet.api.routes import memory

        monkeypatch.setattr(deps, "claim_memory", None, raising=False)

        with pytest.raises(HTTPException) as exc:
            memory.memory_accept(MemoryAcceptRequest(**_valid()))

        assert exc.value.status_code == 404

    def test_it_records_when_claim_memory_is_enabled(self, monkeypatch):
        """The gate must not swallow the real path."""
        from finvet.api import deps
        from finvet.api.routes import memory

        monkeypatch.setattr(deps, "claim_memory", object(), raising=False)
        written = {}

        class _Audit:
            def log_event_persisted(self, **kwargs):
                written.update(kwargs)
                return True

        monkeypatch.setattr(memory, "get_audit_logger", lambda: _Audit())

        result = memory.memory_accept(MemoryAcceptRequest(**_valid()))

        assert result["status"] == "logged"
        assert written["request_id"] == "req_4efc3fdede22"
        assert written["event_type"] == "memory_cache_accepted"
