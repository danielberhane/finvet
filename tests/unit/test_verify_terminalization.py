"""Terminal-outcome contract for /verify and /verify-stream.

Spec invariant 1: every synchronous or streaming request finishes exactly once
as success, pending_review, rejected, guardrail_blocked or error; every
terminal path attempts one audit execution commit and releases its
request-scoped in-memory state; a client must not receive a successful audited
response when persistence failed.

An earlier version of this file asserted the opposite for two cases -- it
required a missing final response *not* to commit, and it checked buffer
cleanup on guardrail and error paths without requiring the execution record
that makes those outcomes auditable at all. Those assertions encoded the
defect as the contract.

Ordering is asserted, not just occurrence: a client that has seen a terminal
response believes the run is on record, so the commit has to precede it.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from finvet.api.models import VerifyClaimRequest
from finvet.api.routes import verify as verify_route
from finvet.utils.exceptions import GuardrailViolation

CLAIM = "TEST revenue was $150 billion"


# ---------------------------------------------------------------------------
# Recording harness: one ordered log of commits and client emissions
# ---------------------------------------------------------------------------

class _Recorder:
    """Audit stub reproducing AuditLogger's buffer semantics, recording the
    order of commits relative to what the client is told."""

    def __init__(self, commit_ok=True):
        self.events = {}
        self.log = []
        self.commits = []
        self.chunks = []
        self.result = None
        self.raised = None
        self._commit_ok = commit_ok

    def log_event(self, event_type=None, request_id=None, data=None, **kw):
        self.events.setdefault(request_id, []).append(event_type)
        return "evt_stub"

    def commit_execution(self, **kwargs):
        self.commits.append(kwargs)
        self.log.append(("commit", kwargs.get("terminal_status")))
        if self._commit_ok:
            self.events.pop(kwargs["request_id"], None)
        return self._commit_ok

    def discard_buffer(self, request_id):
        self.events.pop(request_id, None)

    def buffer_size(self, request_id):
        return len(self.events.get(request_id, []))

    def emit(self, what):
        self.log.append(("emit", what))

    @property
    def terminal_status(self):
        return self.commits[0].get("terminal_status") if self.commits else None

    def commit_preceded_emission(self):
        kinds = [k for k, _ in self.log]
        if "commit" not in kinds or "emit" not in kinds:
            return False
        return kinds.index("commit") < kinds.index("emit")


class _Graph:
    def __init__(self, updates=None, invoke_result=None, raises=None):
        self._updates = updates or []
        self._invoke_result = invoke_result or {}
        self._raises = raises

    def stream(self, state, config, stream_mode=None):
        if self._raises:
            raise self._raises
        for update in self._updates:
            yield update

    def invoke(self, state, config):
        if self._raises:
            raise self._raises
        return self._invoke_result


def _final(status="success", verdict="SUPPORTS"):
    return {
        "status": status,
        "verdict": verdict,
        "confidence": 0.93,
        "claim": CLAIM,
        "metadata": {"data_sources": {"xbrl": {"used": True}}},
    }


SUCCESS_STATE = {"agent_type": "sec", "final_response": _final()}
REJECTED_STATE = {"agent_type": None,
                  "final_response": _final(status="rejected", verdict="REJECTED")}
PENDING_STATE = {"hitl_required": True, "hitl_gate_passed": False,
                 "hitl_triggers": ["low_confidence"], "agent_type": "sec",
                 "agent_evidence": {"agent": "sec", "reasoning": "unsure",
                                    "tools_called": []}}


def _run(monkeypatch, *, streaming, updates=None, invoke_result=None,
         raises=None, commit_ok=True):
    """Drive a real route to a terminal outcome and return the recorder."""
    rec = _Recorder(commit_ok=commit_ok)
    monkeypatch.setattr(verify_route, "get_audit_logger", lambda: rec)
    monkeypatch.setattr(verify_route.deps, "claim_memory", None)
    monkeypatch.setattr(verify_route.deps, "verification_graph",
                        _Graph(updates, invoke_result, raises))

    request = VerifyClaimRequest(claim=CLAIM)

    if streaming:
        response = verify_route.verify_claim_stream(request)

        async def drain():
            async for chunk in response.body_iterator:
                for line in chunk.splitlines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    rec.chunks.append(event)
                    if event.get("type") in ("complete", "guardrail", "error"):
                        rec.emit(event["type"])

        asyncio.run(drain())
    else:
        try:
            rec.result = verify_route.verify_claim(request)
            rec.emit("response")
        except HTTPException as exc:
            rec.raised = exc
            rec.emit("http_error")
        except GuardrailViolation as exc:
            rec.raised = exc
            rec.emit("guardrail")
    return rec


MATRIX = [
    ("success", dict(updates=[{"response_generator": SUCCESS_STATE}],
                     invoke_result=SUCCESS_STATE), "success"),
    ("pending_review", dict(updates=[{"output_guardrails": PENDING_STATE}],
                            invoke_result=PENDING_STATE), "pending_review"),
    ("parser_rejection", dict(updates=[{"response_generator": REJECTED_STATE}],
                              invoke_result=REJECTED_STATE), "rejected"),
    ("guardrail", dict(raises=GuardrailViolation("blocked", "prompt_injection")),
     "guardrail_blocked"),
    ("unexpected_error", dict(raises=RuntimeError("node exploded")), "error"),
    ("missing_final_response", dict(updates=[{"claim_parser": {"parsed_claim": None}}],
                                    invoke_result={}), "error"),
]


@pytest.mark.parametrize("streaming", [True, False], ids=["sse", "sync"])
@pytest.mark.parametrize("name,kwargs,expected_status", MATRIX,
                         ids=[m[0] for m in MATRIX])
class TestEveryTerminalOutcomeIsRecorded:
    """One request, one terminal outcome, one commit -- on both routes."""

    def test_commits_exactly_once(self, monkeypatch, streaming, name, kwargs,
                                  expected_status):
        rec = _run(monkeypatch, streaming=streaming, **kwargs)
        assert len(rec.commits) == 1, (
            f"{name}/{'sse' if streaming else 'sync'} produced "
            f"{len(rec.commits)} commits")

    def test_terminal_status_is_correct(self, monkeypatch, streaming, name,
                                        kwargs, expected_status):
        rec = _run(monkeypatch, streaming=streaming, **kwargs)
        assert rec.terminal_status == expected_status

    def test_commit_precedes_the_terminal_response(self, monkeypatch, streaming,
                                                   name, kwargs, expected_status):
        rec = _run(monkeypatch, streaming=streaming, **kwargs)
        assert rec.commit_preceded_emission(), f"order was {rec.log}"

    def test_buffer_is_released(self, monkeypatch, streaming, name, kwargs,
                                expected_status):
        rec = _run(monkeypatch, streaming=streaming, **kwargs)
        assert rec.events == {}, f"buffered events survived: {rec.events}"


class TestAuditPersistenceFailure:
    """A client must not be told a run succeeded when it was never recorded."""

    def test_sync_returns_an_error_not_a_verdict(self, monkeypatch):
        rec = _run(monkeypatch, streaming=False, invoke_result=SUCCESS_STATE,
                   commit_ok=False)
        assert rec.raised is not None, "sync returned a verdict despite a failed commit"
        assert rec.raised.status_code == 503

    def test_sse_emits_error_not_complete(self, monkeypatch):
        rec = _run(monkeypatch, streaming=True,
                   updates=[{"response_generator": SUCCESS_STATE}], commit_ok=False)
        kinds = {c.get("type") for c in rec.chunks}
        assert "error" in kinds
        assert "complete" not in kinds

    def test_failed_commit_is_not_retried_recursively(self, monkeypatch):
        """Auditing an audit failure would recurse."""
        rec = _run(monkeypatch, streaming=True,
                   updates=[{"response_generator": SUCCESS_STATE}], commit_ok=False)
        assert len(rec.commits) == 1

    def test_buffer_is_released_even_when_persistence_fails(self, monkeypatch):
        rec = _run(monkeypatch, streaming=False, invoke_result=SUCCESS_STATE,
                   commit_ok=False)
        assert rec.events == {}


class TestBothRoutesOpenIdentically:
    """Retained: the two routes must record the start of a request the same way."""

    PRIOR_ID = "req_0123456789ab"
    PRIOR = {"request_id": PRIOR_ID, "claim": "a prior claim",
             "verdict": "SUPPORTS", "confidence": 0.9, "summary": "prior run"}

    def _events(self, monkeypatch, streaming):
        rec = _Recorder()
        logged = []
        original = rec.log_event

        def spy(event_type=None, request_id=None, data=None, **kw):
            logged.append((event_type, data))
            return original(event_type=event_type, request_id=request_id,
                            data=data, **kw)

        rec.log_event = spy
        memory = MagicMock()
        memory.get_claim.return_value = self.PRIOR
        monkeypatch.setattr(verify_route, "get_audit_logger", lambda: rec)
        monkeypatch.setattr(verify_route.deps, "claim_memory", memory)
        monkeypatch.setattr(verify_route.deps, "verification_graph",
                            _Graph([{"response_generator": SUCCESS_STATE}],
                                   SUCCESS_STATE))
        request = VerifyClaimRequest(claim=CLAIM,
                                     memory_context_request_id=self.PRIOR_ID)
        if streaming:
            response = verify_route.verify_claim_stream(request)

            async def drain():
                async for _ in response.body_iterator:
                    pass

            asyncio.run(drain())
        else:
            verify_route.verify_claim(request)
        return logged

    @pytest.mark.parametrize("streaming", [True, False], ids=["sse", "sync"])
    def test_memory_decision_is_recorded(self, monkeypatch, streaming):
        types = [t for t, _ in self._events(monkeypatch, streaming)]
        assert "memory_context_injected" in types

    @pytest.mark.parametrize("streaming", [True, False], ids=["sse", "sync"])
    def test_input_received_carries_a_timestamp(self, monkeypatch, streaming):
        logged = self._events(monkeypatch, streaming)
        first_type, first_data = logged[0]
        assert first_type == "input_received"
        assert "timestamp" in first_data
        datetime.fromisoformat(first_data["timestamp"])
