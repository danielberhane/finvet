"""A claim refused at the door is a block, whichever door refused it.

A claim deliberately too short or too long is rejected with no model call,
which is the expected behaviour, and was recorded as an empty verdict. The
runner mapped HTTP 400 from the input guard to BLOCKED; the
API's own length validation answers first, with HTTP 422, and that came back
as an empty verdict. The mapping is narrow on purpose: only a 422 about the
claim's length is a refusal of the claim.
"""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_golden.py"
spec = importlib.util.spec_from_file_location("run_golden_refusals", SCRIPT)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

ROW = {"id": 1200, "claim": "Apple up", "category": "guard", "strength": "strict",
       "expected": {"verdict": "BLOCKED", "limitation": None, "sources": []}}


class _Response:
    headers = {"content-type": "application/json"}

    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def _run(monkeypatch, status, body):
    monkeypatch.setattr(rg.httpx, "post", lambda *a, **k: _Response(status, body))
    return rg.run_one(ROW)


def _validation(kind, field):
    return {"detail": [{"type": kind, "loc": ["body", field], "msg": "m",
                        "input": "x", "ctx": {}}]}


def test_a_claim_too_short_is_recorded_as_blocked(monkeypatch):
    record = _run(monkeypatch, 422, _validation("string_too_short", "claim"))
    assert record["actual"]["verdict"] == "BLOCKED"
    assert record["http"] == 422


def test_a_claim_too_long_is_recorded_as_blocked(monkeypatch):
    record = _run(monkeypatch, 422, _validation("string_too_long", "claim"))
    assert record["actual"]["verdict"] == "BLOCKED"


def test_a_rejection_of_another_field_is_not_a_block(monkeypatch):
    record = _run(monkeypatch, 422,
                  _validation("string_too_long", "memory_context_request_id"))
    assert record["actual"]["verdict"] is None


def test_the_input_guards_refusal_is_still_a_block(monkeypatch):
    record = _run(monkeypatch, 400, {"detail": "blocked"})
    assert record["actual"]["verdict"] == "BLOCKED"
