"""Judge a recorded golden run. Reads an artifact; never calls the API.

`scripts/run_golden.py` spends the money and records; this spends nothing and
decides. The split matters: `test_claim_matrix.py` asserts by calling the live
API, so every pytest run re-spends and a result can never be re-examined. One
paid run should be re-analysable forever.

**The strength contract**, which is what makes a golden set honest about its own
uncertainty:

    strict   the answer is determined by a filed fact or a structural rule.
             AAPL FY2024 revenue *is* 391,035,000,000; a question is *not* a
             claim. A mismatch is a defect.
    safe     the category is determined but the exact verdict depends on live
             data or model judgement. Assert only that the system did not reach
             the *opposite* conclusion -- SUPPORTS and REFUTES are opposites;
             NOT_ENOUGH_INFO is a decline and never contradicts either.
    observe  recorded, never asserted. Known defects live here so that fixing
             one does not fail a test that was documenting it.

**Escalation is a third outcome, not a failure of the answer.** The dataset's
status vocabulary (success/rejected/blocked) has no word for "sent to a human",
so a run that escalates is reported as ESCALATED rather than scored as a wrong
verdict. On a `strict` row it still fails -- the dataset asserts the system can
answer it -- but the message says which of the two happened.
"""

import json
import os
from pathlib import Path

import pytest

OPPOSITE = {"SUPPORTS": "REFUTES", "REFUTES": "SUPPORTS"}


def _artifact() -> dict | None:
    """The run to judge: FINVET_GOLDEN_RUN, else the newest in the dataset dir."""
    named = os.environ.get("FINVET_GOLDEN_RUN")
    if named:
        path = Path(named).expanduser()
        return json.loads(path.read_text()) if path.exists() else None

    root = os.environ.get("FINVET_GOLDEN_DIR")
    if not root:
        return None
    runs = sorted(Path(root).expanduser().glob("run-*.json"))
    return json.loads(runs[-1].read_text()) if runs else None


ARTIFACT = _artifact()
pytestmark = pytest.mark.skipif(
    ARTIFACT is None,
    reason="no golden run artifact; set FINVET_GOLDEN_DIR and run "
           "scripts/run_golden.py")


def _rows(strength=None, frozen=None):
    rows = (ARTIFACT or {}).get("results", [])
    if strength:
        rows = [r for r in rows if r.get("strength") == strength]
    if frozen is not None:
        rows = [r for r in rows if r.get("frozen") is frozen]
    return rows


def _verdict(row):
    """The verdict a row reached, normalised.

    An input guardrail refuses with HTTP 400 and no verdict field. Artifacts
    written before the runner mapped that recorded None, so the block -- correct
    behaviour -- read as an empty answer. Derived here as well as in the runner
    so an already-paid-for run stays readable.
    """
    actual = row.get("actual") or {}
    verdict = actual.get("verdict")
    if verdict is None and row.get("http") == 400:
        return "BLOCKED"
    return verdict


def _describe(row) -> str:
    actual = row.get("actual") or {}
    return (f"id {row['id']} [{row['category']}] {row['claim'][:60]!r}\n"
            f"    expected {row['expected'].get('verdict')} · "
            f"got {_verdict(row)} "
            f"({'escalated' if actual.get('escalated') else 'answered'})"
            f" · limitation={actual.get('limitation')}")


class TestTheRunItself:
    """Before judging answers, establish the run is worth judging."""

    def test_the_run_records_which_model_produced_it(self):
        """Without this the artifact cannot be compared to any other run."""
        config = (ARTIFACT or {}).get("llm_config") or {}

        assert set(config) >= {"parser", "agent", "verdict"}
        assert config["agent"]["model"]

    def test_nothing_crashed(self):
        errored = [r for r in _rows() if r.get("error")]

        assert not errored, "\n".join(
            f"id {r['id']}: {r['error']}" for r in errored)


class TestStrictRowsMustMatch:
    """A filed fact or a structural rule determines these."""

    def test_every_strict_row_reaches_its_expected_verdict(self):
        failures = []
        for row in _rows(strength="strict"):
            expected = row["expected"].get("verdict")
            if expected is None:
                continue
            if _verdict(row) != expected:
                failures.append(_describe(row))

        assert not failures, (
            f"{len(failures)} strict rows did not match:\n\n"
            + "\n\n".join(failures))

    def test_no_strict_row_was_sent_to_a_human(self):
        """Reported apart from a wrong verdict: escalating is not answering
        incorrectly, it is declining to answer something the dataset says is
        answerable."""
        escalated = [_describe(r) for r in _rows(strength="strict")
                     if (r.get("actual") or {}).get("escalated")]

        assert not escalated, (
            f"{len(escalated)} strict rows escalated instead of answering:\n\n"
            + "\n\n".join(escalated))


class TestSafeRowsMustNotInvertTheTruth:
    """The exact verdict depends on live data, so only the opposite is a defect.

    Confirming a claim needs evidence that asserts it; refuting one on the
    absence of evidence is the fallacy this system is built to avoid. A decline
    is always permitted here."""

    def test_no_safe_row_reaches_the_opposite_conclusion(self):
        failures = []
        for row in _rows(strength="safe"):
            forbidden = OPPOSITE.get(row["expected"].get("verdict"))
            if forbidden and _verdict(row) == forbidden:
                failures.append(_describe(row))

        assert not failures, (
            f"{len(failures)} safe rows reached the opposite verdict:\n\n"
            + "\n\n".join(failures))


class TestDeclinesStateTheirReason:
    """A decline with no reason queues a reviewer who cannot act. Any row whose
    expectation names a limitation must produce that limitation."""

    def test_expected_limitations_are_reported(self):
        failures = []
        for row in _rows():
            expected = row["expected"].get("limitation")
            if not expected:
                continue
            if (row.get("actual") or {}).get("limitation") != expected:
                failures.append(_describe(row))

        assert not failures, (
            f"{len(failures)} rows lack their expected limitation:\n\n"
            + "\n\n".join(failures))


class TestTheReport:
    """Not assertions -- the summary a reader needs. Always passes; run with
    -s to read it."""

    def test_print_summary(self):
        rows = _rows()
        by_outcome = {"answered": 0, "escalated": 0, "error": 0}
        for r in rows:
            actual = r.get("actual") or {}
            key = ("error" if r.get("error")
                   else "escalated" if actual.get("escalated") else "answered")
            by_outcome[key] += 1

        print(f"\n  run       : {(ARTIFACT or {}).get('started_utc')}"
              f"  label={(ARTIFACT or {}).get('label') or '-'}")
        print(f"  model     : {((ARTIFACT or {}).get('llm_config') or {}).get('agent', {}).get('model')}")
        print(f"  rows      : {len(rows)}  "
              f"frozen={len(_rows(frozen=True))} live={len(_rows(frozen=False))}")
        print(f"  outcomes  : {by_outcome}")

        for strength in ("strict", "safe", "observe"):
            subset = _rows(strength=strength)
            if not subset:
                continue
            matched = sum(1 for r in subset
                          if _verdict(r) == r["expected"].get("verdict"))
            print(f"    {strength:<8} {matched:>3}/{len(subset)} matched exactly")

        mismatched = [r for r in rows
                      if _verdict(r) != r["expected"].get("verdict")
                      and r["expected"].get("verdict") is not None]
        if mismatched:
            print(f"\n  {len(mismatched)} rows differ from expectation:")
            for r in mismatched:
                print(f"    id {r['id']:>3} [{r['strength']:<7}] "
                      f"{str(r['expected'].get('verdict')):<16} -> "
                      f"{str(_verdict(r)):<16} {r['claim'][:46]}")
