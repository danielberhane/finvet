"""Loading and normalising recorded golden runs.

The only shared module in this package. Every measure is a pure function of what
this returns, so the layers stay independent of each other and of how a run was
produced.

Three decisions live here because getting any of them wrong would corrupt every
layer at once:

**A blocked claim is a verdict.** An input guardrail refuses with HTTP 400 and
no `verdict` field. Reading `verdict` alone recorded a correct block as an empty
answer, and six guard rows scored as failures. Artifacts written before the
runner mapped this still carry `http`, so the mapping is derived here too.

**Correctness compares against the row's own expectation.** A row whose
`expected.verdict` is null (the known-defect rows, recorded and never asserted)
has no notion of correct and is excluded rather than counted as a failure.

**A run that changed between artifacts is not a stability signal.** Row 88 was
fixed mid-series, so its verdict differs across runs for a reason that has
nothing to do with non-determinism. `stable_ids` excludes rows a code change
moved, which otherwise inflate every reliability figure.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Rows whose behaviour was deliberately changed by a fix during the run series.
# Excluded from cross-run stability, kept for single-run scoring.
CHANGED_BY_FIX = frozenset({88})


@dataclass(frozen=True)
class Run:
    """One recorded execution of the golden set."""

    label: str
    started_utc: str
    model: str
    complete: bool
    rows: Dict[int, Dict[str, Any]]

    def __len__(self) -> int:
        return len(self.rows)


def load_run(path: Path) -> Run:
    """Read one artifact written by `scripts/run_golden.py`."""
    data = json.loads(Path(path).read_text())
    return Run(
        label=data.get("label") or "",
        started_utc=data.get("started_utc") or "",
        model=((data.get("llm_config") or {}).get("agent") or {}).get("model", ""),
        complete=bool(data.get("complete", True)),
        rows={r["id"]: r for r in data.get("results", [])},
    )


def load_runs(paths: Iterable[Path]) -> List[Run]:
    return [load_run(p) for p in paths]


def find_runs(directory: Path, pattern: str = "run-*.json") -> List[Path]:
    """Artifacts in a directory, oldest first.

    Probe and resilience-check artifacts are partial by design; a caller wanting
    only whole runs should filter on `Run.complete`.
    """
    return sorted(Path(directory).expanduser().glob(pattern))


def verdict(row: Dict[str, Any]) -> Optional[str]:
    """The verdict a row reached, with a guardrail block counted as one."""
    actual = row.get("actual") or {}
    value = actual.get("verdict")
    if value is None and row.get("http") == 400:
        return "BLOCKED"
    return value


def expected(row: Dict[str, Any]) -> Optional[str]:
    return (row.get("expected") or {}).get("verdict")


def is_scored(row: Dict[str, Any]) -> bool:
    """Whether this row asserts anything.

    False for the `observe` rows, which record a known defect and must not fail
    a measure that was never meant to judge them.
    """
    return expected(row) is not None


def is_correct(row: Dict[str, Any]) -> bool:
    return is_scored(row) and verdict(row) == expected(row)


def confidence(row: Dict[str, Any]) -> Optional[float]:
    return (row.get("actual") or {}).get("confidence")


def escalated(row: Dict[str, Any]) -> bool:
    return bool((row.get("actual") or {}).get("escalated"))


def stable_ids(runs: List[Run], exclude: Iterable[int] = CHANGED_BY_FIX) -> List[int]:
    """Row ids present in every run, scored, and not moved by a code change.

    The comparable population for anything measured across runs.
    """
    if not runs:
        return []
    common = set(runs[0].rows)
    for run in runs[1:]:
        common &= set(run.rows)
    common -= set(exclude)
    return sorted(i for i in common
                  if all(is_scored(run.rows[i]) for run in runs))
