"""Loads recorded golden runs and normalises them for the measure modules.

The only shared module in this package. Every measure is a pure function of what
this returns, so the layers stay independent of each other and of how a run was
produced.

Three normalisations happen here, because every layer depends on them.

A guardrail block counts as a verdict. A blocked claim returns HTTP 400 with no
verdict field, and is mapped to a recorded block. Artifacts written before the
runner did this carry `http` instead, and are mapped the same way.

Correctness is judged against each row's own expectation. A row whose
expected.verdict is null has no notion of correct and is excluded.

stable_ids() lists rows whose verdict changed for a reason outside the system,
so reliability does not read a code change as non-determinism.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    dataset: str = ""

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
        dataset=Path(data.get("dataset") or "").name,
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


def load_dataset(path: Path) -> Tuple[Dict[int, Dict[str, Any]], str]:
    """A golden dataset file as {id: row}, with the file's sha256."""
    raw = Path(path).read_bytes()
    rows = {r["id"]: r for r in (json.loads(line) for line in raw.decode().splitlines()
                                 if line.strip())}
    return rows, hashlib.sha256(raw).hexdigest()


def rebase(run: Run, dataset: Dict[int, Dict[str, Any]],
           burned: Iterable[int] = ()) -> Tuple[Run, List[int]]:
    """Grade a recorded run against a later version of its dataset.

    A run records the strength and expectation of the dataset it ran against.
    When the dataset changes, a reused answer must be graded by the new
    version, not the one stored beside it. Rows the dataset no longer has, and
    burned rows, are dropped.

    A row the dataset now fills from a live quote (`recipe`) but whose answer
    carries no quote snapshot was answered as a different claim. It is dropped
    and returned as stale: it needs a new run, not a new grade.
    """
    burned = set(burned)
    rows: Dict[int, Dict[str, Any]] = {}
    stale: List[int] = []
    for row_id, row in run.rows.items():
        spec = dataset.get(row_id)
        if spec is None or row_id in burned:
            continue
        if spec.get("recipe") and not row.get("snapshot"):
            stale.append(row_id)
            continue
        rows[row_id] = dict(row, strength=spec["strength"], expected=spec["expected"])
    return replace(run, rows=rows), sorted(stale)


def merge(base: Run, patch: Run, dataset: Dict[int, Dict[str, Any]], dataset_name: str,
          burned: Iterable[int] = ()) -> Run:
    """Old answers plus new answers for the rows a dataset change made stale.

    Refuses rather than guesses: answers from two models are not one run, a
    partial patch would leave rows silently unanswered, and a stale row with no
    new answer would be scored on a claim the dataset no longer asks.
    """
    if base.model != patch.model:
        raise ValueError(f"{patch.label} was answered by {patch.model}, "
                         f"{base.label} by {base.model}; they are not one run")
    if not patch.complete:
        raise ValueError(f"{patch.label} is incomplete")
    if patch.dataset != dataset_name:
        raise ValueError(f"{patch.label} ran against {patch.dataset or 'an unrecorded dataset'}, "
                         f"not {dataset_name}")
    unknown = sorted(set(patch.rows) - set(dataset))
    if unknown:
        raise ValueError(f"{patch.label} answers rows the dataset does not have: {unknown}")
    old, stale = rebase(base, dataset, burned)
    new, new_stale = rebase(patch, dataset, burned)
    if new_stale:
        raise ValueError(f"{patch.label} has no quote snapshot for {new_stale}")
    missing = sorted(set(stale) - set(new.rows))
    if missing:
        raise ValueError(f"rows {missing} need a new answer and {patch.label} has none")
    return replace(old, label=f"{base.label}+{patch.label}", rows={**old.rows, **new.rows})


def regrade(runs: List[Run], pool: List[Run], dataset_path: Path,
            patches: Iterable[str] = (),
            burned: Iterable[int] = ()) -> Tuple[List[Run], Dict[str, Any]]:
    """Apply a dataset version and `OLD=NEW` patch pairs to selected runs.

    `pool` is every run the patches may be found in. Each label must name
    exactly one run. A base run with no patch keeps its answers and loses its
    stale rows, which are reported, so a preview can be scored before the new
    answers exist.
    """
    dataset, sha = load_dataset(dataset_path)
    name = Path(dataset_path).name

    def one(label: str) -> Run:
        found = [r for r in pool if r.label == label]
        if len(found) != 1:
            raise ValueError(f"label {label!r} matches {len(found)} complete runs, not 1")
        return found[0]

    pairs: Dict[str, Run] = {}
    for spec in patches:
        old, sep, new = spec.partition("=")
        if not sep or not old or not new:
            raise ValueError(f"--patch takes OLD=NEW, got {spec!r}")
        if old not in {r.label for r in runs}:
            raise ValueError(f"--patch names {old!r}, which is not among the selected runs")
        pairs[old] = one(new)

    out: List[Run] = []
    dropped: Dict[str, List[int]] = {}
    for run in runs:
        if run.label in pairs:
            out.append(merge(run, pairs[run.label], dataset, name, burned))
        else:
            rebased, stale = rebase(run, dataset, burned)
            out.append(rebased)
            if stale:
                dropped[run.label] = stale
    info = {"dataset": name, "dataset_sha256": sha,
            "patches": {k: v.label for k, v in pairs.items()},
            "stale_dropped": dropped}
    return out, info


def outcome_excluding_live_market(run: Run) -> Tuple[int, int]:
    """Correct and scored rows, minus the live-market category.

    The headline outcome figure: market quotes move between runs, so a quote
    outage would otherwise read as a model difference.
    """
    kept = [r for r in run.rows.values()
            if is_scored(r) and not (r.get("category") or "").startswith("market/")]
    return sum(is_correct(r) for r in kept), len(kept)
