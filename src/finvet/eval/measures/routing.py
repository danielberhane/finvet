"""Routing — did the pipeline reach its answer by the intended path.

A verdict can be right for the wrong reason. An `sec/xbrl` claim answered out of
filing text instead of an XBRL fact still reads SUPPORTS; an `a2a` claim the
News agent never delegated still reads SUPPORTS. Nothing in the outcome layer
can see the difference, because the outcome is identical.

The dataset already says which path each claim should take -- `expected.sources`
names `xbrl`, `rag`, `a2a`, or nothing -- and the runner already records which
were used. The two were never compared. Backtested against three recorded runs,
this finds two reproducible cases: id 51 answered from `rag` when the delegation
should have fired, and id 55 recorded no source at all. Both returned the
expected verdict in every run.

Reported, not asserted. It is a measurement of routing, and a row failing it has
not necessarily answered incorrectly -- what was concluded and how it was
reached are separate questions, measured separately.

**Observe rows are skipped.** A row with no expected verdict carries
`sources: []` meaning "not labelled", not "no source may be used": the
known-defect rows consult XBRL and RAG by design. Treating that empty list as an
assertion produced two false failures, so the guard matches the one
`test_golden.py` already applies to verdicts -- no expected verdict, no claim
about the row.

`data_sources` tracks `xbrl`, `rag` and `a2a` only. Market quotes are not a
tracked source, so a `market/quote` row legitimately records `[]` and its
expectation is `[]` too. Verified: no market row is scored as a mismatch.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .artifacts import Run


@dataclass(frozen=True)
class Routing:
    scored: int
    matched: int
    # row id -> (expected sources, sources actually used)
    mismatches: Dict[int, Tuple[List[str], List[str]]] = field(
        default_factory=dict)
    skipped_unlabelled: List[int] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.matched / self.scored if self.scored else 1.0


def measure(runs: List[Run]) -> Routing:
    """How often the sources used were the sources the dataset expected."""
    scored = matched = 0
    mismatches: Dict[int, Tuple[List[str], List[str]]] = {}
    skipped: List[int] = []

    for run in runs:
        for row_id, row in run.rows.items():
            if row.get("error"):
                continue

            expected = (row.get("expected") or {})
            # No expected verdict means the row is observed, not asserted, and
            # its `sources` is unlabelled rather than empty.
            if expected.get("verdict") is None:
                skipped.append(row_id)
                continue

            want = expected.get("sources")
            got = (row.get("actual") or {}).get("data_sources")
            if want is None or got is None:
                continue

            scored += 1
            if sorted(want) == sorted(got):
                matched += 1
            else:
                # Last run wins on a repeat; a stable mismatch reports the same
                # pair either way, and an unstable one is a reliability finding
                # rather than a routing one.
                mismatches[row_id] = (sorted(want), sorted(got))

    return Routing(scored=scored, matched=matched, mismatches=mismatches,
                   skipped_unlabelled=sorted(set(skipped)))
