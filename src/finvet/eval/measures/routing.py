"""Routing — whether the pipeline reached its answer by the intended path.

A verdict can be right for the wrong reason. An sec/xbrl claim answered from
filing text still reads SUPPORTS, and an a2a claim the News agent never
delegated still reads SUPPORTS. The outcome layer cannot tell the difference,
because the outcome is identical.

Compares expected.sources from the dataset, which names xbrl, rag, a2a or
nothing, against the sources the runner recorded.

Reported, not asserted. A row that fails this has not necessarily answered
incorrectly; what was concluded and how it was reached are separate questions.

Rows with no expected verdict are skipped: their empty sources list means "not
labelled", not "no source may be used". data_sources tracks xbrl, rag and a2a
only, so a market row records an empty list and expects one.
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
