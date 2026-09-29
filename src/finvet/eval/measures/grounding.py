"""Layer 3 — grounding: whether every decisive number traces to a source.

Counts two things over recorded runs:

    traceable     a decisive verdict whose retrieved_value is backed by an
                  observation naming the tool that produced it
    untraceable   a decisive verdict carrying a number with no observation
                  behind it

Escalations and declines are excluded. They assert no number, so there is
nothing to trace.

FinVet enforces this in code: _apply_override compares only a
TrustedObservation, and a numeric claim without one fails closed. The expected
result is therefore 100% with zero untraceable. This measure scores the
enforcement, not the model, and any violation means the guard has a hole.
"""

from dataclasses import dataclass, field
from typing import List

from .artifacts import Run, escalated, verdict

DECISIVE = frozenset({"SUPPORTS", "REFUTES"})


@dataclass(frozen=True)
class Grounding:
    decisive_numeric: int
    traceable: int
    untraceable: List[int] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return (self.traceable / self.decisive_numeric
                if self.decisive_numeric else 1.0)


def measure(runs: List[Run]) -> Grounding:
    """How many decisive numeric verdicts name the source of their number."""
    decisive = traceable = 0
    untraceable: List[int] = []

    for run in runs:
        for row_id, row in run.rows.items():
            actual = row.get("actual") or {}
            if verdict(row) not in DECISIVE or escalated(row):
                continue
            if actual.get("retrieved_value") is None:
                # A decisive verdict on a claim naming no value -- a qualitative
                # SUPPORTS. Nothing numeric to trace.
                continue

            decisive += 1
            if actual.get("observation_tool"):
                traceable += 1
            else:
                untraceable.append(row_id)

    return Grounding(decisive_numeric=decisive, traceable=traceable,
                     untraceable=sorted(set(untraceable)))
