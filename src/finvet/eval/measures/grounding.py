"""Layer 3 — grounding: is every decisive number traceable to a source.

The guide calls this the highest-stakes layer for a verification product: "a
correct-by-luck number that appears in no tool output is a hallucination and
must be counted as one."

FinVet enforces this structurally rather than hoping for it. `_apply_override`
compares only a `TrustedObservation`, and a numeric claim with no observation
fails closed to NOT_ENOUGH_INFO. So this measure is expected to report 100% with
zero violations -- **it scores the enforcement, not the model.**

That is worth stating as a number anyway. "The design prevents it" is a claim; a
count over 400 recorded verdicts is evidence, and a violation would mean the
guard had a hole.

Two things are counted:

`traceable`   a decisive verdict whose `retrieved_value` is backed by an
              observation naming the tool that produced it.
`untraceable` a decisive verdict carrying a number with no observation behind
              it. This is the hallucination case and must be zero.

Escalations and declines are excluded: they assert no number, so there is
nothing to trace.
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
