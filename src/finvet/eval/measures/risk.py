"""Layer 5 — asymmetric risk: which kind of error the system made.

Affirming a false claim and declining a true one are both "not the expected
verdict", and only one puts a wrong number in front of a user. Averaging them
into a single accuracy figure hides the failure that matters, so each is
counted separately:

    dangerous   a decisive verdict (SUPPORTS or REFUTES) that does not match the
                expected one: the opposite answer, or any answer where the
                expected outcome was NOT_ENOUGH_INFO, REJECTED or BLOCKED.
                Each asserts something the source does not establish.
    declined    any other miss: a decline, an escalation, a wrong rejection,
                or no verdict at all (a timeout or error). None of these
                asserts anything to the user.
    correct     matched the expectation

dangerous is reported on its own and never folded into an aggregate.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .artifacts import Run, expected, is_scored, verdict

DECISIVE = frozenset({"SUPPORTS", "REFUTES"})


@dataclass(frozen=True)
class Risk:
    scored: int
    correct: int
    declined: int
    dangerous: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def dangerous_rate(self) -> float:
        return len(self.dangerous) / self.scored if self.scored else 0.0

    @property
    def decline_rate(self) -> float:
        return self.declined / self.scored if self.scored else 0.0


def measure(runs: List[Run]) -> Risk:
    """Count outcomes by their cost, not by whether they matched."""
    scored = correct = declined = 0
    dangerous: List[Tuple[str, int]] = []

    for run in runs:
        label = run.label or run.started_utc
        for row_id, row in run.rows.items():
            if not is_scored(row):
                continue
            scored += 1
            got, want = verdict(row), expected(row)

            if got == want:
                correct += 1
            elif got in DECISIVE:
                # A definite answer the source does not support: the opposite
                # verdict, or a verdict where the expected outcome was a decline,
                # a rejection or a block. The second used to fall into
                # "declined", so the measure could not see the failure the
                # fail-closed design exists to prevent.
                dangerous.append((label, row_id))
            else:
                # Nothing asserted. A missing verdict (timeout, error) used to
                # match the absent "opposite" of NOT_ENOUGH_INFO and be counted
                # dangerous.
                declined += 1

    return Risk(scored=scored, correct=correct, declined=declined,
                dangerous=sorted(dangerous))
