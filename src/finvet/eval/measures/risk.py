"""Layer 5 — asymmetric risk: not all errors cost the same.

For a verification system, confidently affirming a false claim is categorically
worse than declining to judge a true one. Averaging the two into one accuracy
figure hides exactly the failure that matters, so the dangerous error is
reported alone and never folded into an aggregate.

FinVet's own framing, from the README: *silently wrong -- a confident number
that disagrees with the filing -- is treated as the failure that matters.*

Three outcomes, kept apart:

`dangerous`  a decisive verdict opposite to the expected one. SUPPORTS where
             REFUTES was right, or the reverse. **This is the number that
             matters, and it is 0 across four runs.**
`declined`   the system returned NOT_ENOUGH_INFO or escalated where a decisive
             answer was expected. A cost in usefulness, not in correctness.
`correct`    matched the expectation.

The asymmetry is why a decline is not counted as a wrong answer. Both are
"not the expected verdict"; only one puts a false number in front of a user.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .artifacts import Run, expected, is_scored, verdict

DECISIVE = frozenset({"SUPPORTS", "REFUTES"})
OPPOSITE = {"SUPPORTS": "REFUTES", "REFUTES": "SUPPORTS"}


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
            elif got == OPPOSITE.get(want):
                # The only error that puts a false answer in front of a user.
                dangerous.append((label, row_id))
            elif got in DECISIVE and want in DECISIVE:
                # Decisive, wrong, and not the mirror image -- still dangerous.
                dangerous.append((label, row_id))
            else:
                declined += 1

    return Risk(scored=scored, correct=correct, declined=declined,
                dangerous=sorted(dangerous))
