"""Layer 6 — reliability: whether a case succeeds on every attempt, not just one.

Agents are stochastic, so a single pass cannot separate solving a case from
getting lucky. pass^k, from tau-bench, is the rate at which a case succeeds on
all k attempts. pass@1 is the rate on the latest run alone.

Only rows present in every run are counted, and rows whose verdict moved for a
reason outside the system are excluded. A difference with a known cause is not
evidence of non-determinism.

unstable names the rows that did not agree with themselves across runs. Those
cannot carry a comparison against another model later, so they are listed
rather than folded into the rate.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .artifacts import Run, is_correct, stable_ids, verdict


@dataclass(frozen=True)
class Reliability:
    n: int
    k: int
    pass_at_1: float
    pass_hat_k: float
    unstable: List[int] = field(default_factory=list)
    per_run: Dict[str, float] = field(default_factory=dict)

    @property
    def flake_rate(self) -> float:
        """Fraction of rows that disagreed with themselves across runs."""
        return len(self.unstable) / self.n if self.n else 0.0


def measure(runs: List[Run]) -> Reliability:
    """pass@1 and pass^k over the rows every run shares.

    pass@1 is taken from the most recent run rather than averaged: it answers
    "how does the system behave now", which is the question a single run asks.
    """
    ids = stable_ids(runs)
    if not ids or not runs:
        return Reliability(n=0, k=len(runs), pass_at_1=0.0, pass_hat_k=0.0)

    latest = runs[-1]
    pass_1 = sum(is_correct(latest.rows[i]) for i in ids) / len(ids)
    pass_k = sum(all(is_correct(run.rows[i]) for run in runs) for i in ids) / len(ids)

    unstable = [i for i in ids
                if len({verdict(run.rows[i]) for run in runs}) > 1]

    per_run = {
        (run.label or run.started_utc):
            sum(is_correct(run.rows[i]) for i in ids) / len(ids)
        for run in runs
    }
    return Reliability(n=len(ids), k=len(runs), pass_at_1=pass_1,
                       pass_hat_k=pass_k, unstable=unstable, per_run=per_run)
