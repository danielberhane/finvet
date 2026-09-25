"""Layer 4 — calibration: whether stated confidence predicts accuracy.

Measured as Expected Calibration Error. It matters here because confidence
routes claims: output_guardrails sends anything below the threshold to a human,
so uncalibrated confidence makes that routing arbitrary.

Two figures are reported. ece covers every prediction. decisive_ece covers only
decisive verdicts, and is the one to quote for routing behaviour.

The split is decisive verdict against decline, not escalated against not. A
decline is a correct outcome held deliberately at low confidence, so it enters
the aggregate as error without being a miscalibration. confidence measures how
strongly a verdict is held, not whether the outcome was right.
"""

from dataclasses import dataclass, field
from typing import List

from .artifacts import Run, confidence, is_correct, is_scored, verdict

BIN_COUNT = 10
DECISIVE = frozenset({"SUPPORTS", "REFUTES", "REJECTED", "BLOCKED"})


@dataclass(frozen=True)
class Bin:
    lower: float
    n: int
    accuracy: float
    mean_confidence: float

    @property
    def gap(self) -> float:
        return self.accuracy - self.mean_confidence


@dataclass(frozen=True)
class Calibration:
    n: int
    ece: float
    decisive_ece: float
    decisive_n: int
    bins: List[Bin] = field(default_factory=list)


def _ece(pairs) -> float:
    """Expected Calibration Error: the accuracy/confidence gap, weighted by bin."""
    if not pairs:
        return 0.0
    buckets = {}
    for conf, ok in pairs:
        b = min(int(conf * BIN_COUNT), BIN_COUNT - 1)
        n, hits, total = buckets.get(b, (0, 0, 0.0))
        buckets[b] = (n + 1, hits + int(ok), total + conf)
    total_n = len(pairs)
    return sum(n / total_n * abs(hits / n - conf_sum / n)
               for n, hits, conf_sum in buckets.values() if n)


def measure(runs: List[Run]) -> Calibration:
    """ECE over every scored prediction in every run.

    Runs are pooled rather than averaged: calibration is a property of the
    confidence *scale*, and more observations of that scale make the estimate
    better regardless of which run produced them.
    """
    pairs, decisive = [], []
    for run in runs:
        for row in run.rows.values():
            conf = confidence(row)
            if conf is None or not is_scored(row):
                continue
            pair = (float(conf), is_correct(row))
            pairs.append(pair)
            # A decisive outcome states something about the world. A decline
            # states that the system would not, and its confidence is not a
            # prediction about a verdict -- so it is measured separately.
            if verdict(row) in DECISIVE:
                decisive.append(pair)

    buckets = {}
    for conf, ok in pairs:
        b = min(int(conf * BIN_COUNT), BIN_COUNT - 1)
        n, hits, total = buckets.get(b, (0, 0, 0.0))
        buckets[b] = (n + 1, hits + int(ok), total + conf)

    bins = [Bin(lower=b / BIN_COUNT, n=n, accuracy=hits / n,
                mean_confidence=conf_sum / n)
            for b, (n, hits, conf_sum) in sorted(buckets.items()) if n]

    return Calibration(n=len(pairs), ece=_ece(pairs),
                       decisive_ece=_ece(decisive), decisive_n=len(decisive),
                       bins=bins)
