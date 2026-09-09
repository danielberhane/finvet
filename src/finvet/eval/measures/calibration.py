"""Layer 4 — calibration: does stated confidence predict accuracy.

Measured as Expected Calibration Error. It matters here specifically because
confidence *routes* claims: `output_guardrails` sends anything below the
threshold to a human, so uncalibrated confidence makes that routing arbitrary.

Measured over the four DeepSeek runs of the frozen set, the aggregate and the
shape disagree:

    ECE = 0.1351 over 343 predictions
    ECE = 0.0391 over the 281 decisive ones

    bin    n    accuracy  avg_conf     gap
    0.0   13     0.000     0.000     +0.000
    0.2   37     1.000     0.200     +0.800
    0.3    1     1.000     0.350     +0.650
    0.5   11     1.000     0.536     +0.464
    0.9  281     1.000     0.961     +0.039

**The aggregate is misleading on its own.** Every decisive verdict held at 0.9
or above was correct -- 281 of them -- against a stated 0.961, so the system is
mildly *under*confident where it answers. Essentially all the aggregate error
comes from the low bands, where declines are correct while reporting low
confidence.

That is an overloaded scale rather than a miscalibrated one. `confidence`
measures how strongly a verdict is held, not whether the outcome was right, and
a decline is a correct outcome deliberately held weakly. The system already
compensates: `declined_with_reason` in `output_guardrails` suppresses the
low-confidence escalation for exactly these rows.

So the report separates the two populations. The split is **decisive verdict vs
decline**, not escalated vs not: an escalation reports 0.0 confidence and is
counted wrong, which is perfectly calibrated and not where the error lives. The
first split tried was escalation, and it made the "answered" figure *worse* than
the aggregate -- the declines that dominate the error are `unsupported_metric`
and `no_company_identified` rows, which never escalate.

`decisive_ece` is the number to quote for routing behaviour; `ece` is the honest
aggregate and belongs with its explanation rather than alone.
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
