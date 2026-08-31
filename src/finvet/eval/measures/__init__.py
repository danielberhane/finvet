"""Measurement layers for an agentic system, one module per layer.

The layers are defined in `docs/AGENTIC_EVAL_GUIDE.md` §1.2. Each module here
implements exactly one of them, exposes a single `measure(runs)` function, and
imports nothing from its siblings -- only `artifacts`, which loads a recorded
run and decides what "correct" means.

    reliability   Layer 6   pass@1, pass^k
    calibration   Layer 4   ECE, split by answered vs declined
    grounding     Layer 3   decisive numbers traceable to a source
    risk          Layer 5   dangerous errors, reported alone
    trajectory    Layer 2   tool selection against the routing design
    routing       --        sources used vs the sources expected

Layer 1 (outcome) is `tests/integration/test_golden.py`, and Layer 7
(reachability) is answered by whether a capability can occur at all -- neither
is a statistic over runs.

Every measure is a pure function of saved artifacts: no API calls, no cost, and
a run paid for once can be re-analysed indefinitely.
"""

from . import (artifacts, calibration, grounding, reliability, risk,
               routing, trajectory)

__all__ = ["artifacts", "calibration", "grounding", "reliability",
           "risk", "routing", "trajectory"]
