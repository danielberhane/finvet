"""Measurement layers for an agentic system, one module per layer.

Each module implements one layer, exposes a single measure(runs) function, and
imports nothing from its siblings except artifacts, which loads a recorded run
and defines what counts as correct.

    reliability   Layer 6   pass@1, pass^k
    calibration   Layer 4   ECE, split by answered and declined
    grounding     Layer 3   decisive numbers traceable to a source
    risk          Layer 5   dangerous errors, reported alone
    trajectory    Layer 2   tool selection against the routing design
    routing       --        sources used against the sources expected

Layer 1 (outcome) lives in tests/integration/test_golden.py. Layer 7
(reachability) asks whether a capability can occur at all. Neither is a
statistic over runs.

Every measure is a pure function of saved artifacts, so a run paid for once can
be re-analysed without further API calls.
"""

from . import (artifacts, calibration, grounding, reliability, risk,
               routing, trajectory)

__all__ = ["artifacts", "calibration", "grounding", "reliability",
           "risk", "routing", "trajectory"]
