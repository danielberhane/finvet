#!/usr/bin/env python
"""Report the measurement layers over recorded golden runs.

Reads artifacts written by `scripts/run_golden.py`. Calls no API, spends
nothing, and can be re-run against the same artifacts indefinitely -- the runs
were paid for once.

The seven layers are defined in `docs/AGENTIC_EVAL_GUIDE.md` §1.2. Five are
statistics over runs and are computed here. Layer 1 (outcome) is asserted by
`tests/integration/test_golden.py`, and Layer 7 (reachability) asks whether a
behaviour can occur at all, which no statistic answers.

Usage:
    FINVET_GOLDEN_DIR=~/Projects/Active/finvet-golden \\
      .venv/bin/python scripts/eval_layers.py
    .venv/bin/python scripts/eval_layers.py --dir path/to/runs --json out.json
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finvet.eval.measures import (  # noqa: E402
    artifacts, calibration, grounding, reliability, risk, trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.environ.get("FINVET_GOLDEN_DIR"))
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    if not args.dir:
        print("Set FINVET_GOLDEN_DIR or pass --dir.", file=sys.stderr)
        return 2

    paths = artifacts.find_runs(Path(args.dir))
    runs = [r for r in artifacts.load_runs(paths) if r.complete]
    # Probe and resilience artifacts are partial slices of the set; including
    # them would compare rows that were never run against each other.
    runs = [r for r in runs if len(r) >= 80]
    if not runs:
        print(f"No complete runs in {args.dir}", file=sys.stderr)
        return 2

    print(f"\n  runs: {len(runs)}   model: {runs[-1].model}")
    for r in runs:
        print(f"    {r.label or r.started_utc:<24} {len(r):>3} rows")

    rel = reliability.measure(runs)
    cal = calibration.measure(runs)
    gnd = grounding.measure(runs)
    rsk = risk.measure(runs)
    trj = trajectory.measure(runs)

    print(f"\n  LAYER 6  reliability   pass@1 {rel.pass_at_1:.3f}   "
          f"pass^{rel.k} {rel.pass_hat_k:.3f}   n={rel.n}")
    if rel.unstable:
        print(f"           unstable rows: {rel.unstable}")

    print(f"\n  LAYER 2  trajectory    tool correctness "
          f"{trj.required_met}/{trj.scored} ({trj.correctness:.1%})"
          f"   [{trj.scorer}]")
    print(f"           lane violations {len(trj.lane_violations)}   "
          f"zero-tool {trj.zero_tool_actual}/{trj.zero_tool_expected - trj.not_executed} "
          f"executed ({trj.not_executed} blocked before execution)")
    for row_id, why in trj.missing_required[:6]:
        print(f"           MISSING  id {row_id}: {why}")
    for row_id, why in trj.lane_violations[:6]:
        print(f"           OUT OF LANE  id {row_id}: {why}")
    if trj.zero_tool_breaches:
        print(f"           SPENT WHEN IT SHOULD NOT: {trj.zero_tool_breaches}")

    print(f"\n  LAYER 4  calibration   ECE {cal.ece:.4f} over {cal.n} predictions")
    print(f"           decisive only:  ECE {cal.decisive_ece:.4f} "
          f"over {cal.decisive_n}")
    print(f"           {'bin':>5}{'n':>6}{'accuracy':>10}{'conf':>8}{'gap':>8}")
    for b in cal.bins:
        print(f"           {b.lower:>5.1f}{b.n:>6}{b.accuracy:>10.3f}"
              f"{b.mean_confidence:>8.3f}{b.gap:>+8.3f}")

    print(f"\n  LAYER 3  grounding     {gnd.traceable}/{gnd.decisive_numeric} "
          f"decisive numbers traceable ({gnd.rate:.1%})")
    if gnd.untraceable:
        print(f"           UNTRACEABLE: {gnd.untraceable}")

    print(f"\n  LAYER 5  asymmetric    dangerous errors {len(rsk.dangerous)} "
          f"({rsk.dangerous_rate:.2%} of {rsk.scored})")
    print(f"           declined {rsk.declined} ({rsk.decline_rate:.1%})   "
          f"correct {rsk.correct}")
    if rsk.dangerous:
        print(f"           DANGEROUS: {rsk.dangerous}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps({
            "runs": [r.label or r.started_utc for r in runs],
            "reliability": asdict(rel), "calibration": asdict(cal),
            "grounding": asdict(gnd), "risk": asdict(rsk),
            "trajectory": asdict(trj),
        }, indent=1, default=str))
        print(f"\n  written to {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
