#!/usr/bin/env python
"""Report the measurement layers over recorded golden runs.

Reads artifacts written by `scripts/run_golden.py`. Calls no API, spends
nothing, and can be re-run against the same artifacts indefinitely -- the runs
were paid for once.

The seven layers: outcome, trajectory, grounding, calibration, asymmetric risk,
reliability, reachability. Five are
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
    artifacts, calibration, grounding, reliability, risk, routing, trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.environ.get("FINVET_GOLDEN_DIR"))
    parser.add_argument("--json", dest="json_path",
                        help="write the layer summary here (an OUTPUT path)")
    parser.add_argument("--label", action="append", default=[],
                        help="measure only runs whose label contains this; "
                             "repeatable. Default: every complete run.")
    args = parser.parse_args()

    if not args.dir:
        print("Set FINVET_GOLDEN_DIR or pass --dir.", file=sys.stderr)
        return 2

    paths = artifacts.find_runs(Path(args.dir))

    # --json is an output path. Pointed at a run artifact -- which reads as the
    # natural way to say "measure this one file" -- it overwrote 100 recorded
    # rows with this summary, and those rows cost real API spend and cannot be
    # rebuilt. Refusing costs one comparison; the alternative is silent loss.
    if args.json_path:
        out = Path(args.json_path).resolve()
        if out in {p.resolve() for p in paths}:
            print(f"--json would overwrite the run artifact {out.name}. It is an "
                  f"output path, not a selector -- use --label to choose runs.",
                  file=sys.stderr)
            return 2

    runs = [r for r in artifacts.load_runs(paths) if r.complete]
    if args.label:
        runs = [r for r in runs
                if any(sub in (r.label or "") for sub in args.label)]
    # Probe and resilience artifacts are partial slices of the set; including
    # them would compare rows that were never run against each other. Naming a
    # run explicitly overrides that -- a caller who asks for a 40-row segment
    # by label has said which rows they mean.
    if not args.label:
        runs = [r for r in runs if len(r) >= 80]
    if not runs:
        print(f"No complete runs in {args.dir}", file=sys.stderr)
        return 2

    # Per run, not one summary line. These layers pool rows across runs, and
    # pooling two models reads as one -- a segment run under a different model
    # than its label suggests was invisible until the model was printed beside
    # every artifact.
    models = {r.model for r in runs}
    print(f"\n  runs: {len(runs)}")
    for r in runs:
        print(f"    {r.label or r.started_utc:<24} {len(r):>3} rows   {r.model}")
    if len(models) > 1:
        print(f"    NOTE: {len(models)} models pooled {sorted(models)} -- "
              f"these layers mix them. Use --label to separate.")

    rel = reliability.measure(runs)
    cal = calibration.measure(runs)
    gnd = grounding.measure(runs)
    rsk = risk.measure(runs)
    trj = trajectory.measure(runs)
    rte = routing.measure(runs)

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

    print(f"\n  ROUTING       sources used matched expectation "
          f"{rte.matched}/{rte.scored} ({rte.rate:.1%})")
    for row_id, (want, got) in sorted(rte.mismatches.items()):
        print(f"           id {row_id}: expected {want} but used {got}")
    if rte.skipped_unlabelled:
        print(f"           skipped (no expected verdict, sources unlabelled): "
              f"{rte.skipped_unlabelled}")

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
            "trajectory": asdict(trj), "routing": asdict(rte),
        }, indent=1, default=str))
        print(f"\n  written to {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
