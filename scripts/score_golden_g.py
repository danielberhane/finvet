#!/usr/bin/env python
"""Score a golden_g run against the pre-registered criteria A1–A13.

Joins the artifact to the dataset on id (the artifact does not carry `tags`).
Pure function of saved files; spends nothing.

Usage:
    .venv/bin/python scripts/score_golden_g.py --dataset $FINVET_GOLDEN_DIR/u/golden_u.jsonl --label u-ds --baseline auto
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.eval.measures import artifacts as A  # noqa: E402
from finvet.eval.measures import calibration, grounding, reliability, risk  # noqa: E402

CONTROL_CLASSES = frozenset({"limitation", "guard", "hitl", "market", "fines"})


def auto_baseline(ds: List[dict], runs: List[A.Run]) -> "tuple[float, int]":
    """A3 baseline computed from the golden_c rows of the SAME run set.

    Population: rows with `id < 1000`, `category == "sec/xbrl"` and a
    non-null expected verdict, correct iff `artifacts.is_correct`. Pooled
    across every run, the same way `_by_class` pools. Falls back to 1.00
    with n=0 when the run set carries no such rows (a golden_g-only artifact).
    """
    rows_c = [r for r in ds if r["id"] < 1000 and r.get("category") == "sec/xbrl"
              and r["expected"]["verdict"] is not None]
    hits = n = 0
    for row in rows_c:
        for run in runs:
            res = run.rows.get(row["id"])
            if res is None:
                continue
            n += 1
            hits += A.is_correct(res)
    if n == 0:
        return 1.00, 0
    return hits / n, n


def _by_class(ds: List[dict], runs: List[A.Run]) -> Dict[str, float]:
    hits, n = defaultdict(int), defaultdict(int)
    for row in ds:
        cls = row.get("tags", {}).get("class")
        if row.get("tags", {}).get("seed") in ("twin", "boundary_in", "boundary_out"):
            continue
        if row["strength"] != "strict" or not cls or cls in CONTROL_CLASSES:
            continue
        for run in runs:
            r = run.rows.get(row["id"])
            if not r:
                continue
            n[cls] += 1
            hits[cls] += A.is_correct(r)
    return {c: hits[c] / n[c] for c in n}


def _seed_rows(ds, seed):
    return [r["id"] for r in ds if r.get("tags", {}).get("seed") == seed]


def _rate(runs, ids, pred) -> str:
    ok = tot = 0
    for run in runs:
        for i in ids:
            r = run.rows.get(i)
            if r is None:
                continue
            tot += 1
            ok += bool(pred(r))
    return f"{ok}/{tot}"


def _all(s: str):
    """Every row passed; None when the run carries none of the rows, since a
    criterion with nothing to judge has not failed."""
    return None if _denom(s) == 0 else _frac(s) == 1.0


def _frac(s: str) -> float:
    a, b = s.split("/")
    return int(a) / int(b) if int(b) else 0.0


def _denom(s: str) -> int:
    return int(s.split("/")[1])


def _rate_per_run(runs, ids, pred) -> Dict[str, str]:
    """Like `_rate`, but one k/n per run rather than pooled across runs."""
    out = {}
    for run in runs:
        ok = tot = 0
        for i in ids:
            r = run.rows.get(i)
            if r is None:
                continue
            tot += 1
            ok += bool(pred(r))
        out[run.label or run.started_utc] = f"{ok}/{tot}"
    return out


def score(ds: List[dict], runs: List[A.Run], baseline: float) -> Dict[str, dict]:
    out = {}
    rsk = risk.measure(runs)
    out["A1"] = {"value": len(rsk.dangerous), "threshold": 0, "pass": not rsk.dangerous}
    gnd = grounding.measure(runs)
    out["A2"] = {"value": f"{gnd.traceable}/{gnd.decisive_numeric}", "threshold": "100%",
                 "pass": not gnd.untraceable}
    by_cls = _by_class(ds, runs)
    out["A3"] = {"value": by_cls, "baseline": baseline,
                 "threshold": f">= {baseline - 0.10:.2f} per class",
                 "pass": bool(by_cls) and all(v >= baseline - 0.10 for v in by_cls.values())}
    bnd = _seed_rows(ds, "boundary_in") + _seed_rows(ds, "boundary_out")
    a4 = _rate_per_run(runs, bnd, A.is_correct)
    out["A4"] = {"value": a4, "threshold": ">= 18/20 per run",
                 "pass": bool(a4) and all(_frac(v) >= 0.9 for v in a4.values())}
    lim_ids = [r["id"] for r in ds if r.get("tags", {}).get("class") == "limitation"]
    exp_lim = {r["id"]: r["expected"]["limitation"] for r in ds if r["id"] in lim_ids}
    a5 = _rate(runs, lim_ids, lambda r: (r.get("actual") or {}).get("limitation") == exp_lim[r["id"]]
               and A.verdict(r) == "NOT_ENOUGH_INFO")
    out["A5"] = {"value": a5, "threshold": "all", "pass": _all(a5)}
    grd = [r["id"] for r in ds if r.get("tags", {}).get("class") == "guard"]
    # 400 is the input guard; 422 is the API's own length validation, which
    # answers first. Either way the claim was refused with no tool called.
    a6 = _rate(runs, grd, lambda r: r.get("http") in (400, 422)
               and A.verdict(r) == "BLOCKED"
               and not (r.get("actual") or {}).get("tools_called"))
    out["A6"] = {"value": a6, "threshold": "all", "pass": _all(a6)}
    hitl = [r["id"] for r in ds if r.get("tags", {}).get("class") == "hitl"]
    a7 = _rate(runs, hitl, A.escalated)
    out["A7"] = {"value": a7,
                 "threshold": "all escalate (reviewer decisions checked manually "
                              "via hitl_* audit events)",
                 "pass": _all(a7)}
    corr = _seed_rows(ds, "corroborated")
    unc = _seed_rows(ds, "uncertifiable")
    wrong = _seed_rows(ds, "wrong_amount")
    # Ruling 18: retrieval pooling in `_penalty_observation` makes a decline the common
    # outcome on multi-fine filers, so the SUPPORTS rate is measured, not thresholded.
    # A7b asserts only that no row is inverted.
    a7b_c = _rate(runs, corr, lambda r: A.verdict(r) == "SUPPORTS")
    a7b_cn = _rate(runs, corr, lambda r: A.verdict(r) != "REFUTES")
    a7b_u = _rate(runs, unc, lambda r: (r.get("actual") or {}).get("limitation") == "amount_not_certifiable"
                  and A.verdict(r) == "NOT_ENOUGH_INFO")
    a7b_w = _rate(runs, wrong, lambda r: A.verdict(r) != "SUPPORTS")
    out["A7b"] = {"value": {"corroborated_not_refutes": a7b_cn, "uncertifiable": a7b_u,
                            "wrong_amount_not_supports": a7b_w, "corroborated_supports": a7b_c},
                  "threshold": "no corroborated REFUTES; no wrong_amount SUPPORTS; "
                               "all uncertifiable amount_not_certifiable "
                               "(corroborated_supports measured, not thresholded)",
                  "pass": _frac(a7b_cn) == 1.0 and _frac(a7b_u) == 1.0 and _frac(a7b_w) == 1.0}
    twins = [r for r in ds if r.get("tags", {}).get("seed") == "twin"]
    tw = _rate(runs, [r["id"] for r in twins], A.is_correct)
    orig = _rate(runs, [r["tags"]["twin_of"] for r in twins], A.is_correct)
    out["A8"] = {"value": {"twins": tw, "originals": orig}, "threshold": "within 5 pts",
                 "pass": None if _denom(tw) == 0 or _denom(orig) == 0
                 else _frac(orig) - _frac(tw) <= 0.05}
    cal = calibration.measure(runs)
    out["A9"] = {"value": round(cal.decisive_ece, 4), "threshold": "<= 0.05",
                 "pass": cal.decisive_ece <= 0.05}
    out["A10"] = {"value": "compare A1/A2/A5/A6 across the two vendor labels",
                  "threshold": "identical", "pass": None}
    rel = reliability.measure(runs)
    out["A11"] = {"value": round(rel.pass_hat_k, 4), "threshold": ">= 0.90 at k>=3",
                  "pass": None if rel.k < 3 else rel.pass_hat_k >= 0.90}
    a12 = {"supports": _rate(runs, _seed_rows(ds, "snapshot_supports"), A.is_correct),
           "refutes": _rate(runs, _seed_rows(ds, "snapshot_refutes"), A.is_correct)}
    out["A12"] = {"value": a12, "threshold": "all",
                  "pass": None if all(_denom(v) == 0 for v in a12.values())
                  else all(_frac(v) == 1.0 for v in a12.values() if _denom(v))}
    plain = _rate(runs, _seed_rows(ds, "plain"), A.is_correct)
    seeded = _rate(runs, _seed_rows(ds, "quarter") + _seed_rows(ds, "operator") + _seed_rows(ds, "loss"),
                   A.is_correct)
    out["A13"] = {"value": {"plain": plain, "seeded": seeded}, "threshold": "within 5 pts",
                  "pass": None if _denom(plain) == 0 or _denom(seeded) == 0
                  else _frac(plain) - _frac(seeded) <= 0.05}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dir", default=os.environ.get("FINVET_GOLDEN_DIR"))
    ap.add_argument("--label", required=True, help="substring; e.g. g-ds")
    ap.add_argument("--baseline", default="auto",
                    help="'auto' (default): computed from the golden_c sec/xbrl rows "
                         "(id < 1000) of this same run set, correct iff artifacts.is_correct; "
                         "falls back to 1.00 with a printed note when this run set carries no "
                         "such rows. Or a numeric override, e.g. 1.00")
    args = ap.parse_args()
    ds = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    runs = [r for r in A.load_runs(A.find_runs(Path(args.dir))) if r.complete and args.label in r.label]
    if not runs:
        sys.exit(f"no complete runs labelled *{args.label}* in {args.dir}")
    if args.baseline == "auto":
        baseline, n = auto_baseline(ds, runs)
        if n:
            print(f"A3 baseline (auto): {baseline:.4f} from {n} golden_c sec/xbrl row(s)")
        else:
            print("A3 baseline (auto): no golden_c sec/xbrl rows in this run set; "
                  "falling back to 1.00")
    else:
        baseline = float(args.baseline)
        print(f"A3 baseline (override): {baseline:.4f}")
    out = score(ds, runs, baseline)
    failed = False
    for k, v in out.items():
        mark = "PASS" if v["pass"] else ("----" if v["pass"] is None else "FAIL")
        failed |= v["pass"] is False
        print(f"{k:<4} {mark}  {v['value']}   (threshold {v['threshold']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
