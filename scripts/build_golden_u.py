#!/usr/bin/env python
"""Assemble golden_u.jsonl: the union of golden_c (97 rows) and golden_g (256).

The two sets share one schema and disjoint id ranges (1-100, 1001-1257), so
the union is a concatenation -- but a union of two independently built sets
can hide duplicates the ids do not reveal: the same claim text, or the same
(ticker, metric, period, value) asked twice. This script refuses to write if
any such collision exists, and prints the checks it ran so the assessment is
reproducible.

golden_c is read through $FINVET_GOLDEN_DIR (never named on a command line,
per .claude/hooks/protect-artifacts.sh); golden_g is passed as a path.

Usage:
    FINVET_GOLDEN_DIR=/path/to/golden/repo .venv/bin/python scripts/build_golden_u.py \
        --g /path/to/golden_g.jsonl --out /path/to/golden_u.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.eval.dataset import golden_data_file  # noqa: E402

CORE_KEYS = {"id", "claim", "category", "strength", "expected",
             "ground_truth", "source", "gold_parse"}
OPTIONAL_KEYS = {"tags", "recipe"}


def _load(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm(text: str) -> str:
    """Lower-case, strip punctuation and whitespace runs -- catches rephrasings
    that differ only in case or punctuation."""
    return re.sub(r"[^a-z0-9$%.]+", " ", text.lower()).strip()


def _fact_key(row: Dict):
    """What the row asks: ticker, metric, period, operator, value. Two rows with
    the same key are the same question even if phrased differently."""
    g = row.get("gold_parse") or {}
    if not g.get("claim_type") or g.get("claim_type") == "reject":
        return None
    return (g.get("ticker"), g.get("metric"), g.get("period"),
            g.get("operator"), g.get("value"))


def assess(rows: List[Dict]) -> List[str]:
    """Every check the union must pass. Returns the list of failures."""
    fails: List[str] = []

    ids = [r["id"] for r in rows]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    if dup_ids:
        fails.append(f"duplicate ids: {sorted(dup_ids)}")

    for r in rows:
        extra = set(r) - CORE_KEYS - OPTIONAL_KEYS
        missing = CORE_KEYS - set(r)
        if extra or missing:
            fails.append(f"id {r['id']}: keys extra={sorted(extra)} missing={sorted(missing)}")
        if set(r["expected"]) != {"verdict", "limitation", "sources"}:
            fails.append(f"id {r['id']}: expected keys {sorted(r['expected'])}")
        if r["strength"] == "observe" and (r["expected"]["verdict"] is not None
                                           or r["expected"]["limitation"] is not None):
            fails.append(f"id {r['id']}: observe row with a non-null expectation")
        if re.search(r"\b\d{9,10}\b", r["claim"]):
            fails.append(f"id {r['id']}: bare 9-10 digit run in claim (guard would block)")

    # A live-price recipe row's text is a template the runner fills, so two
    # recipe rows on one ticker legitimately share it; they differ by offset.
    by_text: Dict[str, List[int]] = {}
    for r in rows:
        rec = r.get("recipe")
        key = (f"recipe:{rec['ticker']}:{rec['offset_pct']}" if rec
               else _norm(r["claim"]))
        by_text.setdefault(key, []).append(r["id"])
    for text, who in by_text.items():
        if len(who) > 1:
            fails.append(f"same claim text: ids {who}")

    by_fact: Dict[tuple, List[int]] = {}
    for r in rows:
        k = _fact_key(r)
        if k and k[4] is not None:
            by_fact.setdefault(k, []).append(r["id"])
    twins = {r["id"] for r in rows if (r.get("tags") or {}).get("seed") == "twin"}
    observe = {r["id"] for r in rows if r["strength"] == "observe"}
    for k, who in by_fact.items():
        if len(who) > 1 and not any(i in twins for i in who):
            # A phrasing twin deliberately repeats its original's fact. Observe
            # rows are never asserted, so a repeated fact among them is a
            # redundancy to report, not a duplicate that corrupts a score.
            if all(i in observe for i in who):
                WARNINGS.append(f"observe rows share one fact {k}: ids {who}")
            else:
                fails.append(f"same fact asked twice (ticker, metric, period, op, value)={k}: ids {who}")

    return fails


WARNINGS: List[str] = []


def summarize(rows: List[Dict]) -> str:
    c = lambda key: Counter(key(r) for r in rows)  # noqa: E731
    lines = [
        f"rows: {len(rows)}   ids: {min(r['id'] for r in rows)}..{max(r['id'] for r in rows)}",
        f"from golden_c (id < 1000): {sum(1 for r in rows if r['id'] < 1000)}   "
        f"from golden_g: {sum(1 for r in rows if r['id'] >= 1000)}",
        f"strength: {dict(c(lambda r: r['strength']))}",
        f"expected verdict: {dict(c(lambda r: r['expected']['verdict']))}",
        f"category: {dict(sorted(c(lambda r: r['category']).items()))}",
        f"distinct tickers: {len({(r.get('gold_parse') or {}).get('ticker') for r in rows} - {None})}",
        f"tickers in both sets: "
        f"{sorted(({(r.get('gold_parse') or {}).get('ticker') for r in rows if r['id'] < 1000} & {(r.get('gold_parse') or {}).get('ticker') for r in rows if r['id'] >= 1000}) - {None})}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", required=True, help="path to golden_g.jsonl")
    ap.add_argument("--out", required=True, help="where to write golden_u.jsonl")
    args = ap.parse_args()

    c_path = golden_data_file("golden_c.jsonl")
    if c_path is None or not c_path.exists():
        sys.exit("set FINVET_GOLDEN_DIR to the golden repo; golden_c not found")
    rows = _load(c_path) + _load(Path(args.g))
    rows.sort(key=lambda r: r["id"])

    fails = assess(rows)
    print(summarize(rows))
    for w in WARNINGS:
        print("  WARNING " + w)
    if fails:
        print("\nREFUSING TO WRITE -- failures:")
        for f in fails:
            print("  " + f)
        return 1

    out = Path(args.out)
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    out.write_text(text)
    print(f"\nall checks passed; wrote {len(rows)} rows to {out}")
    print(f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
