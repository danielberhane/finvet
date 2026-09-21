"""SUPERSEDED -- DO NOT RUN. Range claims decline; the parser contract has no range bounds.

This script added `range_min`/`range_max` to the 32 range rows of test.jsonl
(decision D16). It migrated only that split: train.jsonl's 4,578 rows,
including all 282 range rows, still carry the midpoint encoding with no
bounds. The model is therefore trained to emit a bandless range and graded
against bounds it was never taught, understating its accuracy.

FinVet has since removed range_min/range_max from ParsedClaim entirely, so
re-running this would produce gold that no longer matches the contract. The
pre-migration file is preserved at data/clean/test.jsonl.pre-range-bounds.

Kept for the record, not for use.

--- original description follows ---

Give every gold `operator="range"` row the bounds the contract requires.

32 of the 383 rows in test.jsonl encode a range as `operator="range"` plus a
single `value` holding the midpoint, and carry no `range_min`/`range_max`.
That is the pre-contract encoding, and it is the encoding that produced a real
defect: collapsing "between $50B and $150B" to $100B and comparing it as
equality refuted a filed $149B at a 32.89% difference, though it sits plainly
inside the stated band. `_apply_override` now tests interval membership and
`ParsedClaim` requires both bounds, so these rows no longer construct.

The bounds are recoverable because every affected row states them in its own
input text, and the recorded midpoint validates the reading: bounds are only
written when `(min + max) / 2` reproduces the stored `value`. A row whose
midpoint does not reconcile is reported and left alone rather than guessed at.

The scale word is read the same way. "$29 billion and $31 billion" and "$124
and $128" differ only in magnitude, and trying each multiplier against the
recorded value settles which was meant without the script having to parse
units correctly on its own.

Dry run by default; pass --apply to write.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.eval.dataset import eval_data_dir  # noqa: E402

# Ordered by specificity: "ranged from A to B" must win over a bare "A to B".
RANGE_PATTERNS = [
    r"between\s+[\$€£¥₩]?\s*([\d,.]+)\s*%?\s*(?:\w+\s+)?(?:and|to)\s+[\$€£¥₩]?\s*([\d,.]+)",
    r"ranged?\s+from\s+[\$€£¥₩]?\s*([\d,.]+)\s*%?\s*(?:\w+\s+)?to\s+[\$€£¥₩]?\s*([\d,.]+)",
    r"from\s+[\$€£¥₩]?\s*([\d,.]+)\s*%?\s*(?:\w+\s+)?to\s+[\$€£¥₩]?\s*([\d,.]+)",
    r"[\$€£¥₩]?\s*([\d,.]+)\s*%?\s*(?:\w+\s+)?to\s+[\$€£¥₩]?\s*([\d,.]+)\s*%?\s*(?:\w+\s+)?range",
]

MULTIPLIERS = [1.0, 1e3, 1e6, 1e9, 1e12]


def _number(raw):
    try:
        return float(raw.replace(",", "").rstrip("."))
    except ValueError:
        return None


def derive_bounds(text, value):
    """(low, high) whose midpoint reproduces `value`, or None.

    Every candidate is checked against the recorded midpoint, so a wrong
    pattern match or a misread scale fails closed instead of writing a
    plausible-looking bound nobody verified.
    """
    for pattern in RANGE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        low, high = _number(match.group(1)), _number(match.group(2))
        if low is None or high is None or low >= high:
            continue
        for scale in MULTIPLIERS:
            lo, hi = low * scale, high * scale
            midpoint = (lo + hi) / 2
            if value and abs(midpoint - value) <= max(abs(value) * 1e-6, 1e-9):
                return lo, hi
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the migrated rows (default: dry run)")
    parser.add_argument("--split", default="test.jsonl")
    args = parser.parse_args()

    gold_dir = eval_data_dir()
    if not gold_dir:
        print("FINVET_EVAL_DATA_DIR is not set; nothing to migrate.")
        return 1

    path = Path(gold_dir) / args.split
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    migrated, unresolved = [], []
    for row in rows:
        gold = row["gold"]
        if gold.get("operator") != "range":
            continue
        if gold.get("range_min") is not None and gold.get("range_max") is not None:
            continue

        bounds = derive_bounds(row.get("input", ""), gold.get("value"))
        if bounds is None:
            unresolved.append(row)
            continue

        gold["range_min"], gold["range_max"] = bounds
        migrated.append(row)

    for row in migrated:
        gold = row["gold"]
        print(f"  id={row['id']:<5} {gold['range_min']:>18,.4g} .. "
              f"{gold['range_max']:<18,.4g} mid={gold['value']:,.4g}")
        print(f"        {row['input']}")

    print(f"\nmigrated: {len(migrated)}   unresolved: {len(unresolved)}")
    for row in unresolved:
        print(f"  UNRESOLVED id={row['id']}: {row['input']}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    if unresolved:
        print("\nRefusing to write while rows remain unresolved.")
        return 1

    backup = path.with_suffix(path.suffix + ".pre-range-bounds")
    shutil.copy2(path, backup)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    print(f"\nWrote {path} ({len(rows)} rows). Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
