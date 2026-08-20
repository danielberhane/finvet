"""Contract conformance: every gold row constructs a ParsedClaim, losslessly.

The decisive measurement of the migration's EXPAND phase. Before stage 03,
test.jsonl scored 287/383 (74.9%) — reject_reason Literal failures — and
metric/operator were silently dropped on every row that passed. The contract
holds only when every row constructs AND every field round-trips.

The gold is contamination-sensitive and lives in the sibling repo: read by
path, never copied here; skip when absent.
"""

import json
from pathlib import Path

import pytest

from finvet.models.claim import ParsedClaim

GOLD_DIR = Path.home() / "Projects/Active/claim_parser_fine-tuned/data/clean"
CONTRACT_FIELDS = ("claim_type", "ticker", "metric", "operator",
                   "value", "period", "reject_reason")

pytestmark = pytest.mark.skipif(
    not GOLD_DIR.exists(), reason="claim-parser gold not present"
)


def _rows(name):
    return [json.loads(line) for line in open(GOLD_DIR / name) if line.strip()]


@pytest.mark.parametrize("split,expected_total", [
    ("test.jsonl", 383),
    ("heldout_real_sourced.jsonl", 523),
])
def test_every_gold_row_constructs_and_round_trips(split, expected_total):
    rows = _rows(split)
    assert len(rows) == expected_total, "gold split changed size — investigate"

    failures = []
    for row in rows:
        gold = row["gold"]
        try:
            claim = ParsedClaim(**gold)
        except Exception as e:
            failures.append(f"row {row['id']}: {type(e).__name__}: {e}")
            continue
        for field in CONTRACT_FIELDS:
            if getattr(claim, field) != gold.get(field):
                failures.append(
                    f"row {row['id']}: {field} lost — gold "
                    f"{gold.get(field)!r} became {getattr(claim, field)!r}"
                )
    assert not failures, (
        f"{len(failures)}/{len(rows)} rows fail the contract:\n  "
        + "\n  ".join(failures[:15])
    )
