"""Burned evaluation rows — permanently excluded from parser benchmarks.

A row is burned when its claim text entered model-facing material or public
git history: a 2026-08-20 prompt revision quoted four eval rows verbatim (that revision
existed in version control, so replacing the examples cannot un-expose them),
and one heldout row's claim text sat in a committed test fixture. Burned rows
never count in a parser benchmark again; deterministic uses are unaffected
(retrieval gold compares code output to SEC values — nothing learns from it).

The internal evaluation-data policy carries the incident log.
"""

BURNED_ROWS: dict[str, dict[int, str]] = {
    "test.jsonl": {
        1: "quoted verbatim in a 2026-08-20 prompt revision",
        11: "quoted verbatim in a 2026-08-20 prompt revision",
        18: "quoted verbatim in a 2026-08-20 prompt revision",
    },
    "val.jsonl": {
        257: "quoted verbatim in a 2026-08-20 prompt revision",
    },
    "heldout_real_sourced.jsonl": {
        12: "claim text appeared in a committed test fixture (2026-08-20); "
            "retrieval-gold use unaffected — deterministic",
    },
}


def burned_ids(split: str) -> frozenset[int]:
    return frozenset(BURNED_ROWS.get(split, {}))
