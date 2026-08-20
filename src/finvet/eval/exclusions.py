"""Burned evaluation rows — permanently excluded from parser benchmarks.

A row is burned when its claim text entered model-facing material or public
git history: the stage-06 prompt quoted four eval rows verbatim (the prompt
is in git history forever, so replacing the examples cannot un-expose them),
and one heldout row's claim text sat in a committed test fixture. Burned rows
never count in a parser benchmark again; deterministic uses are unaffected
(retrieval gold compares code output to SEC values — nothing learns from it).

docs/EVAL_DATA_POLICY.md carries the incident log.
"""

BURNED_ROWS: dict[str, dict[int, str]] = {
    "test.jsonl": {
        1: "quoted verbatim in the stage-06 prompt (7b4cee1)",
        11: "quoted verbatim in the stage-06 prompt (7b4cee1)",
        18: "quoted verbatim in the stage-06 prompt (7b4cee1)",
    },
    "val.jsonl": {
        257: "quoted verbatim in the stage-06 prompt (7b4cee1)",
    },
    "heldout_real_sourced.jsonl": {
        12: "claim text committed in a test fixture (02228e8); "
            "retrieval-gold use unaffected — deterministic",
    },
}


def burned_ids(split: str) -> frozenset[int]:
    return frozenset(BURNED_ROWS.get(split, {}))
