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
    "golden_c.jsonl": {
        1: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
        68: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
        88: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
    },
    "golden_g.jsonl": {
        1079: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
        1151: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
        1186: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
    },
    "golden_u.jsonl": {
        1: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
        68: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
        88: "published in full in docs/eval/DATASET_CARD.md (2026-09-04; section since removed, text remains in git history)",
        1079: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
        1151: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
        1186: "published in full in docs/eval/GOLDEN_G_CARD.md (2026-09-28)",
    },
}
# v1.2 is v1.1 regraded (2026-10-12); a version does not unburn a row.
BURNED_ROWS["golden_u_v1.2.jsonl"] = dict(BURNED_ROWS["golden_u.jsonl"])


def burned_ids(split: str) -> frozenset[int]:
    return frozenset(BURNED_ROWS.get(split, {}))
