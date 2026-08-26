#!/usr/bin/env python
"""Add filing-position identity to `filing_chunks`, and report what is missing.

Release A cites retrieved passages by form, part, item, period, chunk index and
a content-derived `evidence_id`. Legacy rows predate all of that: they carry a
semantic section name (`mda`, `risk_factors`) and nothing that says which part
of a 10-Q they came from.

**Part and item are not inferable from the section name.** A 10-Q restarts item
numbering in each part, so "Item 2" is Management's Discussion in Part I and
Unregistered Sales in Part II. Guessing would produce citations that look
precise and point at the wrong place, which is worse than admitting the rows
are unidentified. Legacy rows must be re-ingested from the source filings.

The script is therefore deliberately unable to finish the job on its own:

    1. `--apply` adds the three nullable columns. Safe on a populated table.
    2. Re-ingest (`--truncate-for-reingest --i-understand-this-deletes
       filing_chunks`, then `python -m finvet.rag.ingest`) repopulates identity.
    3. `--apply` again creates the unique `evidence_id` index, but only once
       no row is missing identity.

Dry run by default; nothing is written without `--apply`.

Usage:
    .venv/bin/python scripts/migrate_rag_release_a.py
    .venv/bin/python scripts/migrate_rag_release_a.py --apply
    .venv/bin/python scripts/migrate_rag_release_a.py --apply \\
        --truncate-for-reingest --i-understand-this-deletes filing_chunks
    .venv/bin/python scripts/migrate_rag_release_a.py --json /tmp/rag-migration.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from finvet.config.database import get_db_session  # noqa: E402

TABLE = "filing_chunks"

# Added nullable on purpose. The table holds rows that cannot be given identity
# without re-ingestion, and a NOT NULL constraint would abort the migration on
# exactly the rows it exists to surface. The unique index on evidence_id is
# created later, once nothing is missing.
NEW_COLUMNS = {
    "part": "VARCHAR(10)",
    "item_number": "VARCHAR(10)",
    "evidence_id": "VARCHAR(64)",
}

EVIDENCE_INDEX = "idx_filing_chunks_evidence_id"


def _existing_columns(session) -> set:
    rows = session.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :t"), {"t": TABLE}).fetchall()
    return {r[0] for r in rows}


def _index_exists(session, name: str) -> bool:
    return session.execute(text(
        "SELECT 1 FROM pg_indexes WHERE tablename = :t AND indexname = :i"),
        {"t": TABLE, "i": name}).fetchone() is not None


def _corpus_state(session, columns: set) -> dict:
    total = session.execute(text(f"SELECT count(*) FROM {TABLE}")).scalar() or 0
    filings = session.execute(text(
        f"SELECT count(*) FROM (SELECT DISTINCT ticker, filing_type, period_end "
        f"FROM {TABLE}) s")).scalar() or 0

    missing = None
    if {"part", "item_number", "evidence_id"} <= columns:
        # part is legitimately NULL for a 10-K, so it cannot indicate absence.
        # item_number and evidence_id must be present on every row.
        missing = session.execute(text(
            f"SELECT count(*) FROM {TABLE} "
            f"WHERE evidence_id IS NULL OR item_number IS NULL")).scalar() or 0

    # A checksum over the corpus's identity, so a report can be tied to the
    # exact contents it described.
    rows = session.execute(text(
        f"SELECT ticker, filing_type, period_end, section, chunk_index "
        f"FROM {TABLE} ORDER BY ticker, filing_type, period_end, section, "
        f"chunk_index")).fetchall()
    digest = hashlib.sha256(
        "\n".join("|".join(str(v) for v in r) for r in rows).encode("utf-8")
    ).hexdigest()

    return {"total_chunks": total, "distinct_filings": filings,
            "missing_identity": missing, "corpus_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write changes. Without it, nothing is modified.")
    parser.add_argument("--truncate-for-reingest", action="store_true",
                        help="Delete every row so the corpus can be re-ingested "
                             "with identity. Requires the confirmation flag.")
    parser.add_argument("--i-understand-this-deletes", metavar="TABLE",
                        help=f"Must be exactly '{TABLE}' to permit truncation.")
    parser.add_argument("--json", dest="json_path",
                        help="Write the report as JSON to this path")
    args = parser.parse_args()

    report = {"applied": bool(args.apply), "actions": [], "warnings": []}

    with get_db_session() as session:
        columns = _existing_columns(session)
        if not columns:
            print(f"Table {TABLE} does not exist; nothing to migrate.")
            return 1

        report["before"] = _corpus_state(session, columns)
        print(f"{TABLE}: {report['before']['total_chunks']} chunks across "
              f"{report['before']['distinct_filings']} filings")
        print(f"  columns: {len(columns)}")

        # --- 1. add the identity columns -----------------------------------
        for name, sql_type in NEW_COLUMNS.items():
            if name in columns:
                print(f"  column {name}: present")
                continue
            if not args.apply:
                report["actions"].append(f"would add column {name}")
                print(f"  column {name}: MISSING (dry run)")
                continue
            session.execute(text(
                f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {name} {sql_type}"))
            report["actions"].append(f"added column {name}")
            print(f"  column {name}: added")

        columns = _existing_columns(session)

        # --- 2. optional truncation for re-ingestion ------------------------
        if args.truncate_for_reingest:
            if args.i_understand_this_deletes != TABLE:
                print(f"\nRefusing to truncate: pass "
                      f"--i-understand-this-deletes {TABLE}")
                return 1
            if not args.apply:
                report["actions"].append("would truncate for re-ingestion")
                print("\n  truncate: dry run, nothing deleted")
            else:
                session.execute(text(f"TRUNCATE TABLE {TABLE}"))
                report["actions"].append("truncated for re-ingestion")
                print("\n  truncate: done — now run "
                      "`python -m finvet.rag.ingest`")

        # --- 3. the unique index, only once identity is complete ------------
        state = _corpus_state(session, columns)
        missing = state["missing_identity"]

        if missing is None:
            report["warnings"].append(
                "identity columns absent; run again with --apply")
        elif missing > 0:
            message = (f"{missing} of {state['total_chunks']} rows have no "
                       f"identity. Part and item cannot be inferred from the "
                       f"section name — re-ingest from source filings.")
            report["warnings"].append(message)
            print(f"\n  {message}")
            print(f"  the unique {EVIDENCE_INDEX} index is NOT created while "
                  f"rows are unidentified")
        elif _index_exists(session, EVIDENCE_INDEX):
            print(f"\n  {EVIDENCE_INDEX}: present")
        elif not args.apply:
            report["actions"].append(f"would create {EVIDENCE_INDEX}")
            print(f"\n  {EVIDENCE_INDEX}: would create (dry run)")
        else:
            session.execute(text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {EVIDENCE_INDEX} "
                f"ON {TABLE} (evidence_id)"))
            report["actions"].append(f"created {EVIDENCE_INDEX}")
            print(f"\n  {EVIDENCE_INDEX}: created")

        report["after"] = _corpus_state(session, _existing_columns(session))

    # An empty table has no unidentified rows, which is not the same as having
    # identity. Reporting "complete" on a truncated corpus would say the
    # migration succeeded at the exact moment there is nothing left to search.
    after = report["after"]
    complete = (after["missing_identity"] == 0
                and after["total_chunks"] > 0
                and not report["warnings"])
    if after["total_chunks"] == 0:
        report["warnings"].append(
            "corpus is empty — run `python -m finvet.rag.ingest` before "
            "treating this migration as finished")
    report["identity_complete"] = complete

    print(f"\ncorpus sha256: {report['after']['corpus_sha256']}")
    print(f"identity complete: {complete}")
    if not args.apply:
        print("\n(dry run — nothing was written; re-run with --apply)")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json_path}")

    return 0 if (complete or not args.apply) else 2


if __name__ == "__main__":
    raise SystemExit(main())
