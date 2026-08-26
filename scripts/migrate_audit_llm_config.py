#!/usr/bin/env python
"""Add `llm_config` to `audit_executions`, and say what it cannot answer.

A stored result that does not record which model produced it cannot be compared
against a result from another model. That comparison is the point: FinVet's
central claim is that a deterministic comparison overrules the LLM, and testing
that claim means varying the model and diffing the outcomes.

`llm_config` holds the parser, agent and verdict roles -- model, base_url,
temperature and structured_output_method for each. `base_url` is included
because a model name alone is ambiguous: the same name served by a hosted API
and by a local Ollama endpoint is not the same run. No API key is stored.

**Existing rows are left NULL, permanently.** The models behind a historical
run were never recorded, so there is nothing to backfill from. Stamping them
with today's configuration would assert something unknown -- and every such row
predates this column precisely because nobody was varying the model then. A
NULL that means "not recorded" is worth more than a value that means "guessed".

The column is nullable for the same reason, so `verify_execution_checksum` keeps
passing on schema-1 envelopes: verification recomputes from the stored envelope,
and those envelopes never held the key.

Dry run by default; nothing is written without `--apply`.

Usage:
    .venv/bin/python scripts/migrate_audit_llm_config.py
    .venv/bin/python scripts/migrate_audit_llm_config.py --apply
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from finvet.config.database import get_db_session  # noqa: E402

TABLE = "audit_executions"
COLUMN = "llm_config"


def column_exists(session) -> bool:
    return session.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :c"
    ), {"t": TABLE, "c": COLUMN}).first() is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the change; otherwise report only")
    args = parser.parse_args()

    with get_db_session() as session:
        present = column_exists(session)
        total = session.execute(
            text(f"SELECT count(*) FROM {TABLE}")).scalar_one()

        print(f"  {TABLE}: {total} rows")
        print(f"  {COLUMN}: {'present' if present else 'absent'}")

        if present:
            unattributed = session.execute(text(
                f"SELECT count(*) FROM {TABLE} WHERE {COLUMN} IS NULL"
            )).scalar_one()
            print(f"  rows with no recorded model: {unattributed}"
                  " (left NULL by design -- nothing to backfill from)")
            print("\n  nothing to do")
            return 0

        if not args.apply:
            print(f"\n  would add: {COLUMN} JSONB NULL")
            print(f"  {total} existing rows would stay NULL")
            print("\n  dry run -- pass --apply to write")
            return 0

        session.execute(text(
            f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} JSONB"))
        print(f"\n  added {COLUMN} JSONB NULL")
        print(f"  {total} existing rows left unattributed, permanently")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
