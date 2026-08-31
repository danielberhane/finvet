#!/usr/bin/env python
"""Add `tools_called` and `parsed_claim` to artifacts that predate them.

A one-time migration. `scripts/run_golden.py` now records the tool names on
every row, but the four runs already on disk predate that, and re-running them
would cost real API calls to recover data the audit trail already holds.

Each row carries a `request_id`; `audit_events` carries one `tool_called` event
per call. Joining the two recovers the trajectory for runs already paid for.

Rows blocked by an input guardrail have no `request_id` -- the request was
refused before an execution existed -- so they get an empty list, which is
accurate: no tool could have been called.

Dry run by default; nothing is written without `--apply`.

Usage:
    .venv/bin/python scripts/backfill_tools.py --dir ~/Projects/Active/finvet-golden
    .venv/bin/python scripts/backfill_tools.py --dir ... --apply
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finvet.config.database import get_db_session  # noqa: E402
from sqlalchemy import text  # noqa: E402


def parses_for(session, request_ids):
    """{request_id: parsed fields} from the claim_parsed audit event.

    Trajectory expectations key on the routing strategy, which follows
    claim_type and metric. The dataset category is not a substitute: two `a2a`
    rows parse as sec claims about what a filing says and route to filing_rag,
    so scoring them as news claims marked six correct runs as failures.
    """
    if not request_ids:
        return {}
    rows = session.execute(text(
        "SELECT request_id, data->'fields' AS fields FROM audit_events "
        "WHERE event_type = 'claim_parsed' AND request_id = ANY(:ids)"
    ), {"ids": list(request_ids)}).fetchall()
    return {rid: fields for rid, fields in rows if fields}


def tools_for(session, request_ids):
    """{request_id: [tool names]} for every call the audit trail recorded."""
    if not request_ids:
        return {}
    rows = session.execute(text(
        "SELECT request_id, data->>'tool' AS tool FROM audit_events "
        "WHERE event_type = 'tool_called' AND request_id = ANY(:ids) "
        "ORDER BY timestamp, event_id"
    ), {"ids": list(request_ids)}).fetchall()

    found = {}
    for request_id, tool in rows:
        if tool:
            found.setdefault(request_id, []).append(tool)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.environ.get("FINVET_GOLDEN_DIR"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.dir:
        print("Set FINVET_GOLDEN_DIR or pass --dir.", file=sys.stderr)
        return 2

    paths = sorted(Path(args.dir).expanduser().glob("run-*.json"))
    if not paths:
        print(f"No artifacts in {args.dir}", file=sys.stderr)
        return 2

    with get_db_session() as session:
        for path in paths:
            data = json.loads(path.read_text())
            results = data.get("results", [])
            if all("parsed_claim" in (r.get("actual") or {}) for r in results):
                print(f"  {path.name}: already backfilled, skipping")
                continue

            ids = [r["request_id"] for r in results if r.get("request_id")]
            by_request = tools_for(session, ids)
            parses = parses_for(session, ids)

            calls = parsed = 0
            for row in results:
                actual = row.get("actual")
                if actual is None:
                    continue
                tools = by_request.get(row.get("request_id"), [])
                actual["tools_called"] = tools
                calls += len(tools)
                parse = parses.get(row.get("request_id"))
                actual["parsed_claim"] = parse
                parsed += bool(parse)

            print(f"  {path.name}: {len(results)} rows, {calls} tool calls, "
                  f"{parsed} parses recovered"
                  + ("" if args.apply else "   (dry run)"))
            if args.apply:
                path.write_text(json.dumps(data, indent=1))

    if not args.apply:
        print("\n  dry run -- pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
