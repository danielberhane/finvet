#!/usr/bin/env python
"""Read-only operator report for the audit trail.

Lists audit records nothing else surfaces: events with no execution row,
reviews stuck in REVIEWING, and reviews awaiting reconciliation.

This script only reads. It has no --apply, and there is nothing to apply:
repairing an execution row from surviving events means inventing the parts the
events do not record, and a reconstructed guess stored beside genuine records
is indistinguishable from one. Retry a stuck finalization through
`POST /review/{request_id}/reconcile`, which uses the outcome the checkpoint
already holds rather than making one up.

Usage:
    .venv/bin/python scripts/report_audit_operations.py
    .venv/bin/python scripts/report_audit_operations.py --stale-after-minutes 30
    .venv/bin/python scripts/report_audit_operations.py --json /tmp/audit-ops.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finvet.audit.operator_report import (  # noqa: E402
    DEFAULT_SAMPLE_LIMIT,
    STALE_REVIEW_MINUTES,
    build_operator_report,
)


def _render(report) -> str:
    lines = [
        "FinVet audit operations report",
        f"  generated at        : {report.generated_at}",
        f"  stale after         : {report.stale_after_minutes} minutes",
        "",
        f"  orphan events       : {report.orphan_event_count}",
        f"  stuck reviews       : {report.stuck_review_count}",
        f"  awaiting reconcile  : {report.awaiting_reconciliation_count}",
    ]

    if report.error:
        lines += ["", f"  ERROR: {report.error}",
                  "  The store could not be read; this is not a clean result."]
        return "\n".join(lines)

    if report.orphan_event_request_ids:
        lines += ["", "  Events with no execution row:"]
        lines += [f"    {rid}" for rid in report.orphan_event_request_ids]

    if report.stuck_reviews:
        lines += ["", "  Reviews stuck in REVIEWING (no reconcile will claim "
                      "these; they need investigation):"]
        lines += [f"    {r['request_id']}  {r['timestamp']}"
                  for r in report.stuck_reviews]

    if report.awaiting_reconciliation:
        lines += ["", "  Awaiting reconciliation (retry with "
                      "POST /review/{id}/reconcile):"]
        lines += [f"    {r['request_id']}  {r['timestamp']}"
                  for r in report.awaiting_reconciliation]

    if report.is_clean:
        lines += ["", "  Nothing to act on."]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stale-after-minutes", type=int,
                        default=STALE_REVIEW_MINUTES,
                        help="How long a REVIEWING row may sit before it is "
                             f"reported as stuck (default: {STALE_REVIEW_MINUTES})")
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT,
                        help="Maximum identifiers listed per category")
    parser.add_argument("--json", dest="json_path",
                        help="Also write the report as JSON to this path")
    args = parser.parse_args()

    report = build_operator_report(
        stale_after_minutes=args.stale_after_minutes,
        sample_limit=args.limit,
    )
    print(_render(report))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report.as_dict(), indent=2))
        print(f"\n  JSON written to {args.json_path}")

    # Non-zero when the store could not be read, so a scheduled run that cannot
    # see the database does not look like a clean system.
    return 1 if report.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
