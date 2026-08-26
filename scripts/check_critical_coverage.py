#!/usr/bin/env python
"""Per-file coverage floors for the modules a wrong answer travels through.

An overall percentage can sit comfortably above its gate while the code that
decides a verdict, records it, or hands it to a reviewer is barely exercised —
a large well-tested module and a small untested one average out fine. These six
files are where a defect becomes a wrong answer someone acts on, so each carries
its own floor.

A file missing from the coverage report is a failure, not a warning. The usual
cause is that it was renamed or excluded, which is exactly when its floor
silently stops being enforced.

Usage:
    .venv/bin/python -m pytest tests/unit -q --cov=src/finvet \\
        --cov-report=json:/tmp/finvet-coverage.json
    .venv/bin/python scripts/check_critical_coverage.py /tmp/finvet-coverage.json
"""

import argparse
import json
from pathlib import Path

CRITICAL = {
    # The two request paths: every verdict the system releases leaves through
    # one of them, and both must terminalize exactly once.
    "src/finvet/api/routes/verify.py": 85.0,
    "src/finvet/api/routes/review.py": 85.0,
    # The audit lifecycle. If this is wrong, nothing else can be checked.
    "src/finvet/audit/database.py": 85.0,
    # The deterministic comparison and the trust boundary around it.
    "src/finvet/agents/base.py": 85.0,
    # Retrieval: what the model is allowed to read, and with what provenance.
    "src/finvet/rag/service.py": 85.0,
    # The only place a claim is escalated to a person.
    "src/finvet/graph/nodes/output_guardrails.py": 85.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", nargs="?",
                        default="/tmp/finvet-coverage.json")
    args = parser.parse_args()

    path = Path(args.coverage_json)
    if not path.exists():
        print(f"coverage report not found: {path}")
        print("run pytest with --cov-report=json:<path> first")
        return 2

    report = json.loads(path.read_text())
    files = report.get("files", {})

    failures, rows = [], []
    for name, floor in sorted(CRITICAL.items()):
        entry = files.get(name)
        if entry is None:
            failures.append(f"{name}: absent from the coverage report")
            rows.append((name, None, floor))
            continue
        percent = entry["summary"]["percent_covered"]
        rows.append((name, percent, floor))
        if percent < floor:
            missing = entry["summary"]["missing_lines"]
            failures.append(
                f"{name}: {percent:.1f}% < {floor:.0f}% ({missing} lines uncovered)")

    width = max(len(n) for n in CRITICAL)
    print("critical-file coverage")
    for name, percent, floor in rows:
        shown = "  ABSENT" if percent is None else f"{percent:>6.1f}%"
        mark = "FAIL" if (percent is None or percent < floor) else "ok"
        print(f"  {name:<{width}} {shown}  (floor {floor:.0f}%)  {mark}")

    overall = report.get("totals", {}).get("percent_covered")
    if overall is not None:
        print(f"\n  overall: {overall:.2f}%")

    if failures:
        print("\ncritical coverage gate FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("\ncritical coverage gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
