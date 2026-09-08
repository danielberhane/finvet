"""DATASET_CARD.md promises ids 1, 68, 88 never count in a scored run after
`*-c1`, and `exclusions.burned_ids` records them -- but nothing consulted it:
a benchmark launched on 2026-09-05 was executing burned rows until killed.
The runner now drops them where rows are selected.

This drives the real entry point (the script, not a reimplementation of its
filter) with FINVET_API_URL pointed at nothing, so it prints its selection
line and dies before spending anything.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_golden.py"


def test_burned_ids_are_not_selected(tmp_path):
    rows = [{"id": i, "claim": f"c{i}", "category": "sec/xbrl",
             "strength": "strict"} for i in (1, 2, 68, 88, 90)]
    (tmp_path / "golden_c.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    # Settings is instantiated at import time and two fields have no default.
    # The subprocess gets dummies explicitly, and runs from tmp_path so the
    # repo's own .env cannot supply them: this test passed locally for five
    # days on that .env and failed on the first CI run, where there is none.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--label", "unit-test"],
        cwd=tmp_path,
        env={"FINVET_GOLDEN_DIR": str(tmp_path),
             "FINVET_API_URL": "http://127.0.0.1:9",  # nothing listens
             "TAVILY_API_KEY": "test-key",
             "POSTGRES_PASSWORD": "test-password",
             "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=60,
    )

    assert "5 rows, 2 selected, 3 burned ids skipped" in proc.stdout, proc.stderr
    assert proc.returncode != 0, "must refuse without a reachable API"
