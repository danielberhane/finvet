"""--dataset selects a second golden file in the same directory."""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_golden.py"


def _row(i):
    return {"id": i, "claim": f"claim {i} about revenue in fiscal 2024",
            "category": "sec/xbrl", "strength": "strict",
            "expected": {"verdict": "SUPPORTS", "limitation": None, "sources": ["xbrl"]},
            "ground_truth": "", "source": "",
            "gold_parse": {"claim_type": "sec", "ticker": "X", "metric": "revenue",
                           "operator": "eq", "value": 1.0, "period": "FY2024",
                           "reject_reason": None, "label_source": "derived_from_source"}}


def test_dataset_flag_selects_golden_g(tmp_path):
    (tmp_path / "golden_g.jsonl").write_text(
        "\n".join(json.dumps(_row(i)) for i in (1001, 1002, 1003)) + "\n")
    env = {"FINVET_GOLDEN_DIR": str(tmp_path), "FINVET_API_URL": "http://127.0.0.1:9",
           "TAVILY_API_KEY": "x", "POSTGRES_PASSWORD": "x", "PATH": os.environ["PATH"]}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--dataset", "golden_g.jsonl"],
                          cwd=tmp_path, env=env, capture_output=True, text=True)
    assert "golden_g.jsonl  (3 rows, 3 selected, 0 burned ids skipped)" in proc.stdout, proc.stderr
    assert proc.returncode != 0, "must refuse without a reachable API"
