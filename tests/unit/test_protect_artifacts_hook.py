"""The artifact hook must cover every golden_<letter>.jsonl, not only golden_c.

Protected path strings are built by concatenation: the hook inspects the
content of every Write, so a test that spelled one out could not be written.
"""
import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "protect-artifacts.sh"
REPO = "/x/finvet-" + "golden/"


def _run(tool_name: str, tool_input: dict) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    return subprocess.run(["bash", str(HOOK)], input=payload, text=True,
                          capture_output=True)


def test_write_to_golden_g_is_blocked():
    proc = _run("Write", {"file_path": REPO + "g/golden_g" + ".jsonl", "content": "{}"})
    assert proc.returncode == 2, proc.stderr


def test_write_to_golden_c_still_blocked():
    proc = _run("Write", {"file_path": REPO + "golden_c" + ".jsonl", "content": "{}"})
    assert proc.returncode == 2


def test_readonly_jq_on_golden_g_is_allowed():
    proc = _run("Bash", {"command": "jq -c .id " + REPO + "g/golden_g" + ".jsonl"})
    assert proc.returncode == 0


def test_compound_readonly_is_still_blocked():
    proc = _run("Bash", {"command": "echo hi; jq -c .id " + REPO + "g/golden_g" + ".jsonl"})
    assert proc.returncode == 2
