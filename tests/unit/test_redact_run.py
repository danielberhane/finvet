"""The published-artifact redaction is a producer; these tests drive it.

The hand-made redactions kept the absolute dataset path and the gateway
address, and both reached git history. Every rule here is one the hand
missed or nearly missed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "redact_run.py"
_spec = importlib.util.spec_from_file_location("redact_run", _SCRIPT)
redact_run = importlib.util.module_from_spec(_spec)
sys.modules["redact_run"] = redact_run
_spec.loader.exec_module(redact_run)


def _artifact() -> dict:
    gateway = {"model": "MiniMax-M2.7", "base_url": "http://10.20.30.40/v1",
               "temperature": 0.0, "structured_output_method": "json_mode"}
    public = {"model": "deepseek-chat", "base_url": "https://api.deepseek.com",
              "temperature": 0.0, "structured_output_method": "json_mode"}
    return {
        "label": "minimax-c9",
        "complete": True,
        "api": "http://127.0.0.1:8000",
        "dataset": "/Users/someone/private-gold/golden_c.jsonl",
        "llm_config": {"parser": dict(gateway), "agent": dict(gateway),
                       "verdict": dict(public)},
        "llm_config_client": {"parser": dict(gateway)},
        "results": [
            {"id": 1, "claim": "Apple's total revenue was $391 billion in fiscal year 2024",
             "expected": {"verdict": "SUPPORTS"}, "actual": {"verdict": "SUPPORTS"}},
            {"id": 5, "claim": "a held-out claim that must never be published",
             "expected": {"verdict": "REFUTES"}, "actual": {"verdict": "REFUTES"}},
        ],
    }


class TestRedactRules:

    def test_burned_rows_keep_their_text_and_others_are_withheld(self):
        out = redact_run.redact(_artifact())
        by_id = {r["id"]: r["claim"] for r in out["results"]}
        assert by_id[1].startswith("Apple's total revenue")
        assert by_id[5] == redact_run.WITHHELD

    def test_labels_and_behaviour_survive(self):
        out = redact_run.redact(_artifact())
        assert out["results"][1]["expected"] == {"verdict": "REFUTES"}
        assert out["results"][1]["actual"] == {"verdict": "REFUTES"}
        assert out["llm_config"]["parser"]["model"] == "MiniMax-M2.7"

    def test_dataset_path_is_reduced_to_basename(self):
        assert redact_run.redact(_artifact())["dataset"] == "golden_c.jsonl"

    def test_private_gateway_is_replaced_and_public_host_kept(self):
        out = redact_run.redact(_artifact())
        assert out["llm_config"]["parser"]["base_url"] == redact_run.GATEWAY_PLACEHOLDER
        assert out["llm_config_client"]["parser"]["base_url"] == redact_run.GATEWAY_PLACEHOLDER
        assert out["llm_config"]["verdict"]["base_url"] == "https://api.deepseek.com"

    def test_input_is_not_mutated(self):
        src = _artifact()
        redact_run.redact(src)
        assert src["results"][1]["claim"].startswith("a held-out")

    def test_nothing_private_remains_in_the_serialised_text(self):
        text = json.dumps(redact_run.redact(_artifact()))
        assert redact_run.leaks(text) == []
        assert "/Users/" not in text and "10.20.30.40" not in text


class TestWriter:

    def test_writes_next_to_nothing_else_and_refuses_to_overwrite(self, tmp_path):
        src_dir = tmp_path / "gold"
        src_dir.mkdir()
        src = src_dir / "run-20260901T000000Z-minimax-c9.json"
        src.write_text(json.dumps(_artifact()))
        out_dir = tmp_path / "public"
        out_dir.mkdir()

        dest = redact_run.write_redacted(src, out_dir)
        assert dest.name == "run-20260901T000000Z-minimax-c9-redacted.json"
        assert redact_run.leaks(dest.read_text()) == []

        with pytest.raises(SystemExit, match="exists"):
            redact_run.write_redacted(src, out_dir)
        redact_run.write_redacted(src, out_dir, force=True)

    def test_refuses_to_write_into_the_source_directory(self, tmp_path):
        src = tmp_path / "run-20260901T000000Z-minimax-c9.json"
        src.write_text(json.dumps(_artifact()))
        with pytest.raises(SystemExit, match="source directory"):
            redact_run.write_redacted(src, tmp_path)

    def test_refuses_to_publish_a_leak_the_rules_do_not_cover(self, tmp_path):
        artifact = _artifact()
        artifact["results"][1]["actual"]["note"] = "seen at /Users/someone/x"
        src_dir = tmp_path / "gold"
        src_dir.mkdir()
        src = src_dir / "run-20260901T000000Z-minimax-c9.json"
        src.write_text(json.dumps(artifact))
        out_dir = tmp_path / "public"
        out_dir.mkdir()
        with pytest.raises(SystemExit, match="still contains"):
            redact_run.write_redacted(src, out_dir)
        assert not list(out_dir.iterdir())
