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
             "gold_parse": {"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",
                            "operator": "eq", "value": 391000000000, "period": "FY2024",
                            "reject_reason": None},
             "expected": {"verdict": "SUPPORTS"}, "actual": {"verdict": "SUPPORTS"}},
            {"id": 5, "claim": "a held-out claim that must never be published",
             "gold_parse": {"claim_type": "sec", "ticker": "XXXX", "metric": "revenue",
                            "operator": "eq", "value": 1, "period": "FY2024",
                            "reject_reason": None},
             "expected": {"verdict": "REFUTES"}, "actual": {"verdict": "REFUTES"}},
        ],
    }


class TestRedactRules:

    def test_burned_rows_keep_their_text_and_others_are_withheld(self):
        out = redact_run.redact(_artifact())
        by_id = {r["id"]: r["claim"] for r in out["results"]}
        assert by_id[1].startswith("Apple's total revenue")
        assert by_id[5] == redact_run.WITHHELD

    def test_gold_parse_is_withheld_except_for_burned_rows(self):
        out = redact_run.redact(_artifact())
        by_id = {r["id"]: r["gold_parse"] for r in out["results"]}
        assert by_id[1] == {"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",
                            "operator": "eq", "value": 391000000000, "period": "FY2024",
                            "reject_reason": None}
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


class TestTheParseCannotRebuildTheClaim:
    """A withheld claim is not withheld if its parse is published beside it.

    The recorded parse carried ticker, metric, value and period, and the row
    carried the filed figure and its period end -- enough to write the claim
    back out for most of the set. These rows drive the real `redact`.
    """

    def _artifact(self):
        parse = {"claim_type": "sec", "ticker": "RIVN", "metric": "net_income",
                 "operator": "eq", "value": 3630000000.0, "period": "fiscal 2025",
                 "reject_reason": None}
        actual = {"verdict": "REFUTES", "confidence": 0.85,
                  "retrieved_value": -3646000000.0,
                  "observation_tool": "get_income_statement",
                  "observation_period_end": "2025-12-31",
                  "tools_called": ["get_company_info", "get_income_statement"],
                  "parsed_claim": parse, "data_sources": ["xbrl"],
                  "escalated": False}
        return {
            "label": "u-ds-c9", "complete": True,
            "dataset": "/Users/someone/private-gold/golden_u.jsonl",
            "llm_config": {}, "llm_config_client": {},
            "results": [
                {"id": 1073, "claim": "hidden", "gold_parse": dict(parse),
                 "category": "sec/xbrl", "strength": "strict",
                 "expected": {"verdict": "SUPPORTS", "sources": ["xbrl"]},
                 "actual": json.loads(json.dumps(actual))},
                {"id": 1, "claim": "burned", "gold_parse": dict(parse),
                 "category": "sec/xbrl", "strength": "strict",
                 "expected": {"verdict": "SUPPORTS", "sources": ["xbrl"]},
                 "actual": json.loads(json.dumps(actual))},
            ],
        }

    def test_identifying_fields_leave_a_withheld_row(self):
        row = redact_run.redact(self._artifact())["results"][0]["actual"]
        assert row["parsed_claim"] == {"claim_type": "sec", "metric": "net_income"}
        assert row["retrieved_value"] == redact_run.WITHHELD
        assert row["observation_period_end"] == redact_run.WITHHELD
        text = json.dumps(row)
        for value in ("RIVN", "3630000000", "3646000000", "fiscal 2025", "2025-12-31"):
            assert value not in text

    def test_a_burned_row_keeps_its_parse(self):
        row = redact_run.redact(self._artifact())["results"][1]["actual"]
        assert row["parsed_claim"]["ticker"] == "RIVN"
        assert row["retrieved_value"] == -3646000000.0

    def test_the_measures_that_read_these_fields_are_unchanged(self):
        from finvet.eval.measures import grounding, trajectory
        from finvet.eval.measures.artifacts import Run

        def as_run(artifact):
            return Run(label=artifact["label"], started_utc="", model="m",
                       complete=True,
                       rows={r["id"]: r for r in artifact["results"]})

        before = self._artifact()
        after = redact_run.redact(before)
        for row_before, row_after in zip(before["results"], after["results"]):
            assert (trajectory._strategy_of(row_before)
                    == trajectory._strategy_of(row_after))
        assert grounding.measure([as_run(before)]) == grounding.measure([as_run(after)])
