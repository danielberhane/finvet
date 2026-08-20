"""Tests for the stage-06 parser prompt.

The prompt moves to agents/prompts/parser_system.txt (the agents' convention)
and is rendered at load time: the metric whitelist is injected from
config/metrics.py, never hand-copied, so the prompt and the validator cannot
drift — a copied list would eventually tell the model about metrics the
resolver rejects, surfacing as a mysterious residual rate.
"""

import json
import re
from pathlib import Path

import pytest

from finvet.config.metrics import METRIC_WHITELIST
from finvet.graph.nodes.claim_parser import PARSER_SYSTEM_PROMPT, _PROMPT_PATH


class TestPromptIsRenderedFromTheConstant:

    def test_prompt_lives_in_the_prompts_directory(self):
        assert _PROMPT_PATH.name == "parser_system.txt"
        assert _PROMPT_PATH.parent.name == "prompts"
        assert _PROMPT_PATH.exists()

    def test_every_whitelisted_metric_reaches_the_model(self):
        """All 74, from the constant — the drift-proofing property."""
        missing = [m for ct in ("sec", "market", "news")
                   for m in METRIC_WHITELIST[ct]
                   if m not in PARSER_SYSTEM_PROMPT]
        assert not missing, f"metrics absent from the prompt: {missing}"

    def test_no_unrendered_placeholder_remains(self):
        assert "__METRIC_WHITELIST__" not in PARSER_SYSTEM_PROMPT

    def test_whitelist_is_grouped_by_claim_type(self):
        for header in ('claim_type == "sec"', 'claim_type == "market"',
                       'claim_type == "news"'):
            assert header in PARSER_SYSTEM_PROMPT


class TestPromptSpeaksTheContract:

    def test_schema_orders_claim_type_before_metric(self):
        """Decoding is left-to-right: metric must condition on a claim_type
        the model has already committed to."""
        schema_pos = PARSER_SYSTEM_PROMPT.find('"claim_type"')
        metric_pos = PARSER_SYSTEM_PROMPT.find('"metric"')
        assert -1 < schema_pos < metric_pos

    def test_operator_is_the_field_name_not_comparison(self):
        assert '"operator"' in PARSER_SYSTEM_PROMPT
        assert '"comparison"' not in PARSER_SYSTEM_PROMPT

    def test_approx_and_range_are_documented(self):
        assert '"approx"' in PARSER_SYSTEM_PROMPT
        assert '"range"' in PARSER_SYSTEM_PROMPT
        assert "midpoint" in PARSER_SYSTEM_PROMPT

    def test_currency_is_gone(self):
        assert '"currency"' not in PARSER_SYSTEM_PROMPT

    def test_reject_vocabulary_is_open(self):
        for reason in ("future_prediction", "out_of_domain", "ambiguous_entity"):
            assert reason in PARSER_SYSTEM_PROMPT

    def test_segment_trap_is_a_prohibition_with_the_live_caught_example(self):
        """'Amazon's advertising revenue' -> revenue was the one fail-closed
        miss the stage-01 probe caught live; it is now a worked example."""
        assert "advertising revenue" in PARSER_SYSTEM_PROMPT
        assert "iPhone" in PARSER_SYSTEM_PROMPT

    def test_forecast_trap_is_stated(self):
        assert "estimated_revenue" in PARSER_SYSTEM_PROMPT

    def test_examples_emit_seven_keys_in_contract_order(self):
        """Every worked example must model the exact output shape."""
        order = ["claim_type", "ticker", "metric", "operator",
                 "value", "period", "reject_reason"]
        examples = re.findall(r'\{[^{}]*"claim_type"[^{}]*\}',
                              PARSER_SYSTEM_PROMPT)
        assert len(examples) >= 8, "prompt should keep a broad example set"
        for ex in examples:
            keys = re.findall(r'"(\w+)":', ex)
            assert keys == order, f"example breaks key order: {keys}"


class TestVerdictPromptKnowsApprox:
    """Stage-05 verification note: the verdict-extraction prompt named the
    base tolerances but not the approx widening. The override corrects the
    verdict deterministically either way; this just stops the LLM arguing."""

    def test_approx_widening_is_stated(self):
        import inspect
        from finvet.agents.base import BaseVerificationAgent
        src = inspect.getsource(BaseVerificationAgent._extract_verdict)
        assert "approx" in src


GOLD_DIR = Path.home() / "Projects/Active/claim_parser_fine-tuned/data/clean"


@pytest.mark.skipif(not GOLD_DIR.exists(), reason="parser gold not present")
class TestPromptExamplesAreNotEvalRows:
    """The evaluation-data policy (internal), enforced in code. The stage-06 rewrite lifted four
    worked examples verbatim from eval sets — caught by audit, worth ~0.3%
    inflated val exact-match. Prompt examples come from train.jsonl or are
    invented; anything the prompt was shaped on cannot measure the prompt."""

    def test_no_worked_example_appears_in_any_eval_split(self):
        def norm(t):
            return re.sub(r"[^a-z0-9]", "", t.lower())
        examples = re.findall(r'Input: "(.*?)"', PARSER_SYSTEM_PROMPT)
        assert len(examples) >= 8
        for split in ("val.jsonl", "test.jsonl",
                      "heldout_real_sourced.jsonl", "heldout_real.jsonl"):
            rows = {norm(json.loads(line)["input"])
                    for line in open(GOLD_DIR / split)}
            leaked = [e for e in examples if norm(e) in rows]
            assert not leaked, f"prompt examples found in {split}: {leaked}"


class TestBurnedRowsRegistry:

    def test_registry_names_the_known_incident_rows(self):
        from finvet.eval.exclusions import burned_ids
        assert burned_ids("test.jsonl") == {1, 11, 18}
        assert burned_ids("val.jsonl") == {257}
        assert burned_ids("heldout_real_sourced.jsonl") == {12}

    def test_unknown_split_burns_nothing(self):
        from finvet.eval.exclusions import burned_ids
        assert burned_ids("train.jsonl") == frozenset()
