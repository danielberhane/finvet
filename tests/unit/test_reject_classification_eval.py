"""Tests for the reject-classification evaluation harness.

The harness answers what FinVet actually does with a claim that should not be
verified at all. Two independent mechanisms can reject one: the input guardrail
raises before the parser ever runs, or the parser classifies it as "reject".
Both are correct outcomes, and collapsing them into a single pass rate would
hide which layer is doing the work.

These tests cover the pure logic — outcome classification, guard-layer
attribution, case loading, aggregation — so a harness failure means FinVet's
reject path is wrong rather than the scorer.
"""

import json

import pytest

from finvet.eval.reject_classification import (
    OUTCOME_ACCEPTED,
    OUTCOME_BLOCKED_BY_GUARD,
    OUTCOME_ERROR,
    OUTCOME_FALSE_REJECT,
    OUTCOME_MISSED,
    OUTCOME_REJECTED_BY_PARSER,
    build_case,
    classify_outcome,
    guard_layer_for,
    load_cases,
    summarize,
)


class TestClassifyOutcome:
    """Gold label crossed with what the pipeline did."""

    def test_gold_reject_blocked_by_guard(self):
        assert classify_outcome("reject", guard_blocked=True, parsed_claim_type=None) == (
            OUTCOME_BLOCKED_BY_GUARD
        )

    def test_gold_reject_rejected_by_parser(self):
        assert classify_outcome("reject", False, "reject") == OUTCOME_REJECTED_BY_PARSER

    @pytest.mark.parametrize("parsed", ["sec", "market", "news"])
    def test_gold_reject_routed_to_an_agent_is_missed(self, parsed):
        """The failure case: an unverifiable claim gets verified anyway."""
        assert classify_outcome("reject", False, parsed) == OUTCOME_MISSED

    def test_gold_non_reject_blocked_by_guard_is_a_false_reject(self):
        assert classify_outcome("sec", guard_blocked=True, parsed_claim_type=None) == (
            OUTCOME_FALSE_REJECT
        )

    def test_gold_non_reject_rejected_by_parser_is_a_false_reject(self):
        assert classify_outcome("market", False, "reject") == OUTCOME_FALSE_REJECT

    @pytest.mark.parametrize("gold", ["sec", "market", "news"])
    def test_gold_non_reject_routed_to_an_agent_is_accepted(self, gold):
        assert classify_outcome(gold, False, gold) == OUTCOME_ACCEPTED

    def test_misrouted_non_reject_still_counts_as_accepted(self):
        """Routing sec->market is a routing error, not a reject error. This
        harness scores the reject decision only."""
        assert classify_outcome("sec", False, "market") == OUTCOME_ACCEPTED


class TestGuardLayerFor:
    """The composite guard reports provider='composite' on failure, so the
    violation type is what identifies which layer actually fired."""

    @pytest.mark.parametrize("violation", [
        "INJECTION_DETECTED", "PII_DETECTED", "CLAIM_TOO_SHORT",
        "CLAIM_TOO_LONG", "UNSUPPORTED_LANGUAGE",
    ])
    def test_regex_violations(self, violation):
        assert guard_layer_for(violation) == "regex"

    def test_llama_guard_violation(self):
        assert guard_layer_for("LLAMA_GUARD_UNSAFE") == "llama_guard"

    def test_unrecognised_violation_is_not_silently_attributed(self):
        assert guard_layer_for("SOMETHING_NEW") == "unknown"

    def test_missing_violation_type(self):
        assert guard_layer_for(None) == "unknown"


class TestBuildCase:

    def _row(self, claim_type="reject", reason="future_prediction", **prov):
        return {
            "id": 2,
            "input": "Amazon's net income is projected to grow 30% in 2025.",
            "gold": {"claim_type": claim_type, "ticker": None, "metric": None,
                     "operator": None, "value": None, "period": None,
                     "reject_reason": reason},
            "provenance": prov or {"source_type": "reject_authored"},
        }

    def test_carries_the_gold_label(self):
        case = build_case(self._row())
        assert case.row_id == 2
        assert case.gold_claim_type == "reject"
        assert case.gold_reject_reason == "future_prediction"
        assert case.source_type == "reject_authored"

    def test_carries_the_claim_text(self):
        assert build_case(self._row()).claim.startswith("Amazon's net income")

    def test_non_reject_row_has_no_reject_reason(self):
        case = build_case(self._row(claim_type="sec", reason=None))
        assert case.gold_claim_type == "sec"
        assert case.gold_reject_reason is None

    def test_row_without_provenance_still_builds(self):
        """test.jsonl carries no provenance block at all."""
        row = {"id": 9, "input": "x", "gold": {"claim_type": "reject",
                                               "reject_reason": "question"}}
        case = build_case(row)
        assert case.source_type is None
        assert case.gold_reject_reason == "question"


class TestLoadCases:

    def _write(self, tmp_path, rows):
        p = tmp_path / "gold.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def _row(self, rid, claim_type="reject"):
        return {"id": rid, "input": f"claim {rid}",
                "gold": {"claim_type": claim_type, "reject_reason": "question"
                         if claim_type == "reject" else None}}

    def test_loads_every_row_by_default(self, tmp_path):
        """Non-reject rows are scored too, or false rejects stay invisible."""
        rows = [self._row(1), self._row(2, "sec"), self._row(3, "market")]
        assert len(load_cases(self._write(tmp_path, rows))) == 3

    def test_claim_type_filter_selects(self, tmp_path):
        rows = [self._row(1), self._row(2, "sec"), self._row(3)]
        got = load_cases(self._write(tmp_path, rows), claim_type="reject")
        assert [c.row_id for c in got] == [1, 3]

    def test_limit_truncates(self, tmp_path):
        rows = [self._row(i) for i in range(1, 6)]
        assert len(load_cases(self._write(tmp_path, rows), limit=2)) == 2

    def test_selection_is_deterministic(self, tmp_path):
        rows = [self._row(i) for i in range(1, 20)]
        path = self._write(tmp_path, rows)
        assert [c.row_id for c in load_cases(path, limit=5)] == [
            c.row_id for c in load_cases(path, limit=5)
        ]

    def test_blank_lines_are_skipped(self, tmp_path):
        p = tmp_path / "gold.jsonl"
        p.write_text(json.dumps(self._row(1)) + "\n\n" + json.dumps(self._row(2)))
        assert len(load_cases(p)) == 2


class TestSummarize:

    def _case(self, gold, outcome, layer=None, reason=None):
        case = build_case({"id": 1, "input": "x",
                           "gold": {"claim_type": gold, "reject_reason": reason}})
        case.outcome = outcome
        case.guard_layer = layer
        return case

    def test_reject_recall_counts_both_mechanisms(self):
        """Guard-blocked and parser-rejected are both correct rejections."""
        cases = [
            self._case("reject", OUTCOME_BLOCKED_BY_GUARD, "regex"),
            self._case("reject", OUTCOME_REJECTED_BY_PARSER),
            self._case("reject", OUTCOME_MISSED),
            self._case("reject", OUTCOME_MISSED),
        ]
        s = summarize(cases)
        assert s["gold_reject"] == 4
        assert s["rejected"] == 2
        assert s["missed"] == 2
        assert s["reject_recall"] == 0.5

    def test_reject_precision_counts_false_rejects(self):
        cases = [
            self._case("reject", OUTCOME_REJECTED_BY_PARSER),
            self._case("reject", OUTCOME_REJECTED_BY_PARSER),
            self._case("sec", OUTCOME_FALSE_REJECT),
            self._case("market", OUTCOME_ACCEPTED),
        ]
        s = summarize(cases)
        assert s["false_reject"] == 1
        assert s["reject_precision"] == pytest.approx(2 / 3)

    def test_guard_and_parser_split_is_reported_separately(self):
        cases = [
            self._case("reject", OUTCOME_BLOCKED_BY_GUARD, "regex"),
            self._case("reject", OUTCOME_BLOCKED_BY_GUARD, "llama_guard"),
            self._case("reject", OUTCOME_REJECTED_BY_PARSER),
        ]
        s = summarize(cases)
        assert s["blocked_by_guard"] == 2
        assert s["rejected_by_parser"] == 1
        assert s["by_guard_layer"] == {"regex": 1, "llama_guard": 1}

    def test_errors_are_excluded_from_the_rates(self):
        """A transport failure is not evidence about the reject decision."""
        cases = [
            self._case("reject", OUTCOME_REJECTED_BY_PARSER),
            self._case("reject", OUTCOME_ERROR),
        ]
        s = summarize(cases)
        assert s["errors"] == 1
        assert s["gold_reject"] == 1
        assert s["reject_recall"] == 1.0

    def test_breakdown_by_reject_reason(self):
        cases = [
            self._case("reject", OUTCOME_REJECTED_BY_PARSER, reason="question"),
            self._case("reject", OUTCOME_MISSED, reason="question"),
            self._case("reject", OUTCOME_MISSED, reason="future_prediction"),
        ]
        by_reason = summarize(cases)["by_reject_reason"]
        assert by_reason["question"] == {"total": 2, "rejected": 1}
        assert by_reason["future_prediction"] == {"total": 1, "rejected": 0}

    def test_empty_input_does_not_divide_by_zero(self):
        s = summarize([])
        assert s["reject_recall"] == 0.0
        assert s["reject_precision"] == 0.0

    def test_no_gold_rejects_yields_zero_recall_not_an_error(self):
        s = summarize([self._case("sec", OUTCOME_ACCEPTED)])
        assert s["gold_reject"] == 0
        assert s["reject_recall"] == 0.0
