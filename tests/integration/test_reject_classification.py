"""Reject classification against the claim-parser gold labels.

Runs input_guardrails and claim_parser for real, so it needs DeepSeek and — while
ENABLE_LLAMA_GUARD is set — a reachable Ollama. Ground truth is read by path from
the sibling claim-parser project; that set is contamination-sensitive and is
never copied here, so the tests skip when it is absent.

Opt in with:  pytest tests/integration/test_reject_classification.py -m integration
"""

import pytest

from finvet.eval.reject_classification import (
    DEFAULT_GOLD,
    OUTCOME_ERROR,
    OUTCOME_PENDING,
    load_cases,
    run_case,
    summarize,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DEFAULT_GOLD.exists(),
        reason=f"real-sourced gold set not present at {DEFAULT_GOLD}",
    ),
]

# Rows scored per run. Each costs a parser LLM call plus a Llama Guard pass, so
# keep it small; use the CLI for a full sweep.
SAMPLE_SIZE = 12

# Ratchet — raise as the reject path improves, never lower.
#
#   0.33  2026-08-20  baseline on this 12-row slice (4/12). The full 120-row
#                     real-sourced set scores 41.2% (49/119); this slice runs
#                     colder. Comparable figures for the fine-tuned adapter are
#                     49.2% on the same set and 99.2% on synthetic — FinVet's
#                     DeepSeek parser scores 41.2% and 87.9% respectively.
#                     Both mechanisms count: guard-blocked and parser-rejected.
MIN_REJECT_RECALL = 0.33

# A verifiable claim stopped by mistake is worse than a missed reject: the user
# is refused an answer FinVet could have given. Nothing in the sample should
# false-reject.
MAX_FALSE_REJECTS = 0


@pytest.fixture(scope="module")
def scored_reject_cases():
    cases = load_cases(DEFAULT_GOLD, limit=SAMPLE_SIZE, claim_type="reject")
    for case in cases:
        run_case(case)
    return cases


@pytest.fixture(scope="module")
def scored_verifiable_cases():
    cases = load_cases(DEFAULT_GOLD, limit=SAMPLE_SIZE, claim_type="sec")
    for case in cases:
        run_case(case)
    return cases


def test_gold_set_yields_reject_cases(scored_reject_cases):
    assert scored_reject_cases, "no reject rows loaded from the gold set"


def test_every_case_reached_a_terminal_outcome(scored_reject_cases):
    """A row left pending means the harness silently skipped it."""
    pending = [c.row_id for c in scored_reject_cases if c.outcome == OUTCOME_PENDING]
    assert not pending, f"rows never scored: {pending}"


def test_reject_recall_does_not_regress(scored_reject_cases):
    summary = summarize(scored_reject_cases)
    assert summary["reject_recall"] >= MIN_REJECT_RECALL, (
        f"reject recall {summary['reject_recall']:.1%} fell below the "
        f"{MIN_REJECT_RECALL:.1%} ratchet. Missed: "
        + "; ".join(
            f"row {c.row_id} ({c.gold_reject_reason}) parsed as {c.parsed_claim_type}"
            for c in scored_reject_cases if c.outcome == "missed"
        )
    )


def test_verifiable_claims_are_not_rejected(scored_verifiable_cases):
    """False rejects refuse an answer FinVet could have given."""
    summary = summarize(scored_verifiable_cases)
    assert summary["false_reject"] <= MAX_FALSE_REJECTS, (
        "verifiable claims were rejected: "
        + "; ".join(
            f"row {c.row_id}: {c.violation_type or c.parsed_reject_reason}"
            for c in scored_verifiable_cases if c.outcome == "false_reject"
        )
    )


def test_guard_and_parser_rejections_are_attributed_separately(scored_reject_cases):
    """Both mechanisms are correct outcomes, but they must stay distinguishable —
    collapsing them hides what breaks if one layer is disabled."""
    summary = summarize(scored_reject_cases)
    assert (summary["blocked_by_guard"] + summary["rejected_by_parser"]
            == summary["rejected"])
    for case in scored_reject_cases:
        if case.outcome == "blocked_by_guard":
            assert case.guard_layer in ("regex", "llama_guard", "unknown")
            assert case.violation_type


def test_errors_do_not_silently_inflate_recall(scored_reject_cases):
    """Errored rows are excluded from the denominator, so they must be counted."""
    summary = summarize(scored_reject_cases)
    errored = sum(1 for c in scored_reject_cases if c.outcome == OUTCOME_ERROR)
    assert summary["errors"] == errored
    assert summary["gold_reject"] == len(scored_reject_cases) - errored
