"""XBRL retrieval accuracy against SEC primary-source values.

Requires the SEC EDGAR MCP server and network access, and reads ground truth by
path from the external evaluation dataset. That gold set is contamination-
sensitive and is never copied into this repo — the tests skip when it is absent.

Opt in with:  pytest tests/integration/test_xbrl_retrieval.py -m integration
"""

import pytest

from finvet.eval.xbrl_retrieval import (
    DEFAULT_GOLD,
    STATUS_PENDING,
    STATUS_UNSUPPORTED_CONCEPT,
    SECEdgarClient,
    load_cases,
    run_case,
    summarize,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        DEFAULT_GOLD is None or not DEFAULT_GOLD.exists(),
        reason="eval dataset not present (set FINVET_EVAL_DATA_DIR)",
    ),
]

# Cases scored per run. Each is an MCP round trip (~5-20s), so keep it small
# enough to stay usable; raise it for a full sweep via the CLI instead.
SAMPLE_SIZE = 12

# Ratchet — raise as retrieval improves, never lower.
#
#   0.00  2026-08-20  baseline. get_financials sent no period to the MCP server,
#                     which returns one arbitrary XBRL context per concept, so a
#                     filing's comparative years came back interchangeably. 0/7.
#   0.40  2026-08-20  period_end threaded through and facts selected from
#                     companyconcept by end date AND duration. 5/7 on an 8-row
#                     sample; 2/5 here because this slice holds both rows whose
#                     concept has no companyconcept fact at all.
#   0.70  2026-08-20  MCP timeout raised 15s -> 60s (8070648) removed all 23
#                     systematic transport failures; the full corpus then
#                     measured 90.6% twice (117 scored cases, then 180 after
#                     the frames-API gold fill — via the CLI sweep). This test
#                     samples only 12 rows, and at a true rate of 0.906 a 0.85
#                     floor flakes roughly one run in nine on binomial variance
#                     alone; 0.70 cannot be tripped by variance, only by real
#                     regression. The sharper safety property is asserted
#                     separately and deterministically below: a wrong value is
#                     NEVER carried with consolidated=True.
#
# Transport failures count against the rate: a case that times out is a case
# where FinVet did not produce the number, whatever the cause.
MIN_PASS_RATE = 0.70


@pytest.fixture(scope="module")
def scored_cases():
    # Stride-sample across the whole corpus rather than taking the first N:
    # the head of the gold file concentrates both rows whose concept has no
    # companyconcept fact at all, so a first-N slice scored 0.67 while the
    # full corpus measured 90.6%. A stride is just as deterministic but
    # representative.
    all_cases = load_cases(DEFAULT_GOLD)
    stride = max(1, len(all_cases) // SAMPLE_SIZE)
    cases = all_cases[::stride][:SAMPLE_SIZE]
    client = SECEdgarClient()
    try:
        for case in cases:
            run_case(case, client)
    finally:
        client.close()
    return cases


def test_gold_set_yields_scoreable_cases(scored_cases):
    assert scored_cases, "no cases loaded from the gold set"


def test_every_case_reached_a_terminal_status(scored_cases):
    """A case left PENDING means the harness silently skipped it."""
    pending = [c.row_id for c in scored_cases if c.status == STATUS_PENDING]
    assert not pending, f"cases never scored: {pending}"


def test_pass_rate_does_not_regress(scored_cases):
    summary = summarize(scored_cases)
    assert summary["pass_rate"] >= MIN_PASS_RATE, (
        f"XBRL retrieval pass rate {summary['pass_rate']:.1%} fell below the "
        f"{MIN_PASS_RATE:.1%} ratchet. Failures: "
        + "; ".join(
            f"row {c.row_id} {c.concept}: filed {c.expected_value:,.0f} "
            f"({c.expected_period_end}) vs got {c.finvet_value:,.0f} "
            f"({c.returned_period_end})"
            for c in scored_cases
            if c.status not in ("PASS", STATUS_UNSUPPORTED_CONCEPT) and c.finvet_value
        )
    )


def test_unsupported_concepts_are_reported_not_scored(scored_cases):
    """Concepts absent from CONCEPTS_BY_TYPE are a coverage gap. They must not
    be counted as retrieval failures, or the pass rate understates reality."""
    summary = summarize(scored_cases)
    unsupported = sum(1 for c in scored_cases if c.status == STATUS_UNSUPPORTED_CONCEPT)
    assert summary["unsupported_concept"] == unsupported
    assert summary["scored"] == len(scored_cases) - unsupported


def test_wrong_values_are_never_silently_trusted(scored_cases):
    """The invariant that matters more than the rate. Across every sweep since
    the period fix (500+ scored cases) no wrong value has ever been returned
    with consolidated=True — failures carry consolidated=False, telling the
    caller the figure could not be verified for the requested period. A single
    silently-trusted wrong value is a worse regression than any rate drop."""
    silently_wrong = [
        c for c in scored_cases
        if c.status == "FAIL" and c.consolidated_flag is True
    ]
    assert not silently_wrong, (
        "wrong values carried as verified: "
        + "; ".join(
            f"row {c.row_id} {c.concept}: got {c.finvet_value:,.0f} "
            f"vs filed {c.expected_value:,.0f}"
            for c in silently_wrong
        )
    )
