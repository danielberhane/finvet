"""Every figure the README publishes is recomputed here from the artifacts.

`test_docs_match_the_code.py` guards the documented *vocabulary* -- status
names, config defaults, version ranges -- and it has earned its place twice
over. The numbers were the category it did not cover, and they are the ones a
reader is most likely to act on and least able to check.

They drift the same way everything else does. A review of this repository found
the reliability docstring quoting pass^4 = 0.975 over 79 rows, the calibration
docstring quoting an ECE and a bin table that no longer reproduced, and a
benchmark row reporting n=93 against 94 scored rows with no explanation. None
of it was caught by anything; all of it was found by a person reading closely,
which does not scale and does not run on every push.

So this reads the README, pulls the published figures out of it, recomputes
each one from the redacted artifacts in `docs/eval/`, and fails when they
disagree. The README's own claim is that "every figure in the table recomputes
from them" -- this is that sentence, executed.

**Two figures are deliberately not checked**, because they cannot be: XBRL
retrieval at 198/199 and the 70-case filing-text set are measured against gold
sets held privately with the claim set. The README says so where it states
them. A test that silently skipped them would be worse than their absence, so
`test_the_unverifiable_figures_are_marked_as_such` asserts the disclosure
instead.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from finvet.eval.measures import (  # noqa: E402
    artifacts as A, calibration, grounding, reliability, risk, routing,
    trajectory,
)

README = Path("README.md")
EVAL_DIR = Path("docs/eval")


def _runs(label):
    paths = [p for p in A.find_runs(EVAL_DIR) if label in p.name]
    if not paths:
        pytest.skip(f"no published artifact labelled {label!r}")
    return A.load_runs(paths)


def _table_row(prefix):
    """The two published cells of the results row starting with `prefix`.

    Matched on the row label rather than a line number so reordering the table
    -- which happened once already, when the layers were numbered -- does not
    silently stop checking a row.
    """
    for line in README.read_text().splitlines():
        if line.startswith(f"| {prefix}"):
            cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
            return cells[1], cells[2]
    pytest.fail(f"no results row starting {prefix!r}; did the table change?")


def _pct(cell):
    return float(cell.rstrip("%"))


def _outcome_excluding_live_market(runs):
    """The headline figure: scored rows, minus the live-market category.

    Market quotes move between runs, so a vendor outage during one run would
    otherwise read as a model difference. The exclusion is stated in the row
    label itself, and this is the arithmetic behind it.
    """
    rows = runs[0].rows
    scored = {i: r for i, r in rows.items() if A.is_scored(r)}
    live = {i for i, r in rows.items()
            if (r.get("category") or "").startswith("market/")}
    kept = {i: r for i, r in scored.items() if i not in live}
    return 100 * sum(A.is_correct(r) for r in kept.values()) / len(kept)


LAYERS = [
    ("1. Outcome", lambda runs: _outcome_excluding_live_market(runs), _pct, 0.1),
    ("2. Tool trajectory",
     lambda runs: 100 * trajectory.measure(runs).required_met
     / trajectory.measure(runs).scored, _pct, 0.1),
    ("3. Grounding", lambda runs: 100 * grounding.measure(runs).rate, _pct, 0.1),
    ("4. Calibration", lambda runs: calibration.measure(runs).decisive_ece,
     float, 0.001),
    ("5. Asymmetric risk, confidently-wrong",
     lambda runs: float(len(risk.measure(runs).dangerous)), float, 0.0),
    ("5. Asymmetric risk, declined",
     lambda runs: 100 * risk.measure(runs).decline_rate, _pct, 0.1),
    ("7. Reachability", lambda runs: 100 * routing.measure(runs).rate, _pct, 0.1),
]


class TestTheResultsTableRecomputes:
    """One case per layer per model, so a failure names the cell."""

    @pytest.mark.parametrize("prefix,compute,parse,tolerance", LAYERS)
    @pytest.mark.parametrize("column,label", [(0, "deepseek-c1"),
                                              (1, "minimax-c1")])
    def test_the_published_cell_matches_the_artifact(
            self, prefix, compute, parse, tolerance, column, label):
        published = parse(_table_row(prefix)[column])
        actual = compute(_runs(label))

        assert abs(published - actual) <= tolerance, (
            f"README row {prefix!r} column {label} publishes {published}, "
            f"the artifact gives {actual:.4f}")


class TestTheStabilityFiguresRecompute:
    """pass@1 and pass^4, quoted in prose rather than the table."""

    def _reliability(self):
        return reliability.measure(_runs("deepseek"))

    def test_pass_at_1_matches(self):
        published = float(re.search(r"pass@1 ([\d.]+)%",
                                    README.read_text()).group(1))
        assert abs(published - 100 * self._reliability().pass_at_1) <= 0.1

    def test_pass_hat_4_matches(self):
        published = float(re.search(r"pass\^4 ([\d.]+)%",
                                    README.read_text()).group(1))
        assert abs(published - 100 * self._reliability().pass_hat_k) <= 0.1

    def test_the_quoted_run_count_is_the_number_of_artifacts(self):
        """'Four runs on one model' has to stay true as runs are published."""
        assert "Four runs" in README.read_text()
        assert self._reliability().k == 4

    def test_every_missed_row_escalated_rather_than_answering_wrongly(self):
        """The README's strongest sentence about pass^4, and the one most
        easily falsified by a later run."""
        runs = _runs("deepseek")
        ids = A.stable_ids(runs)
        missed = [i for i in ids
                  if not all(A.is_correct(r.rows[i]) for r in runs)]
        decisive = {"SUPPORTS", "REFUTES"}

        wrong = [(i, run.label) for i in missed for run in runs
                 if A.verdict(run.rows[i]) != A.expected(run.rows[i])
                 and A.verdict(run.rows[i]) in decisive]

        assert not wrong, (
            f"the README says every pass^4 miss escalated rather than "
            f"answering wrongly; these answered wrongly: {wrong}")


class TestTheClaimsAboutTheCodeHold:

    def test_the_count_of_declined_metrics_is_the_real_one(self):
        """'The other 45 metrics the parser accepts' comes from the routing
        table, not from a run, so it drifts whenever a metric is added."""
        from types import SimpleNamespace

        from finvet.config.metrics import METRIC_WHITELIST, verification_strategy_for

        unsupported = sum(
            1
            for claim_type, metrics in METRIC_WHITELIST.items()
            for metric in metrics
            if verification_strategy_for(
                SimpleNamespace(claim_type=claim_type, metric=metric))
            == "unsupported")
        published = int(re.search(r"other (\d+) metrics",
                                  README.read_text()).group(1))

        assert published == unsupported

    def test_the_zero_spend_claim_counts_the_right_rows(self):
        """The published count must be the rows that *must* spend nothing, not
        the subset that reached execution.

        It said 26, which is `zero_tool_actual`: the rows declined before an
        agent ran. Six more are refused by the input guardrail before any
        execution exists, so 32 claims carry the expectation and 26 was
        describing a subset while claiming the whole. A per-run figure, and the
        burned-row exclusions moved it after c1.
        """
        published = int(re.search(r"the (\d+) claims that must spend nothing",
                                  README.read_text()).group(1))
        measured = trajectory.measure(_runs("deepseek-c1"))

        assert published == measured.zero_tool_expected
        assert measured.zero_tool_breaches == [], (
            "a claim that must spend nothing called a tool; the README's "
            "'enforced by construction' claim no longer holds")


class TestTheUnverifiableFiguresAreMarkedAsSuch:
    """Neither retrieval figure can be recomputed here. Saying so is the
    contract; quietly publishing them as though they could is not."""

    def test_the_retrieval_numbers_declare_they_are_not_reproducible(self):
        text = " ".join(README.read_text().split())
        window = text[text.index("**Retrieval.**"):]
        window = window[:window.index("**Evidence.**")]

        assert "198/199" in window, "the XBRL figure moved; re-check this guard"
        assert re.search(r"held privately|not recomputable", window), (
            "the retrieval figures are measured against private gold sets and "
            "the README must say so wherever it states them")
