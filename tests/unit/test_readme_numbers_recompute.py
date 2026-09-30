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
    @pytest.mark.parametrize("column,label", [(0, "u-ds-c1"),
                                              (1, "u-qw-c1")])
    def test_the_published_cell_matches_the_artifact(
            self, prefix, compute, parse, tolerance, column, label):
        published = parse(_table_row(prefix)[column])
        actual = compute(_runs(label))

        assert abs(published - actual) <= tolerance, (
            f"README row {prefix!r} column {label} publishes {published}, "
            f"the artifact gives {actual:.4f}")


def _paragraph(marker: str) -> str:
    """The README paragraph that opens with `marker`."""
    text = README.read_text()
    start = text.index(marker)
    return text[start:text.index("\n\n", start)]


class TestTheTableNamesTheModelsTheArtifactsRecord:
    """The header was checked by nothing, and it was wrong.

    Every cell of the results table is recomputed above, and the column
    headers were not. They named `deepseek-v4-flash`, a string that appeared
    nowhere else in the repository and in none of the five artifacts, all of
    which record `deepseek-chat`. The rows that once explained it -- a
    "requested as" / "actually served" pair describing the alias -- were cut
    as redundant, which left the label with nothing behind it.

    In a project whose runner refuses to start on model drift because 56 rows
    were once mislabelled, the headline table naming an unrecorded model is
    the failure the machinery exists to prevent, one layer up.
    """

    @pytest.mark.parametrize("column,label", [(1, "u-ds-c1"),
                                              (2, "u-qw-c1")])
    def test_the_column_header_matches_the_artifact(self, column, label):
        header = next(line for line in README.read_text().splitlines()
                      if line.startswith("| Layer |"))
        published = header.split("|")[column + 1].strip()
        assert "`" not in published and "(" not in published, (
            f"the table column reads {published!r}; it should be the "
            "model's plain name, with the served id stated under Models")

        served = _runs(label)[0].model
        if published == served:
            return

        # The header gives the model's proper name; the Models paragraph
        # must tie that name to the id the artifact records.
        models = _paragraph("**Models.**")
        assert re.search(
            rf"{re.escape(published)} is served by .*? under the id "
            rf"`{re.escape(served)}`", models, re.DOTALL), (
            f"the table column reads {published!r}, but the Models "
            f"paragraph does not say it is served under the id {served!r}, "
            f"which the {label} artifact records")


class TestTheStabilityFiguresRecompute:
    """pass@1 and pass^4 for each model, quoted in prose rather than the table.

    The Stability paragraph names each model before its figures, so the
    figures are matched after the model's name and checked against that
    model's four artifacts.
    """

    MODELS = [("DeepSeek-V4.1-Flash", "u-ds"), ("Qwen3.8", "u-qw")]

    def _paragraph(self):
        return _paragraph("**Stability.**")

    def _figures(self, name):
        after = self._paragraph()[self._paragraph().index(name):]
        at_1 = float(re.search(r"pass@1 ([\d.]+)%", after).group(1))
        hat_4 = float(re.search(r"pass\^4 ([\d.]+)%", after).group(1))
        return at_1, hat_4

    @pytest.mark.parametrize("name,label", MODELS)
    def test_pass_at_1_matches(self, name, label):
        published, _ = self._figures(name)
        assert abs(published - 100 * reliability.measure(_runs(label)).pass_at_1) <= 0.1

    @pytest.mark.parametrize("name,label", MODELS)
    def test_pass_hat_4_matches(self, name, label):
        _, published = self._figures(name)
        assert abs(published - 100 * reliability.measure(_runs(label)).pass_hat_k) <= 0.1

    @pytest.mark.parametrize("name,label", MODELS)
    def test_the_quoted_run_count_is_the_number_of_artifacts(self, name, label):
        """'Four runs per model' has to stay true as runs are published."""
        assert "Four runs per model" in README.read_text()
        assert reliability.measure(_runs(label)).k == 4

    @pytest.mark.parametrize("name,label,wrong_rows", [
        ("DeepSeek-V4.1-Flash", "u-ds", 1),
        ("Qwen3.8", "u-qw", 0),
    ])
    def test_the_account_of_the_missed_rows_holds(self, name, label, wrong_rows):
        """The README's strongest sentences about pass^4, and the ones most
        easily falsified by a later run: how many rows missed pass^4, and how
        many of those ever returned a wrong verdict rather than escalating,
        declining or being lost."""
        runs = _runs(label)
        ids = A.stable_ids(runs)
        missed = [i for i in ids
                  if not all(A.is_correct(r.rows[i]) for r in runs)]
        decisive = {"SUPPORTS", "REFUTES"}
        wrong = {i for i in missed for run in runs
                 if A.verdict(run.rows[i]) != A.expected(run.rows[i])
                 and A.verdict(run.rows[i]) in decisive}

        after = self._paragraph()[self._paragraph().index(name):]
        published_missed = int(re.search(r"(\d+) (?:rows that missed|misses)", after).group(1))
        assert published_missed == len(missed), (
            f"{name}: README says {published_missed} rows missed pass^4; "
            f"the artifacts give {len(missed)}")
        assert len(wrong) == wrong_rows, (
            f"{name}: README describes {wrong_rows} wrongly answered row(s); "
            f"the artifacts give {sorted(wrong)}")


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
        measured = trajectory.measure(_runs("u-ds-c1"))

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
