"""The four measures that read a recorded run, tested on hand-built runs.

Synthetic rows rather than real artifacts: each test needs one specific shape --
a row that flips between runs, a decline scored at low confidence, a decisive
verdict with no observation behind it -- and constructing those directly is the
only way to assert what each measure does at its boundaries.

The real artifacts are scored by `scripts/eval_layers.py`, which reports:

    pass@1 0.989 · pass^4 0.945 (n=91) · ECE 0.039 decisive · 0 dangerous errors

Those differ from the figures computed while planning, and the modules are
right. Planning counted the four `observe` rows -- known defects whose
`expected.verdict` is null -- as failures, which dragged pass@1 to 0.940. A row
that asserts nothing cannot be got wrong.
"""

from finvet.eval.measures import artifacts, calibration, grounding, reliability, risk


def _row(rid, expected="SUPPORTS", got="SUPPORTS", conf=0.95, *,
         escalated=False, http=200, retrieved=1.0, tool="get_income_statement"):
    return {
        "id": rid,
        "expected": {"verdict": expected},
        "http": http,
        "actual": {"verdict": got, "confidence": conf, "escalated": escalated,
                   "retrieved_value": retrieved, "observation_tool": tool},
    }


def _run(label, rows):
    return artifacts.Run(label=label, started_utc="t", model="m",
                         complete=True, rows={r["id"]: r for r in rows})


class TestArtifactsNormalisation:

    def test_a_guardrail_block_counts_as_a_verdict(self):
        """HTTP 400 carries no verdict field; reading it raw scored six correct
        blocks as empty answers."""
        row = _row(1, expected="BLOCKED", got=None, http=400)

        assert artifacts.verdict(row) == "BLOCKED"
        assert artifacts.is_correct(row)

    def test_an_unscored_row_is_never_counted_wrong(self):
        """`observe` rows record a known defect and assert nothing."""
        row = _row(1, expected=None, got="REFUTES")

        assert not artifacts.is_scored(row)
        assert not artifacts.is_correct(row)

    def test_a_row_changed_by_a_fix_is_excluded_from_stability(self):
        runs = [_run("a", [_row(1), _row(88)]), _run("b", [_row(1), _row(88)])]

        assert artifacts.stable_ids(runs) == [1]


class TestReliability:

    def test_pass_hat_k_requires_success_in_every_run(self):
        """The point of pass^k: one lucky run is not a solve."""
        runs = [_run("a", [_row(1), _row(2)]),
                _run("b", [_row(1), _row(2, got="REFUTES")])]

        out = reliability.measure(runs)

        assert out.pass_at_1 == 0.5      # latest run: 1 of 2
        assert out.pass_hat_k == 0.5     # only row 1 passed both
        assert out.unstable == [2]

    def test_a_perfectly_stable_pair_scores_one(self):
        runs = [_run("a", [_row(1)]), _run("b", [_row(1)])]

        out = reliability.measure(runs)

        assert out.pass_hat_k == 1.0
        assert out.flake_rate == 0.0

    def test_no_runs_is_not_a_crash(self):
        assert reliability.measure([]).n == 0


class TestCalibration:

    def test_a_confident_correct_answer_is_well_calibrated(self):
        runs = [_run("a", [_row(i, conf=0.95) for i in range(1, 21)])]

        out = calibration.measure(runs)

        assert out.ece < 0.06           # 1.00 accuracy vs 0.95 confidence

    def test_declines_are_reported_apart_from_decisive_verdicts(self):
        """The real finding: a correct decline scored 0.2 dominates the
        aggregate while the decisive band is near-perfect.

        The split is decisive-vs-decline, not escalated-vs-not. Escalations
        report 0.0 and are counted wrong, which is well calibrated; the
        declines that carry the error never escalate."""
        decisive = [_row(i, conf=0.95) for i in range(1, 11)]
        declines = [_row(100 + i, expected="NOT_ENOUGH_INFO",
                         got="NOT_ENOUGH_INFO", conf=0.2)
                    for i in range(10)]

        out = calibration.measure([_run("a", decisive + declines)])

        assert out.decisive_n == 10
        assert out.decisive_ece < out.ece, (
            "pooling declines with decisive verdicts should worsen the aggregate")

    def test_bins_report_their_own_gap(self):
        out = calibration.measure([_run("a", [_row(1, conf=0.95)])])

        assert out.bins[0].lower == 0.9
        assert out.bins[0].gap == 1.0 - 0.95


class TestGrounding:

    def test_a_decisive_number_without_an_observation_is_flagged(self):
        """The hallucination case: a verdict carrying a figure nothing
        produced."""
        runs = [_run("a", [_row(1, retrieved=391e9, tool=None)])]

        out = grounding.measure(runs)

        assert out.untraceable == [1]
        assert out.rate == 0.0

    def test_a_traceable_verdict_scores(self):
        out = grounding.measure([_run("a", [_row(1, retrieved=391e9)])])

        assert out.rate == 1.0 and out.untraceable == []

    def test_a_qualitative_verdict_has_nothing_to_trace(self):
        """SUPPORTS on a claim naming no value asserts no number."""
        out = grounding.measure([_run("a", [_row(1, retrieved=None, tool=None)])])

        assert out.decisive_numeric == 0


class TestRisk:

    def test_an_opposite_verdict_is_dangerous(self):
        out = risk.measure([_run("a", [_row(1, expected="SUPPORTS",
                                            got="REFUTES")])])

        assert len(out.dangerous) == 1

    def test_a_decline_is_not_dangerous(self):
        """Both miss the expectation; only one shows a user a false answer."""
        out = risk.measure([_run("a", [_row(1, expected="SUPPORTS",
                                            got="NOT_ENOUGH_INFO")])])

        assert out.dangerous == []
        assert out.declined == 1

    def test_an_escalation_is_not_dangerous(self):
        out = risk.measure([_run("a", [_row(1, expected="REFUTES",
                                            got="PENDING", escalated=True)])])

        assert out.dangerous == []
        assert out.declined == 1
