import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_golden_g.py"
spec = importlib.util.spec_from_file_location("score_golden_g", SCRIPT)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)


def _ds(rid, cls, seed, verdict="SUPPORTS", strength="strict", category="sec/xbrl", limitation=None, twin_of=None):
    tags = {"class": cls, "seed": seed}
    if twin_of is not None:
        tags["twin_of"] = twin_of
    return {"id": rid, "claim": "c", "category": category, "strength": strength,
            "expected": {"verdict": verdict, "limitation": limitation, "sources": []},
            "ground_truth": "", "source": "", "gold_parse": {},
            "tags": tags}


def _res(rid, verdict, expected, conf=0.95, escalated=False, limitation=None, http=200):
    decisive = verdict in ("SUPPORTS", "REFUTES")
    return {"id": rid, "expected": {"verdict": expected, "limitation": limitation},
            "http": http, "request_id": "r", "error": None,
            "actual": {"verdict": verdict, "confidence": conf, "limitation": limitation,
                       "retrieved_value": 1.0 if decisive else None,
                       "observation_tool": "get_income_statement" if decisive else None,
                       "tools_called": [], "escalated": escalated, "a2a_status": None}}


def _run(results, label="g-ds-c1"):
    return sg.A.Run(label, "x", "m", True, {r["id"]: r for r in results})


def test_a3_flags_a_weak_class():
    ds = [_ds(1, "financials", "plain"), _ds(2, "financials", "plain"),
          _ds(3, "mid_cap", "plain"), _ds(4, "mid_cap", "plain")]
    res = [_res(1, "SUPPORTS", "SUPPORTS"), _res(2, "NOT_ENOUGH_INFO", "SUPPORTS"),
           _res(3, "SUPPORTS", "SUPPORTS"), _res(4, "SUPPORTS", "SUPPORTS")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A3"]["pass"] is False
    assert out["A3"]["value"]["financials"] == 0.5


def test_a7_counts_escalations_on_hitl_rows():
    ds = [_ds(11, "hitl", "range", verdict=None, strength="observe", category="known-defect")]
    res = [_res(11, "PENDING", None, conf=0.0, escalated=True)]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A7"]["value"] == "1/1" and out["A7"]["pass"] is True


def test_a3_skips_twin_rows():
    """Verify twins do not count in per-class accuracy.
    Two plain rows (1 correct, 1 wrong → 0.5) + one correct twin in same class.
    Twin should not lift accuracy to 0.667; must stay at 0.5.
    """
    ds = [_ds(1, "financials", "plain"), _ds(2, "financials", "plain"),
          _ds(3, "financials", "twin", twin_of=1)]
    res = [_res(1, "SUPPORTS", "SUPPORTS"), _res(2, "NOT_ENOUGH_INFO", "SUPPORTS"),
           _res(3, "SUPPORTS", "SUPPORTS")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A3"]["value"]["financials"] == 0.5


def test_a3_skips_twin_and_boundary_rows():
    """Verify twin and boundary rows do not count in per-class accuracy.
    Two plain rows (1 correct, 1 wrong → 0.5) + one correct twin +
    one correct boundary_in, same class. Neither should count.
    Must stay at 0.5, not lifted.
    """
    ds = [_ds(1, "financials", "plain"), _ds(2, "financials", "plain"),
          _ds(3, "financials", "twin", twin_of=1), _ds(4, "financials", "boundary_in")]
    res = [_res(1, "SUPPORTS", "SUPPORTS"), _res(2, "NOT_ENOUGH_INFO", "SUPPORTS"),
           _res(3, "SUPPORTS", "SUPPORTS"), _res(4, "SUPPORTS", "SUPPORTS")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A3"]["value"]["financials"] == 0.5


def test_a7b_tolerates_a_decline_but_not_an_inversion():
    """Ruling 18: A7b asserts only that no fines row is inverted.

    A corroborated row that declines (NOT_ENOUGH_INFO) must not fail A7b — the
    SUPPORTS rate is reported as a measurement. A corroborated row that comes
    back REFUTES asserts the opposite of the filing and must fail.
    """
    ds = [_ds(21, "fines", "corroborated", strength="safe", category="a2a"),
          _ds(22, "fines", "uncertifiable", verdict="NOT_ENOUGH_INFO", strength="safe",
              category="a2a", limitation="amount_not_certifiable"),
          _ds(23, "fines", "wrong_amount", verdict="REFUTES", strength="safe", category="a2a")]
    unc_ok = _res(22, "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", limitation="amount_not_certifiable")
    wrong_ok = _res(23, "REFUTES", "REFUTES")

    declined = sg.score(ds, [_run([_res(21, "NOT_ENOUGH_INFO", "SUPPORTS"), unc_ok, wrong_ok])],
                        baseline=1.0)
    assert declined["A7b"]["pass"] is True
    assert declined["A7b"]["value"]["corroborated_supports"] == "0/1"
    assert declined["A7b"]["value"]["corroborated_not_refutes"] == "1/1"

    inverted = sg.score(ds, [_run([_res(21, "REFUTES", "SUPPORTS"), unc_ok, wrong_ok])],
                        baseline=1.0)
    assert inverted["A7b"]["pass"] is False
    assert inverted["A7b"]["value"]["corroborated_not_refutes"] == "0/1"


def test_a4_is_per_run_not_pooled():
    """One run at 17/20 must fail A4 even though the pooled rate (37/40) is >= 0.9."""
    ds = [_ds(i, "boundary", "boundary_in" if i % 2 == 0 else "boundary_out")
          for i in range(1, 21)]
    res_full = [_res(i, "SUPPORTS", "SUPPORTS") for i in range(1, 21)]
    res_17 = [_res(i, "SUPPORTS", "SUPPORTS") if i <= 17
              else _res(i, "NOT_ENOUGH_INFO", "SUPPORTS") for i in range(1, 21)]
    run_ok = _run(res_full, label="run-ok")
    run_weak = _run(res_17, label="run-weak")
    out = sg.score(ds, [run_ok, run_weak], baseline=1.0)
    assert out["A4"]["value"]["run-ok"] == "20/20"
    assert out["A4"]["value"]["run-weak"] == "17/20"
    assert out["A4"]["pass"] is False


def test_a5_requires_the_not_enough_info_verdict():
    """A row with the right limitation but a SUPPORTS verdict must fail A5."""
    ds = [_ds(1, "limitation", "plain", limitation="non_usd_amount")]
    res = [_res(1, "SUPPORTS", "SUPPORTS", limitation="non_usd_amount")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A5"]["pass"] is False


def test_a11_is_unasserted_below_k3():
    ds = [_ds(1, "financials", "plain")]
    res = [_res(1, "SUPPORTS", "SUPPORTS")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A11"]["pass"] is None


def _c_row(rid, verdict="SUPPORTS"):
    """A golden_c-style sec/xbrl row: id < 1000, no tags."""
    return {"id": rid, "claim": "c", "category": "sec/xbrl", "strength": "strict",
            "expected": {"verdict": verdict, "limitation": None, "sources": []},
            "ground_truth": "", "source": "", "gold_parse": {}}


def test_a3_baseline_auto_from_same_run():
    """Two golden_c-style sec/xbrl rows (ids 4, 13 -- one correct, one wrong) give
    an auto baseline of 0.5, computed from the same run set. A G class also at
    0.5 then clears the 0.4 (baseline - 0.10) bar, and the baseline is exposed
    on the result.
    """
    ds = [_c_row(4), _c_row(13),
          _ds(1001, "mid_cap", "plain"), _ds(1002, "mid_cap", "plain")]
    res = [_res(4, "SUPPORTS", "SUPPORTS"), _res(13, "NOT_ENOUGH_INFO", "SUPPORTS"),
           _res(1001, "SUPPORTS", "SUPPORTS"), _res(1002, "NOT_ENOUGH_INFO", "SUPPORTS")]
    run = _run(res)

    baseline, n = sg.auto_baseline(ds, [run])
    assert baseline == 0.5 and n == 2

    out = sg.score(ds, [run], baseline)
    assert out["A3"]["baseline"] == 0.5
    assert out["A3"]["value"]["mid_cap"] == 0.5
    assert out["A3"]["pass"] is True


def test_a3_baseline_auto_falls_back_without_golden_c_rows():
    """A golden_g-only artifact (no id < 1000 sec/xbrl rows) falls back to 1.00
    with n=0."""
    ds = [_ds(1001, "mid_cap", "plain")]
    res = [_res(1001, "SUPPORTS", "SUPPORTS")]
    baseline, n = sg.auto_baseline(ds, [_run(res)])
    assert baseline == 1.00 and n == 0


def test_a3_fails_closed_on_empty_class_map():
    """Verify A3 fails when all rows are control-class or excluded seeds.
    If by_cls is empty, A3 should fail (fail closed), not vacuously pass.
    """
    ds = [_ds(1, "limitation", "plain"), _ds(2, "guard", "plain"),
          _ds(3, "financials", "twin", twin_of=1), _ds(4, "financials", "boundary_in")]
    res = [_res(1, "SUPPORTS", "SUPPORTS"), _res(2, "NOT_ENOUGH_INFO", "SUPPORTS"),
           _res(3, "SUPPORTS", "SUPPORTS"), _res(4, "SUPPORTS", "SUPPORTS")]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A3"]["pass"] is False


def test_a6_counts_a_length_rejection_as_a_block():
    """The API's own length validation answers before the guard, with 422."""
    ds = [_ds(21, "guard", "too_short", verdict="BLOCKED", category="guard"),
          _ds(22, "guard", "injection", verdict="BLOCKED", category="guard")]
    res = [_res(21, "BLOCKED", "BLOCKED", http=422),
           _res(22, "BLOCKED", "BLOCKED", http=400)]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A6"]["value"] == "2/2" and out["A6"]["pass"] is True


def test_a6_does_not_count_an_unanswered_request_as_a_block():
    ds = [_ds(21, "guard", "too_short", verdict="BLOCKED", category="guard")]
    res = [_res(21, None, "BLOCKED", http=422)]
    out = sg.score(ds, [_run(res)], baseline=1.0)
    assert out["A6"]["pass"] is False


def test_a_criterion_with_no_rows_is_unasserted_not_failed():
    """A run that carries none of a criterion's rows says nothing about it."""
    ds = [_ds(1, "financials", "plain")]
    out = sg.score(ds, [_run([_res(1, "SUPPORTS", "SUPPORTS")])], baseline=1.0)
    for criterion in ("A5", "A6", "A7", "A12"):
        assert out[criterion]["pass"] is None, criterion
