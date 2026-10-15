"""Grading recorded runs against a later dataset version, and patching in new answers.

Drives the published 2026-09-29 artifacts in docs/eval/runs (what
scripts/run_golden.py actually wrote), not hand-built rows. The patch rows are
real recorded live-quote rows re-numbered onto the stale ids, so they carry the
snapshot shape the runner emits. The dataset is built from the artifact the way
golden_u v1.2 changed v1.1: safe rows made strict, rows removed, and eight
fixed rows turned into live-quote recipes.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from finvet.eval.measures import artifacts as A

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "eval" / "runs"
STALE = list(range(57, 65))
REMOVED = {49, 98}
DATASET_NAME = "golden_u_v1.2.jsonl"


def _artifact_path(label):
    found = sorted(RUNS.glob(f"run-*-{label}-redacted.json"))
    if len(found) != 1:
        pytest.skip(f"published artifact {label} not present")
    return found[0]


@pytest.fixture
def base_json():
    return json.loads(_artifact_path("u-ds-c1").read_text())


@pytest.fixture
def dataset(base_json):
    rows = {}
    for r in base_json["results"]:
        if r["id"] in REMOVED:
            continue
        rows[r["id"]] = {"id": r["id"], "strength": "strict", "expected": r["expected"],
                         "category": r["category"]}
    for i in STALE:
        rows[i]["recipe"] = {"ticker": "AAPL", "operator": "lt", "offset_pct": 5}
    for r in base_json["results"]:
        if r.get("snapshot"):
            rows[r["id"]]["recipe"] = {"ticker": "JPM", "operator": "gt", "offset_pct": -10}
    return rows


def _patch_json(base_json, ids=STALE, **over):
    quotes = [r for r in base_json["results"] if r.get("snapshot")]
    results = []
    for new_id, row in zip(ids, quotes):
        results.append(dict(copy.deepcopy(row), id=new_id))
    out = dict(copy.deepcopy(base_json), label="u12-ds-c1", dataset=DATASET_NAME,
               selected=len(results), results=results)
    out.update(over)
    return out


def _run(data, tmp_path, name):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return A.load_run(path)


def test_rebase_grades_by_the_dataset_not_the_record(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    assert any(r["strength"] == "safe" for r in base.rows.values())

    run, stale = A.rebase(base, dataset)

    assert stale == STALE
    assert {r["strength"] for r in run.rows.values()} == {"strict"}
    assert not (REMOVED | set(STALE)) & set(run.rows)
    assert len(run) == len(base) - len(REMOVED) - len(STALE)


def test_rebase_drops_burned_rows(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    run, _ = A.rebase(base, dataset, burned={1073})
    assert 1073 not in run.rows


def test_merge_answers_the_stale_rows_from_the_patch(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json), tmp_path, "run-patch.json")

    merged = A.merge(base, patch, dataset, DATASET_NAME)

    assert merged.label == "u-ds-c1+u12-ds-c1"
    assert len(merged) == len(base) - len(REMOVED)
    assert all(merged.rows[i]["snapshot"] for i in STALE)
    assert all(merged.rows[i]["expected"] == dataset[i]["expected"] for i in STALE)


@pytest.mark.parametrize("change,message", [
    ({"llm_config": {"agent": {"model": "other"}}}, "not one run"),
    ({"complete": False}, "incomplete"),
    ({"dataset": "golden_u.jsonl"}, "ran against"),
])
def test_merge_refuses_a_patch_that_is_not_the_same_run(base_json, dataset, tmp_path,
                                                        change, message):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json, **change), tmp_path, "run-patch.json")
    with pytest.raises(ValueError, match=message):
        A.merge(base, patch, dataset, DATASET_NAME)


def test_merge_refuses_rows_the_dataset_does_not_have(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json, ids=STALE[:7] + [99999]), tmp_path, "p.json")
    with pytest.raises(ValueError, match="does not have"):
        A.merge(base, patch, dataset, DATASET_NAME)


def test_merge_refuses_to_leave_a_stale_row_unanswered(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json, ids=STALE[:7]), tmp_path, "p.json")
    with pytest.raises(ValueError, match=r"\[64\] need a new answer"):
        A.merge(base, patch, dataset, DATASET_NAME)


def test_merge_refuses_a_patch_row_with_no_quote(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    data = _patch_json(base_json)
    data["results"][0]["snapshot"] = None
    patch = _run(data, tmp_path, "p.json")
    with pytest.raises(ValueError, match="no quote snapshot"):
        A.merge(base, patch, dataset, DATASET_NAME)


def _write_dataset(dataset, tmp_path):
    path = tmp_path / DATASET_NAME
    path.write_text("\n".join(json.dumps(r) for r in dataset.values()) + "\n")
    return path


@pytest.mark.parametrize("specs,message", [
    (["u-ds-c1"], "OLD=NEW"),
    (["u-ds-c9=u12-ds-c1"], "not among the selected runs"),
    (["u-ds-c1=u12-ds-c7"], "matches 0 complete runs"),
])
def test_regrade_refuses_unclear_patch_pairs(base_json, dataset, tmp_path, specs, message):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json), tmp_path, "run-patch.json")
    with pytest.raises(ValueError, match=message):
        A.regrade([base], [base, patch], _write_dataset(dataset, tmp_path), specs)


def test_regrade_refuses_a_label_two_runs_share(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    patch = _run(_patch_json(base_json), tmp_path, "run-patch.json")
    with pytest.raises(ValueError, match="matches 2 complete runs"):
        A.regrade([base], [base, patch, patch], _write_dataset(dataset, tmp_path),
                  ["u-ds-c1=u12-ds-c1"])


def test_regrade_without_a_patch_reports_what_needs_a_new_run(base_json, dataset, tmp_path):
    base = _run(base_json, tmp_path, "run-base.json")
    runs, info = A.regrade([base], [base], _write_dataset(dataset, tmp_path))
    assert info["stale_dropped"] == {"u-ds-c1": STALE}
    assert info["dataset"] == DATASET_NAME and len(info["dataset_sha256"]) == 64
    assert len(runs[0]) == len(base) - len(REMOVED) - len(STALE)


@pytest.fixture
def run_dir(base_json, dataset, tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    (d / "run-20260929T000000Z-u-ds-c1.json").write_text(json.dumps(base_json))
    (d / "run-20261015T000000Z-u12-ds-c1.json").write_text(json.dumps(_patch_json(base_json)))
    return d, _write_dataset(dataset, tmp_path)


def _script(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                          capture_output=True, text=True)


def test_eval_layers_scores_the_merged_run(run_dir, tmp_path):
    d, ds = run_dir
    out = tmp_path / "layers.json"
    proc = _script("eval_layers.py", "--dir", d, "--label", "u-ds-c1", "--dataset", ds,
                   "--patch", "u-ds-c1=u12-ds-c1", "--json", out)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(out.read_text())
    assert summary["runs"] == ["u-ds-c1+u12-ds-c1"]
    assert summary["regraded"]["patches"] == {"u-ds-c1": "u12-ds-c1"}
    assert summary["regraded"]["stale_dropped"] == {}
    assert "u-ds-c1+u12-ds-c1" in summary["regraded"]["outcome"]
    assert "LAYER 1  outcome" in proc.stdout


def test_eval_layers_refuses_a_patch_without_a_dataset(run_dir):
    d, _ = run_dir
    proc = _script("eval_layers.py", "--dir", d, "--label", "u-ds-c1",
                   "--patch", "u-ds-c1=u12-ds-c1")
    assert proc.returncode == 2 and "--patch needs --dataset" in proc.stderr


def test_eval_layers_refuses_a_stale_row_left_unanswered(base_json, run_dir):
    d, ds = run_dir
    short = _patch_json(base_json, ids=STALE[:7], label="u12-ds-c2")
    (d / "run-20261015T010000Z-u12-ds-c2.json").write_text(json.dumps(short))
    proc = _script("eval_layers.py", "--dir", d, "--label", "u-ds-c1", "--dataset", ds,
                   "--patch", "u-ds-c1=u12-ds-c2")
    assert proc.returncode == 2 and "need a new answer" in proc.stderr


def test_score_golden_g_scores_the_merged_run(run_dir):
    d, ds = run_dir
    proc = _script("score_golden_g.py", "--dir", d, "--label", "u-ds-c1", "--dataset", ds,
                   "--patch", "u-ds-c1=u12-ds-c1")
    assert "patched: u-ds-c1 + u12-ds-c1" in proc.stdout, proc.stderr
    assert "A1 " in proc.stdout


def test_the_strength_contract_judges_the_merged_run(run_dir, base_json):
    d, ds = run_dir
    env = {"FINVET_GOLDEN_RUN": str(d / "run-20260929T000000Z-u-ds-c1.json"),
           "FINVET_GOLDEN_DATASET": str(ds),
           "FINVET_GOLDEN_PATCH": str(d / "run-20261015T000000Z-u12-ds-c1.json")}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
                           str(ROOT / "tests" / "integration" / "test_golden.py"), "-k", "summary"],
                          capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"rows      : {len(base_json['results']) - len(REMOVED)}" in proc.stdout
    assert "label=u-ds-c1+u12-ds-c1" in proc.stdout
