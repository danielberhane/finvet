# FinVet evaluation

Two models, identical code, four runs each on the 325-claim held-out set `golden_u`
(v1.2, SHA-256 `f602b2fe…3a342`). 317 claims were answered on 2026-09-29 with code at
commit `e0b3d22`. The 8 claims that v1.2 changed were answered on 2026-10-15 with code at
`7f94785`, whose pipeline is the same. Every figure below recomputes from the artifacts
in [`runs/`](runs/), and `tests/unit/test_readme_numbers_recompute.py` does so on every
CI run.

## Dataset

325 financial claims across 72 US-listed companies, chosen to be hard (non-calendar
fiscal years, banks and insurers, mid-caps, non-primary revenue concepts, restated
periods), with boundary, fines, live-price, limitation, guard and phrasing-twin strata.
Every claim carries an expected verdict. Composition, labelling and limitations:
[`DATASET_CARD.md`](DATASET_CARD.md). The pass/fail criteria were committed before the
set was first run: [`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md).

### What changed in v1.2

Version 1.1 had 343 claims. 58 of them were graded leniently, failing only on the
opposite verdict, and 14 were recorded but not graded. Version 1.2 grades every claim
strictly:

- **Lenient claims became strict.** 46 of the 58 now must match their label exactly. Of
  the other 12, four were removed and eight were rewritten, as below.
- **18 claims were removed:** four fines claims that only the news could answer, and the
  14 ungraded ones. This was decided after the v1.1 results were seen, and one removed
  claim had failed seven of eight answers.
- **Eight fixed-price claims became live-price claims.** Their thresholds were so far from
  the price that the answer could not change. They were run again, four times per model,
  on 2026-10-15.
- **The human-review gate is no longer measured.** Six of the removed claims were its
  only test, so criterion A7 has no rows.

The 317 unchanged claims keep their 2026-09-29 answers and are graded by v1.2. Two things
differ for the eight new runs. They ran during US market hours. Qwen's requests for them
did not carry the gateway's cache opt-out, but all 32 claim texts were different, so no
answer could be a cached repeat. No check was run that either model still answers as it
did on 2026-09-29.

## Scoring

A claim is correct when its verdict matches the label. Each run is scored on seven
layers (`src/finvet/eval/measures/`):

1. **Outcome** — verdict accuracy, excluding the 28 live-price claims.
2. **Tool trajectory** — DeepEval's `ToolCorrectnessMetric` over the claims whose
   strategy requires a tool.
3. **Grounding** — every decisive number traced to an XBRL fact, a quote or the
   deterministic penalty extraction.
4. **Calibration** — expected calibration error over decisive verdicts.
5. **Asymmetric risk** — dangerous verdicts (a SUPPORTS or REFUTES that does not match the expected answer), and the decline rate.
6. **Reliability** — pass^k over repeated runs.
7. **Routing** — evidence arrived by the expected path.

Three properties are enforced by the code rather than measured: an agent cannot call
another agent's tools, the 59 claims that must spend nothing never reach one, and a
decisive numeric verdict is unreachable without a trusted observation.

## Evaluation report

### Models

| column | model | served by | context | id in the artifacts |
|---|---|---|---|---|
| DeepSeek | DeepSeek-V4.1-Flash | DeepSeek API | 1,048,576 tokens | `deepseek-flash` |
| Qwen | Qwen3.8 | vLLM 0.22.1 on a university research cluster, behind a LiteLLM gateway | 262,144 tokens | `Qwen3.8` |

The gateway in front of Qwen caches responses. In the 2026-09-29 runs every request
carried the per-request opt-out (`cache: {"no-cache": true, "no-store": true}`) through a
local pass-through that counted its traffic: 7,404 requests, 0 served from the cache. The
32 Qwen answers of 2026-10-15 went to the gateway directly; see above. Neither provider
publishes the parameter count.

### Results

Rows 1 to 5 and 7: the first run of each model. Row 6: all four runs.

| Layer | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| 1. Outcome — verdict accuracy (297 claims) | **97.3%** | **97.0%** |
| 2. Tool trajectory — required tools called (252 / 253) | 100% | 98.8% |
| 3. Grounding — decisive numbers traced (238 / 235) | **100%** | **100%** |
| 4. Calibration — decisive-verdict ECE | 0.049 | 0.050 |
| 5. Asymmetric risk — dangerous verdicts (of 325) | 1 | **0** |
| 5. Asymmetric risk — declined | 2.2% | 3.1% |
| 6. Reliability — pass^4 (325 claims) | **95.4%** | **92.0%** |
| 7. Routing — evidence by the expected path | 95.4% | 98.5% |

### Per run

Each run is a 2026-09-29 run with the eight changed claims answered on 2026-10-15.

| run | model | outcome (297) | all claims (325, pass@1) | dangerous | declined | claims lost |
|---|---|---|---|---|---|---|
| u-ds-c1 + u12-ds-c1 | DeepSeek | 97.3% | 97.5% | 1 (row 1073) | 7 | 0 |
| u-ds-c2 + u12-ds-c2 | DeepSeek | 98.3% | 98.5% | 0 | 5 | 0 |
| u-ds-c3 + u12-ds-c3 | DeepSeek | 97.3% | 97.2% | 1 (row 1073) | 8 | 0 |
| u-ds-c4 + u12-ds-c4 | DeepSeek | 97.3% | 97.2% | 0 | 9 | 0 |
| u-qw-c1 + u12-qw-c1 | Qwen | 97.0% | 96.9% | 0 | 10 | 1 |
| u-qw-c2 + u12-qw-c2 | Qwen | 96.3% | 94.5% | 0 | 18 | 3 |
| u-qw-c3 + u12-qw-c3 | Qwen | 97.0% | 96.0% | 0 | 13 | 1 |
| u-qw-c4 + u12-qw-c4 | Qwen | 96.3% | 95.7% | 0 | 14 | 1 |

Timings are not given: each run's answers come from two dates.

### Pre-registered criteria (`scripts/score_golden_g.py`, all four runs)

| criterion | DeepSeek | Qwen | note |
|---|---|---|---|
| A1 zero dangerous verdicts | fail (2, both row 1073) | pass (0) | a claim that timed out with no verdict counts as declined, not dangerous (PR #37) |
| A2 every decisive number traceable | pass 952/952 | pass 927/927 | |
| A3 per-class accuracy within 10 points of the baseline | pass, lowest class 96.2% | fail, restated 86.5% | Qwen runs out of steps on restated years and escalates |
| A4 tolerance boundary claims | pass 20/20 every run | pass, 19/20 in two runs | |
| A5 limitation codes | fail 75/76 | fail 74/76 | one claim rejected by the parser instead of declined with a reason |
| A6 guards | pass 40/40 | pass 40/40 | |
| A7 fail-closed claims escalate | not measured | not measured | its six claims were removed in v1.2 |
| A7b fines: no corroborated REFUTES, no wrong-amount SUPPORTS | pass | pass | |
| A8 phrasing twins within 5 points of originals | pass | pass | |
| A9 decisive ECE ≤ 0.05 | fail by 0.0003 | pass 0.048 | |
| A11 pass^4 ≥ 0.90 | **pass 0.954** | **pass 0.920** | |
| A12 live-price claims | fail 110/112 | fail 97/112 | every miss is a parser decline, not a price move |
| A13 plain vs seeded phrasing within 5 points | pass | pass | |

### Retrieval, measured separately

XBRL lookups against SEC primary-source values: 198/199, the miss a decline rather than
a wrong number, on a gold set held privately and not recomputable here.

Filing-text retrieval, on a held-out set of 50 queries frozen before the run
([`RAG_CARD.md`](RAG_CARD.md), artifacts in `runs/`). The first four rows are scored by
passage id with no judge; the last is RAGAS faithfulness, judged by a second model:

| measure | bar | result |
|---|---|---|
| labelled passage in the top five (hit rate@5) | ≥ 0.87 | 0.80, 24/30 |
| mean reciprocal rank | ≥ 0.70 | 0.68 |
| off-topic queries returning nothing | 12/12 | 12/12 |
| wrong-period or wrong-form queries returning nothing | 8/8 | 8/8 |
| faithfulness of the model's reading, 50 claims (RAGAS, judge-scored) | ≥ 0.85 mean, none below 0.50 | 0.687 mean, 0.125 lowest, 49 graded |

Four of the six misses returned a same-subject passage from another section of the same
filing; two returned text off the subject. The faithfulness test fails both parts of its bar,
with nine claims below 0.50; whether the shortfall is the model's or the judge's has not yet
been checked by hand. Its verdicts matched on 38 of 50 claims: every contradicted claim was
declined, because FinVet turns a REFUTES on a claim with no number into a decline. The earlier 70-case calibration set, on which the
relevance floor was tuned, is a fit and is not reported as a result.

## Findings

1. **Every decisive verdict rests on a retrieved fact.** Grounding is 100% in all eight
   runs: 1,879 decisive numbers, each traceable to an XBRL fact, a quote or the
   deterministic penalty extraction.
2. **The one wrong verdict is a parsing defect, not a retrieval one.** Row 1073 states a
   net loss. In two of four DeepSeek runs the parser recorded the amount as positive, the
   comparison against the filed negative figure failed, and a true claim was refuted at
   0.85. Qwen read the sign correctly in all four runs. No other claim was answered wrongly
   by either model in any run, across 2,600 judged answers.
3. **The comparator decides, for better and once for worse.** Across the eight runs the
   deterministic comparison changed the model's own verdict on 218 of 2,600 answers: 194
   times to the labelled verdict, 22 times to a decline or an escalation, and twice (row
   1073) from the model's correct SUPPORTS to a wrong REFUTES, because the parsed claim it
   was handed carried the wrong sign. Nine answers, on two restated-period claims, were
   declined because they matched a figure the issuer had since replaced.

**Stability.** Four runs per model. DeepSeek-V4.1-Flash: pass^4 95.4%, 15 misses on at
least one run, one of them the wrong verdict above. Qwen3.8: pass^4 92.0%, 26 misses, none a
wrong verdict.

Qwen's gap to DeepSeek is caution: its declines cluster on restated years, where it
reaches the step limit and escalates, and on live-price claims, where its parser
sometimes names no measure (8 of its 32 answers to the new live-price claims). It also
lost six claims across four runs to timeouts or an empty parser reply (row 1101, every
run).

## What this does not establish

- Rows 1 to 5 and 7 are one run per model; differences of a point or two between the
  models are within the run-to-run spread in the per-run table.
- The answers come from runs two weeks apart, and no check was run that the served models
  did not change in between.
- The Qwen runs shared one gateway, so their timings say little about Qwen's speed and
  nothing about its cost.
- Live-price claims were verified against quotes fetched on the day of the run and are
  reported only under A12.
- The human-review gate has no test in this set.
- Retrieval is measured on private sets and is not recomputable here.

## Reproducing the tables

The merged v1.2 runs are rebuilt from the public artifacts by the recompute test:

```
OPENAI_API_KEY=any .venv/bin/python -m pytest tests/unit/test_readme_numbers_recompute.py -q
```

With the private dataset file, the layer and criteria scripts give every table:

```
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-ds \
  --dataset <golden_u v1.2 file> --patch u-ds-c1=u12-ds-c1 --patch u-ds-c2=u12-ds-c2 \
  --patch u-ds-c3=u12-ds-c3 --patch u-ds-c4=u12-ds-c4
.venv/bin/python scripts/score_golden_g.py --dir docs/eval/runs --label u-ds \
  --dataset <golden_u v1.2 file> --patch u-ds-c1=u12-ds-c1 ...   # same pairs; u-qw likewise
```

## Changelog

- **2026-10-15** — golden_u v1.2: 325 claims, all graded strictly; the eight changed
  claims run four times per model; the report above.
- **2026-09-29** — golden_u v1.1, 343 claims, eight runs (four per model).
