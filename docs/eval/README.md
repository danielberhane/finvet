# FinVet evaluation

Two models, identical code, four runs each on the 349-claim held-out set `golden_u`
(v1.1, SHA-256 `fb2f98ca…f23c2`), 2026-09-29, code at commit `e0b3d22`. Every figure
below recomputes from the artifacts in [`runs/`](runs/), and
`tests/unit/test_readme_numbers_recompute.py` does so on every CI run.

## Dataset

349 financial claims across 76 US-listed companies, in two parts that share no ticker:
97 written claims on five mega-caps, and 252 script-labelled claims on 71 filers chosen
to be hard (non-calendar fiscal years, banks and insurers, mid-caps, non-primary revenue
concepts, restated periods), with boundary, fines, live-price, limitation, guard and
phrasing-twin strata. Six burned rows are excluded, so a scored run is 343 rows, 329
with an expected verdict. Composition, labelling and limitations:
[`DATASET_CARD.md`](DATASET_CARD.md). The pass/fail criteria were committed before the
set was run: [`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md).

## Scoring

A row is correct when the verdict matches the label; a `safe` row fails only on the
opposite verdict; an `observe` row is recorded, never asserted. Each run is scored on
seven layers (`src/finvet/eval/measures/`):

1. **Outcome** — verdict accuracy, excluding the 28 live-price rows.
2. **Tool trajectory** — DeepEval's `ToolCorrectnessMetric` over the rows whose
   strategy requires a tool.
3. **Grounding** — every decisive number traced to an XBRL fact, a quote or the
   deterministic penalty extraction.
4. **Calibration** — expected calibration error over decisive verdicts.
5. **Asymmetric risk** — confidently-wrong verdicts (≥ 0.8), and the decline rate.
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

The gateway in front of Qwen caches responses. Every request carried the per-request
opt-out (`cache: {"no-cache": true, "no-store": true}`) through a local pass-through that
counted its traffic: 7,404 requests, 0 served from the cache. Neither provider publishes
the parameter count.

### Results

Rows 1 to 5 and 7: the first run of each model. Row 6: all four runs.

| Layer | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| 1. Outcome — verdict accuracy (301 claims) | **97.0%** | **96.7%** |
| 2. Tool trajectory — required tools called (265 / 266) | 99.2% | 97.7% |
| 3. Grounding — decisive numbers traced (242 / 239) | **100%** | **100%** |
| 4. Calibration — decisive-verdict ECE | 0.049 | 0.050 |
| 5. Asymmetric risk — confidently-wrong verdicts (of 329) | 1 | **0** |
| 5. Asymmetric risk — declined | 2.7% | 3.6% |
| 6. Reliability — pass^4 (329 rows) | **94.8%** | **91.5%** |
| 7. Routing — evidence by the expected path | 94.8% | 97.9% |

### Per run

| run | model | outcome (329, pass@1) | confidently wrong | declined | rows lost | p50 / p95 per claim | duration |
|---|---|---|---|---|---|---|---|
| u-ds-c1 | DeepSeek | 97.0% | 1 (row 1073) | 9 | 0 | 11 s / 76 s | 116 min |
| u-ds-c2 | DeepSeek | 97.9% | 0 | 7 | 0 | 12 s / 71 s | 114 min |
| u-ds-c3 | DeepSeek | 96.7% | 1 (row 1073) | 10 | 0 | 12 s / 60 s | 111 min |
| u-ds-c4 | DeepSeek | 97.0% | 0 | 10 | 0 | 11 s / 60 s | 105 min |
| u-qw-c1 | Qwen | 96.4% | 0 | 12 | 2 | 40 s / 193 s | 322 min |
| u-qw-c2 | Qwen | 94.5% | 0 | 17 | 3 | 40 s / 160 s | 325 min |
| u-qw-c3 | Qwen | 95.1% | 0 | 16 | 2 | 39 s / 141 s | 316 min |
| u-qw-c4 | Qwen | 96.4% | 0 | 12 | 1 | 40 s / 161 s | 293 min |

The four Qwen runs ran side by side through one gateway, as did DeepSeek runs 2 to 4;
the durations say nothing about either model alone.

### Pre-registered criteria (`scripts/score_golden_g.py`, all four runs)

| criterion | DeepSeek | Qwen | note |
|---|---|---|---|
| A1 zero confidently-wrong verdicts | fail (2, both row 1073) | fail (1) | Qwen's one is a row that timed out with no verdict, counted against the expected decline |
| A2 every decisive number traceable | pass 969/969 | pass 948/948 | |
| A3 per-class accuracy within 10 points of the mega-cap baseline | pass, lowest class 96.2% | fail, restated 86.5% | Qwen runs out of steps on restated years and escalates |
| A4 tolerance boundary rows | pass 20/20 every run | pass, 19/20 in two runs | |
| A5 limitation codes | fail 75/76 | fail 74/76 | one claim rejected by the parser instead of declined with a reason |
| A6 guards | pass 40/40 | pass 40/40 | |
| A7 fail-closed rows escalate | pass 24/24 | fail 23/24 | |
| A7b fines: no corroborated REFUTES, no wrong-amount SUPPORTS | pass | pass | |
| A8 phrasing twins within 5 points of originals | pass | pass | |
| A9 decisive ECE ≤ 0.05 | fail by 0.0004 | pass 0.048 | |
| A11 pass^4 ≥ 0.90 | **pass 0.948** | **pass 0.915** | |
| A12 live-price rows | fail 79/80 | fail 73/80 | Qwen's misses are parser declines, not price moves |
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
| faithfulness of the model's reading, 50 claims (RAGAS, judge-scored) | ≥ 0.85 | pending |

Four of the six misses returned a same-subject passage from another section of the same
filing; two returned text off the subject. The earlier 70-case calibration set, on which the
relevance floor was tuned, is a fit and is not reported as a result.

## Findings

1. **Every decisive verdict rests on a retrieved fact.** Grounding is 100% in all eight
   runs: 1,917 decisive numbers, each traceable to an XBRL fact, a quote or the
   deterministic penalty extraction.
2. **The one wrong verdict is a parsing defect, not a retrieval one.** Row 1073 states a
   net loss. In two of four DeepSeek runs the parser recorded the amount as positive, the
   comparison against the filed negative figure failed, and a true claim was refuted at
   0.85. Qwen read the sign correctly in all four runs. No other row was answered wrongly
   by either model in any run, across 2,632 judged answers.
3. **The comparator decides, for better and once for worse.** Across the eight runs the
   deterministic comparison changed the model's own verdict on 267 of 2,744 answers: 199
   times to the labelled verdict, 66 times to a decline or an escalation, and twice (row
   1073) from the model's correct SUPPORTS to a wrong REFUTES, because the parsed claim it
   was handed carried the wrong sign. Nine restated-period claims were declined because
   they matched a figure the issuer had since replaced.

**Stability.** Four runs per model. DeepSeek-V4.1-Flash: pass^4 94.8%, 17 misses on at
least one run, one of them the wrong verdict above. Qwen3.8: pass^4 91.5%, 28 misses, none a
wrong verdict.

Qwen's gap to DeepSeek is caution: its declines cluster on restated years, where it
reaches the step limit and escalates, and on live-price claims, where its parser
sometimes names no measure. It also lost eight rows across four runs to timeouts or an
empty parser reply (row 1101, every run).

## What this does not establish

- Rows 1 to 5 and 7 are one run per model; differences of a point or two between the
  models are within the run-to-run spread in the per-run table.
- The Qwen runs shared one gateway, so their timings say little about Qwen's speed and
  nothing about its cost.
- Live-price rows were verified against quotes fetched on the day of the run and are
  reported only under A12.
- Retrieval is measured on private sets and is not recomputable here.

## Reproducing the tables

```
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-ds-c1
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-ds      # four runs, pass^4
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-qw-c1
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-qw
.venv/bin/python -m pytest tests/unit/test_readme_numbers_recompute.py -q
```

## Changelog

- **2026-09-29** — golden_u v1.1, eight runs (four per model), the report above.
- **2026-08-31** — an earlier benchmark on the 97-claim first part alone, with
  `deepseek-chat` and MiniMax-M2.7. Superseded by the above and removed from the tree;
  its write-up and artifacts remain in git history.
