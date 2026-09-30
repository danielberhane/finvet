# Cross-model benchmark on golden_u — 2026-09-29

Two models, identical code, four runs each on the 349-claim combined set
(`golden_u.jsonl` v1.1, SHA-256 `fb2f98ca…f23c2`; 343 scored after the six burned
rows, 329 with an expected verdict). Code at commit `e0b3d22`. Every figure below
recomputes from the redacted artifacts beside this file
(`run-*-u-ds-c[1-4]-redacted.json`, `run-*-u-qw-c[1-4]-redacted.json`) and the
layer summaries (`layers-u-*.json`); `tests/unit/test_readme_numbers_recompute.py`
does so on every CI run.

## Models

| column | model | served by | context | id in the artifacts |
|---|---|---|---|---|
| DeepSeek | DeepSeek-V4.1-Flash | DeepSeek API | 1,048,576 tokens | `deepseek-flash` |
| Qwen | Qwen3.8 | vLLM 0.22.1 on a university research cluster, behind a LiteLLM gateway | 262,144 tokens | `Qwen3.8` |

The gateway in front of Qwen caches responses. Every request in these runs carried
the per-request opt-out (`cache: {"no-cache": true, "no-store": true}`), sent by a
local pass-through that counted its own traffic: 7,404 requests, 0 served from the
cache. Without the opt-out, repeated runs would replay the first run's answers and
pass^k would measure the cache. Neither provider publishes the parameter count of
these models.

## Results

Rows 1 to 5 and 7: the first run of each model. Row 6: all four runs.

| Layer | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| 1. Outcome — verdict accuracy (301 claims) | **97.0%** | **96.7%** |
| 2. Tool trajectory — required tools called | 99.2% (263/265) | 97.7% (260/266) |
| 3. Grounding — decisive numbers traced to a source | **100%** (242/242) | **100%** (239/239) |
| 4. Calibration — decisive-verdict ECE | 0.049 | 0.050 |
| 5. Asymmetric risk — confidently-wrong verdicts (of 329) | 1 | **0** |
| 5. Asymmetric risk — declined rather than answered | 2.7% | 3.6% |
| 6. Reliability — pass^4 over four runs (329 rows) | **94.8%** | **91.5%** |
| 7. Reachability — evidence path matches expectation | 94.8% | 97.9% |

Outcome excludes the 28 live-market rows, whose answer depends on the share price at
the moment of the run. Every other layer includes them.

Populations: outcome is scored on the 329 claims with an expected verdict minus those 28;
trajectory uses DeepEval's `ToolCorrectnessMetric` over the claims whose strategy requires a
tool; grounding covers every decisive number the run produced; asymmetric risk and pass^4 are
over all 329. Three properties are enforced by the code rather than measured: an agent cannot
call another agent's tools, the 59 claims that must spend nothing never reach one, and a
decisive numeric verdict is unreachable without a trusted observation.

### Per run

| run | model | outcome (329 rows, pass@1) | confidently wrong | declined | rows lost to timeouts or errors | p50 / p95 per claim | duration |
|---|---|---|---|---|---|---|---|
| u-ds-c1 | DeepSeek | 97.0% | 1 (row 1073) | 9 | 0 | 11 s / 76 s | 116 min |
| u-ds-c2 | DeepSeek | 97.9% | 0 | 7 | 0 | 12 s / 71 s | 114 min |
| u-ds-c3 | DeepSeek | 96.7% | 1 (row 1073) | 10 | 0 | 12 s / 60 s | 111 min |
| u-ds-c4 | DeepSeek | 97.0% | 0 | 10 | 0 | 11 s / 60 s | 105 min |
| u-qw-c1 | Qwen | 96.4% | 0 | 12 | 2 | 40 s / 193 s | 322 min |
| u-qw-c2 | Qwen | 94.5% | 0 | 17 | 3 | 40 s / 160 s | 325 min |
| u-qw-c3 | Qwen | 95.1% | 0 | 16 | 2 | 39 s / 141 s | 316 min |
| u-qw-c4 | Qwen | 96.4% | 0 | 12 | 1 | 40 s / 161 s | 293 min |

The four Qwen runs ran side by side through one gateway, which is why each took five
hours; alone, Qwen answers a claim in about 40 seconds at the median. The three later
DeepSeek runs also ran side by side.

### Pre-registered criteria (`docs/eval/GOLDEN_G_CARD.md`, scored by `scripts/score_golden_g.py` over all four runs)

| criterion | DeepSeek | Qwen | note |
|---|---|---|---|
| A1 zero confidently-wrong verdicts | fail (2, both row 1073) | fail (1) | Qwen's one is a row that timed out with no verdict, which the scorer counts against the expected decline; no wrong verdict was published |
| A2 every decisive number traceable | pass 969/969 | pass 948/948 | |
| A3 per-class accuracy within 10 points of the mega-cap baseline | pass, lowest class 96.2% | fail, restated 86.5% | Qwen runs out of steps on restated years and escalates |
| A4 tolerance boundary rows | pass 20/20 in every run | pass, 19/20 in two runs | |
| A5 limitation codes | fail 75/76 | fail 74/76 | one claim rejected by the parser instead of declined with a reason |
| A6 guards | pass 40/40 | pass 40/40 | |
| A7 fail-closed rows escalate | pass 24/24 | fail 23/24 | |
| A7b fines: no corroborated REFUTES, no wrong-amount SUPPORTS | pass | pass | |
| A8 phrasing twins within 5 points of originals | pass | pass | |
| A9 decisive ECE ≤ 0.05 | fail by 0.0004 | pass 0.048 | |
| A11 pass^4 ≥ 0.90 | **pass 0.948** | **pass 0.915** | |
| A12 live-price rows | fail 79/80 | fail 73/80 | Qwen's misses are parser declines (no measure named), not price moves |
| A13 plain vs seeded phrasing within 5 points | pass | pass | |

## Findings

1. **Both models rest every decisive verdict on a retrieved fact.** Grounding is
   100% in all eight runs: 1,917 decisive numbers, every one traceable to an XBRL
   fact, a quote or the deterministic penalty extraction.

2. **The one wrong verdict is a parsing defect, not a retrieval one.** Row 1073 states
   a net loss. In two of four DeepSeek runs the parser recorded the amount as positive,
   the comparison against the filed negative figure failed, and a true claim was refuted
   at 0.85. Qwen read the sign correctly in all four runs. No other row was answered
   wrongly by either model in any run — 2,632 judged answers.

3. **Qwen's gap to DeepSeek is caution.** Its declines and escalations cluster on the
   restated-year class, where it keeps calling tools until the step limit and FinVet
   hands the claim to a reviewer, and on live-price claims, where its parser sometimes
   names no measure. Both end in a decline. It also lost eight rows across four runs to
   timeouts or an empty parser reply (row 1101, in every run).

4. **Stability holds for both.** pass^4 of 94.8% and 91.5% against a pre-registered
   bar of 90%. Every DeepSeek miss but row 1073, and every Qwen miss, was an
   escalation, a decline or a lost row rather than a wrong answer.

5. **The comparator decides, for better and once for worse.** Across the eight runs
   the deterministic comparison changed the model's own verdict on 267 of 2,744
   answers: 199 times to the labelled verdict, 66 times to a decline or an
   escalation, and twice — row 1073 — from the model's correct SUPPORTS to a wrong
   REFUTES, because the parsed claim it was handed carried the wrong sign. The
   comparison is only as good as the parse it receives. Nine restated-period claims
   were declined because they matched a figure the issuer had since replaced.

## What this does not establish

- One vendor run per model was scored for rows 1 to 5 and 7; the four-run figures
  are in the layer summaries. Differences of a point or two between the two models
  on those rows are within the run-to-run spread shown in the per-run table.
- The Qwen runs shared one gateway, so their timings say little about Qwen's speed
  and nothing about its cost. The eight Qwen rows lost to timeouts include ones the
  gateway would likely have answered alone.
- The live-price rows were verified against quotes fetched during and after market
  hours on the day of the run; those rows are excluded from the outcome row for that
  reason and reported in A12.
- Retrieval was measured separately and is not recomputable from this repository.

## Reproducing the table

```
.venv/bin/python scripts/eval_layers.py --dir docs/eval --label u-ds-c1
.venv/bin/python scripts/eval_layers.py --dir docs/eval --label u-ds        # four runs, pass^4
.venv/bin/python scripts/eval_layers.py --dir docs/eval --label u-qw-c1
.venv/bin/python scripts/eval_layers.py --dir docs/eval --label u-qw
.venv/bin/python -m pytest tests/unit/test_readme_numbers_recompute.py -q
```
