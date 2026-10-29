# FinVet evaluation

Two models, identical code, four runs each on a 325-claim held-out set
([`DATASET_CARD.md`](DATASET_CARD.md)). 317 claims were answered on 2026-09-29 and the
8 live-price claims on 2026-10-15, on the same pipeline. The pass/fail criteria were
committed before the set was first run ([`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md)). Every
figure below except the claim-parsing table and the XBRL lookup count recomputes from the
artifacts in [`runs/`](runs/), and `tests/unit/test_readme_numbers_recompute.py` checks
the verification results on every CI run.

## Claim parsing

The claim parser performs structured extraction, mapping each claim to a JSON object. A
parse is correct when every field matches the gold parse. Scored on the 309 claims that
reach the parser; the other 16 are refused by the input guardrails before parsing.
Averaged over all four runs.

| | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| Exact match | **97.5%** | **89.3%** |

## Scoring

A claim is correct when its verdict matches the label. Each run is scored on seven
layers (`src/finvet/eval/measures/`):

1. **Outcome** — verdict accuracy over all 325 claims. The per-run table also gives it
   without the 28 live-price claims.
2. **Tool trajectory** — DeepEval's `ToolCorrectnessMetric` over the claims whose
   strategy requires a tool.
3. **Grounding** — every decisive number traced to an XBRL fact, a quote or the
   deterministic penalty extraction.
4. **Calibration** — expected calibration error over decisive verdicts.
5. **Asymmetric risk** — dangerous verdicts (a SUPPORTS or REFUTES that does not match
   the expected answer), and the decline rate.
6. **Reliability** — pass^k over repeated runs.
7. **Routing** — evidence arrived by the expected path.

Three properties are enforced by the code rather than measured: an agent cannot call
another agent's tools, the 59 claims that must spend nothing never reach one, and a
decisive numeric verdict is unreachable without a trusted observation.

## Models

| column | model | served by | context | id in the artifacts |
|---|---|---|---|---|
| DeepSeek | DeepSeek-V4.1-Flash | DeepSeek API | 1,048,576 tokens | `deepseek-flash` |
| Qwen | Qwen3.8 | vLLM 0.22.1 on a university research cluster, behind a LiteLLM gateway | 262,144 tokens | `Qwen3.8` |

The gateway in front of Qwen caches responses. Every 2026-09-29 request to it carried
the per-request cache opt-out, and each of the 32 live-price requests of 2026-10-15
carried a distinct claim text, so no run reused a cached answer.

## Results

Rows 1 to 5 and 7: the first run of each model. Row 6: all four runs.

| Layer | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| 1. Outcome — verdict accuracy (325 claims) | **97.5%** | **96.9%** |
| 2. Tool trajectory — required tools called (252 / 253) | 100% | 98.8% |
| 3. Grounding — decisive numbers traced (238 / 235) | **100%** | **100%** |
| 4. Calibration — decisive-verdict ECE | 0.049 | 0.050 |
| 5. Asymmetric risk — dangerous verdicts (of 325) | 1 | **0** |
| 5. Asymmetric risk — declined | 2.2% | 3.1% |
| 6. Reliability — pass^4 (325 claims) | **95.4%** | **92.0%** |
| 7. Routing — evidence by the expected path | 95.4% | 98.5% |

### Per run

Each run is a 2026-09-29 run with the eight live-price claims answered on 2026-10-15.
Timings are not given, since each run's answers come from two dates.

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

### Pre-registered criteria (`scripts/score_golden_g.py`, all four runs)

| criterion | DeepSeek | Qwen | note |
|---|---|---|---|
| A1 zero dangerous verdicts | fail (2, both row 1073) | pass (0) | a claim that timed out with no verdict counts as declined, not dangerous (PR #37) |
| A2 every decisive number traceable | pass 952/952 | pass 927/927 | |
| A3 per-class accuracy within 10 points of the baseline | pass, lowest class 96.2% | fail, restated 86.5% | Qwen runs out of steps on restated years and escalates |
| A4 tolerance boundary claims | pass 20/20 every run | pass, 19/20 in two runs | |
| A5 limitation codes | fail 75/76 | fail 74/76 | one claim rejected by the parser instead of declined with a reason |
| A6 guards | pass 40/40 | pass 40/40 | |
| A7 fail-closed claims escalate | not measured | not measured | the set has no rows for it |
| A7b fines: no corroborated REFUTES, no wrong-amount SUPPORTS | pass | pass | |
| A8 phrasing twins within 5 points of originals | pass | pass | |
| A9 decisive ECE ≤ 0.05 | fail by 0.0003 | pass 0.048 | |
| A11 pass^4 ≥ 0.90 | **pass 0.954** | **pass 0.920** | |
| A12 live-price claims | fail 110/112 | fail 97/112 | every miss is a parser decline, not a price move |
| A13 plain vs seeded phrasing within 5 points | pass | pass | |

### Retrieval, measured separately

XBRL lookups against SEC primary-source values: 198/199, the miss a decline rather than
a wrong number, on a gold set held privately and not recomputable here.

Filing-text retrieval (RAG) and the faithfulness of the model's reading are measured on
their own held-out sets ([`RAG_CARD.md`](RAG_CARD.md); artifacts
`runs/rag-heldout-2026-10-01.json` and `runs/rag-ragas-2026-10-05.json`):

| Test | Sample | Result |
|---|---|---|
| Finds the right passage in its top five | 30 queries | 24 of 30 (80%) |
| Ranks it near the top (mean reciprocal rank) | 30 queries | 0.68 |
| Returns nothing for off-topic or wrong-filing queries | 20 queries | 20 of 20 |
| Explanation stays inside the passages (RAGAS faithfulness) | 49 of 50 claims graded | 0.687 mean, 0.125 lowest |

## Findings

1. **The one wrong verdict is a parsing defect, not a retrieval one.** Row 1073 states a
   net loss. In two of four DeepSeek runs the parser recorded the amount as positive, the
   comparison against the filed negative figure failed, and a true claim was refuted at
   0.85. Qwen read the sign correctly in all four runs. No other claim was answered wrongly
   by either model in any run, across 2,600 judged answers.
2. **The comparator decides, for better and once for worse.** Across the eight runs the
   deterministic comparison changed the model's own verdict on 193 of 2,600 answers: 169
   times to the labelled verdict, 22 times to a decline or an escalation, and twice (row
   1073) from the model's correct SUPPORTS to a wrong REFUTES, because the parsed claim it
   was handed carried the wrong sign. In 25 further answers, all on fines claims, the
   comparison set the news agent's verdict aside and the company's filing then settled the
   claim with the same verdict. Nine answers, on two restated-period claims, were
   declined because they matched a figure the issuer had since replaced.

**Stability.** Four runs per model. DeepSeek-V4.1-Flash: pass^4 95.4%, 15 misses on at
least one run, one of them the wrong verdict above. Qwen3.8: pass^4 92.0%, 26 misses, none a
wrong verdict.

Qwen's gap to DeepSeek is caution: its declines cluster on restated years, where it
reaches the step limit and escalates, and on live-price claims, where its parser
sometimes names no measure. It also lost six claims across four runs to timeouts or an
empty parser reply.

## What this does not establish

- Rows 1 to 5 and 7 are one run per model; differences of a point or two between the
  models are within the run-to-run spread in the per-run table.
- The answers come from runs two weeks apart, and no check was run that the served models
  did not change in between.
- The human-review gate has no test in this set.
- Retrieval is measured on private sets and is not recomputable here.

## Reproducing the tables

The recompute test rebuilds every run from the public artifacts, joining each 2026-09-29
run with its 2026-10-15 live-price answers:

```
OPENAI_API_KEY=any .venv/bin/python -m pytest tests/unit/test_readme_numbers_recompute.py -q
```

With the private dataset file, the layer and criteria scripts give every table:

```
OPENAI_API_KEY=any .venv/bin/python scripts/eval_layers.py --dir docs/eval/runs --label u-ds \
  --dataset <dataset file> --patch u-ds-c1=u12-ds-c1 --patch u-ds-c2=u12-ds-c2 \
  --patch u-ds-c3=u12-ds-c3 --patch u-ds-c4=u12-ds-c4
.venv/bin/python scripts/score_golden_g.py --dir docs/eval/runs --label u-ds \
  --dataset <dataset file> --patch u-ds-c1=u12-ds-c1 ...   # same pairs; u-qw likewise
```
