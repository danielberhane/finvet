# Cross-model benchmark — 2026-08-31

Same 97 claims, same code, same dataset. Nothing changed between the two runs.

    code    finvet-v2.0.9 @ 90d1f85
    data    golden_c.jsonl @ c46f823   (97 rows, 7-field parser contract)

| | MiniMax-M2.7 | deepseek-chat |
|---|---|---|
| artifact | `run-20260831T200936Z-minimax-c1.json` | `run-20260831T210242Z-deepseek-c1.json` |
| endpoint | self-hosted LiteLLM gateway | api.deepseek.com |
| elapsed | 47.7 min | 17.2 min |
| outcome, all scored (n=93) | 87 (93.5%) | 90 (96.8%) |
| **outcome, excl. live market (n=86)** | **83 (96.5%)** | **83 (96.5%)** |
| dangerous errors | 0 / 94 | 0 / 94 |
| grounding | 42/42 (100%) | 42/42 (100%) |
| trajectory (DeepEval) | 55/58 (94.8%) | 60/61 (98.4%) |
| lane violations | 0 | 0 |
| zero-tool discipline | 26/26 | 26/26 |
| calibration, decisive ECE | 0.0456 | 0.0389 |
| routing | 90/93 (96.8%) | 92/94 (97.9%) |
| declined | 7 (7.4%) | 3 (3.2%) |

## Rows where the models disagree (live-market excluded)

| id | strength | expected | MiniMax | DeepSeek |
|---|---|---|---|---|
| 17 | strict | REFUTES | REFUTES | PENDING |
| 22 | safe | REFUTES | REFUTES | PENDING |
| 51 | safe | SUPPORTS | PENDING | SUPPORTS |
| 81 | strict | REJECTED | NOT_ENOUGH_INFO | REJECTED |

Two each. Ids 17 and 22 are the rows four earlier DeepSeek runs already flagged
as unstable (Llama Guard S6 false positive), so DeepSeek reproduced its own
known flake rather than showing a capability difference.

## Findings

**The models tie once the live category is removed.** The 3.3-point raw gap is
Finnhub: it returned 502s throughout the MiniMax run and had recovered by the
DeepSeek run. Row 62 errored at 600s under MiniMax and answered in 11.5s under
DeepSeek — same claim, same code, different vendor weather.

**Five properties held identically under both models**: zero dangerous errors,
100% grounding, zero lane violations, 26/26 zero-tool discipline, and perfect
accuracy in the 0.9 confidence bin. These are enforced by the deterministic
layer, and their invariance across two very different models is the evidence
that the scaffolding rather than the LLM is doing the safety work.

**The routing layer earned its place.** Id 51 under DeepSeek returned the
expected verdict while retrieving from `rag` where `a2a` was expected — a
correct answer by the wrong path, invisible to outcome accuracy. MiniMax
escalated the same row instead, so the silent form is DeepSeek-specific.
Id 55 fails under both: `search_filing_text` is never called.

## What this does not establish

One run per model. `pass^1` is pass@1 restated and says nothing about
stability; the measured noise floor from four earlier runs was pass^4 = 0.975,
so a four-row difference across 86 sits inside it. Ranking the models would take
roughly three runs each.

## Known environmental defect

Finnhub degraded mid-benchmark (502s, 6-11s latency against a 10s client
timeout). Each call is bounded but the retry loop is not, so a market row can
exhaust the agent's iteration budget. Affected ids 58, 61, 62, 63 under MiniMax.
Not fixed during the run, to keep the two runs comparable.
