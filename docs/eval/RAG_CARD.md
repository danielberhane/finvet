# Filing-text retrieval (RAG) — evaluation card

**Frozen 2026-10-01 · 50 queries · corpus of 127 filings, 24,560 passages, 115 companies**
Case file: `tests/accuracy/rag_heldout_cases.json` · corpus SHA-256 `7364dc31…` (hash of every passage id)

## What is being measured

When a claim is about what a filing says, FinVet searches its passage index and hands the
five best passages to the model. This card measures that search on its own, on queries it
has never seen, with labels that need no judge. Separately, it measures whether the model's
reading of the passages stays inside them, on the 12 benchmark claims where that reading is
the answer.

Not measured here: retrieval on questions phrased by real users (the queries below were
written from the passages), the embedding model against alternatives, and anything about
numeric claims, whose verdicts are computed from XBRL and never from passage text.

## Test A — does the search find the passage? (30 queries)

**Built how.** Thirty passages drawn from the live index by a seeded sample
(`random.Random(20261001)`), stratified 10 risk factors, 8 legal proceedings, 6
management's discussion, 6 financial notes; at least 200 tokens; one company each; none of
the five companies the 70-case calibration set used (AAPL, AMZN, MSFT, NVDA, TSLA). One
drawn passage (CVX) was a bare table with no prose and was replaced by the next in seeded
order. For each passage the evaluator wrote a one-line query about its subject in different
words, avoiding the passage's distinctive terms so the keyword arm cannot win by string
match. This is known-item retrieval: the labels are exact, the phrasing is not user-like.

**Expected ids.** The passage's own id plus its immediate neighbours in the same section
(chunk index ± 1), because passages overlap by 100 tokens and the same sentences can sit
in either. 29 of 30 cases have neighbours.

**No section filter.** Positives pass only the ticker, so the test measures retrieval by
content. Five of the 30 drawn passages sit under a section label that does not match their
content (a Walmart accounting note labelled legal proceedings, for example); the case file
flags them. That is a property of the index's heading parser, recorded here as a finding.

**Scored.** recall@5 (any expected id in the top five), mean reciprocal rank, nDCG@5.
Textbook definitions, implemented in `scripts/eval_rag_heldout.py` and checked once
against `ranx` 0.3.21 on this set.

## Test B — does the search return nothing when it should? (20 queries)

- **12 off-topic** queries (recipes, sport, astronomy …), each asked against a real
  company's filings. Tests the relevance floor (0.55 cosine, calibrated on the 70-case set):
  without it the vector arm always returns a nearest neighbour.
- **8 near-miss** queries, on topic for a company the index holds, scoped by `period_end`
  or `filing_type` to a filing it does not hold. Tests the period and form filters. The
  case builder verified each filter matches zero passages.

**Scored.** Count returning zero results.

## Test C — does the model stay inside the passages? (12 claims)

The benchmark's 12 `sec/qualitative` claims (ids 37–48; all expect SUPPORTS, so the sample
has no refuting case). Each is sent through the API; the passages the SEC agent retrieved
and the model's explanation are captured from the response. Two judge-scored measures,
DeepEval's `FaithfulnessMetric` (every statement in the explanation supported by the
passages) and `AnswerRelevancyMetric` (the explanation addresses the claim), with DeepSeek
as the judge. Three of the twelve are also read by hand.

**Judge-scored, and said so.** These two figures depend on a model's reading. Tests A and B
do not. The artifact withholds the explanation text, because a response that restates a
held-out claim would publish it; it records scores, passage hashes and a hash of the text.

## Thresholds, committed before the run

| test | measure | bar |
|---|---|---|
| A | recall@5 | ≥ 26/30 (0.87) |
| A | MRR | ≥ 0.70 |
| B | off-topic returning nothing | 12/12 |
| B | near-miss returning nothing | 8/8 |
| C | faithfulness, mean | ≥ 0.85, no claim below 0.50 |

A miss is reported with its id and what came back instead. Nothing is tuned on this set.
If the floor, the chunker or the embedding model changes, this set is retired and a new one
drawn, because its corpus hash will no longer match.

## Results

*(filled after the run)*

## How to run

```
set -a; source .env; set +a                      # database and embedder settings
PYTHONPATH=src .venv/bin/python scripts/eval_rag_heldout.py       # tests A and B, ~2 min, no model
PYTHONPATH=src .venv/bin/python scripts/eval_rag_faithfulness.py  # test C, 12 API calls + judge
.venv/bin/python -m pytest tests/unit/test_rag_heldout_cases.py -q
```

The runner refuses to score a corpus whose hash differs from the frozen one, and refuses
when the embedder is not answering, since the search would then silently run on keywords
alone.
