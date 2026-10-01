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

**Scored.** hit rate@5 (any expected id in the top five; the pre-registration commit called
this recall@5 with the same definition, and the name was corrected when a cross-check with
`ranx` 0.3.21 showed that library's recall divides by the number of expected ids), mean
reciprocal rank, nDCG@5. Implemented in `scripts/eval_rag_heldout.py`; MRR and nDCG
matched ranx to four decimals, hit rate matched its `hit_rate@5`.

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
as the judge. The judge's reasons for three claims (the two lowest and one perfect) were read
by hand to confirm each deduction names a concrete, checkable fault.

**Judge-scored, and said so.** These two figures depend on a model's reading. Tests A and B
do not. The artifact withholds the explanation text, because a response that restates a
held-out claim would publish it; it records scores, passage hashes and a hash of the text.
The judge's reasons are withheld for the same reason: they paraphrase the claim.

## Thresholds, committed before the run

| test | measure | bar |
|---|---|---|
| A | hit rate@5 | ≥ 26/30 (0.87) |
| A | MRR | ≥ 0.70 |
| B | off-topic returning nothing | 12/12 |
| B | near-miss returning nothing | 8/8 |
| C | faithfulness, mean | ≥ 0.85, no claim below 0.50 |

A miss is reported with its id and what came back instead. Nothing is tuned on this set.
If the floor, the chunker or the embedding model changes, this set is retired and a new one
drawn, because its corpus hash will no longer match.

## Results — 2026-10-01

Artifact: `runs/rag-heldout-2026-10-01.json`. Corpus hash matched the frozen one; the
embedder was live. A second run produced identical numbers.

| test | measure | bar | result | |
|---|---|---|---|---|
| A | hit rate@5 | ≥ 0.87 | **0.80** (24/30) | fail |
| A | MRR | ≥ 0.70 | **0.68** | fail |
| A | nDCG@5 | — | 0.49 | reported only |
| B | off-topic returning nothing | 12/12 | **12/12** | pass |
| B | near-miss returning nothing | 8/8 | **8/8** | pass |

**The six misses.** In every one, the keyword arm scored zero (expected: the queries avoid
the passages' own words), so the vector arm decided alone. Four returned a different passage
on the same subject from another section of the same filing: Goldman Sachs's anti-money-
laundering text from Business rather than Risk Factors (pos-09); Wendy's franchise counts from
Business rather than the MD&A overview (pos-24); Snap's AI-regulation risk from a neighbouring
risk factor (pos-04); AES's Chilean operations from Business rather than the legal proceeding
(pos-16). A reader given those passages would find the subject, not the labelled passage. Two
are wrong in substance: CF's debt question returned impairment and fertilizer-market text
(pos-06); Disney's carriage dispute returned reputation and intellectual-property text (pos-07).

### Test C — faithfulness (judge-scored)

Artifact: `runs/rag-faithfulness-2026-10-01.json`. All 12 claims returned SUPPORTS with
four to six passages each; all 12 were scored.

| measure | bar | result | |
|---|---|---|---|
| faithfulness, mean | ≥ 0.85 | **0.974** | pass |
| faithfulness, lowest claim | ≥ 0.50 | **0.80** | pass |
| answer relevancy, mean | — | 0.971 | reported only |

Ten claims scored 1.00 on faithfulness. The two deductions are real faults, not judge noise:
in one, the explanation added a sentence about how the company competes that the passages
do not contain (0.80); in the other, it assigned two shipping-cost figures to the wrong years
(0.89). Neither changed the verdict, which was correct in both, but the second is the kind of
misreading the deterministic comparator exists to prevent on numeric claims, and here it sits
in prose the comparator does not check.

**Reading of A and B.** Test B passes cleanly: the floor and the filters refuse what they should. Test A
misses its bar by two cases. Known-item labelling with one passage per query counts a
same-subject passage from elsewhere in the filing as a miss, so the 0.80 understates what a
reader experiences, but it is the honest figure for "did the search return the labelled
passage". nDCG is low by construction: the ideal ranking assumes all three overlapping
neighbours are returned, and the search returns one. The two substantive misses are the
finding: paraphrased questions about debt terms and a named dispute are not reliably found
when none of the query's words appear in the passage. Nothing was re-tuned.

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
