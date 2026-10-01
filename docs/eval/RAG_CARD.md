# Filing-text retrieval (RAG) — evaluation card

**Frozen 2026-10-01 · 50 queries · corpus of 127 filings, 24,560 passages, 115 companies**
Case file: `tests/accuracy/rag_heldout_cases.json` · corpus SHA-256 `7364dc31…` (hash of every passage id)

## What is being measured

When a claim is about what a filing says, FinVet searches its passage index and hands the
five best passages to the model. This card measures that search on its own, on queries it
has never seen, with labels that need no judge. Separately, it measures with RAGAS whether the model's
reading of the passages stays inside them, on 50 claims where that reading is the answer.

Not measured here: retrieval on questions phrased by real users (the queries below were
written from the passages), the embedding model against alternatives, and anything about
numeric claims, whose verdicts are computed from XBRL and never from passage text.

## Test A — does the search find the passage? (30 queries)

**Built how.** Thirty passages drawn from the live index by a seeded sample
(`random.Random(20261001)`), stratified 10 risk factors, 8 legal proceedings, 6
management's discussion, 6 financial notes; at least 200 tokens; one company each; none of
the five companies the 70-case calibration set used (AAPL, AMZN, MSFT, NVDA, TSLA). One
drawn passage (CVX) was a bare table with no prose and was replaced by the next in seeded
order. For each passage an AI assistant (Claude) wrote a one-line query about its subject in
different words, avoiding the passage's distinctive terms so the keyword arm cannot win by
string match. The queries have not been independently reviewed. This is known-item retrieval: the labels are exact, the phrasing is not user-like.

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

## Test C — does the model stay inside the passages? (RAGAS, 50 claims)

**Claims.** The benchmark's 12 `sec/qualitative` claims (ids 37–48, all supported) plus 38
further claims (ids 2001–2038), frozen 2026-10-01 before any answer was collected, SHA-256
`46402ac0…996a2c`, held privately like the benchmark set:

| kind | n | built how | expected verdict |
|---|---|---|---|
| supported | 16 | restates a drawn passage | SUPPORTS |
| contradicted | 12 | changes one named fact the passage states otherwise; no numbers | REFUTES |
| not in the filing | 10 | a plausible topic none of the company's stored passages mention | NOT_ENOUGH_INFO |

The 28 passages behind the first two kinds were drawn by `random.Random(20261002)` from 10-K
prose of 28 companies outside both the calibration five and the 30 Test A companies; the 10
not-in-filing companies follow in the same seeded order (one replaced by rule: a digit in the
company name could parse as a claimed value). Absence was verified against every stored
passage of the company. The 38 were written by an AI assistant (Claude) and approved by the
author before freezing.

**Method.** RAGAS 0.4.3. Each claim goes through FinVet's `/verify` unchanged; the model's
reasoning is scored against the passages it retrieved:

| RAGAS metric | question | |
|---|---|---|
| Faithfulness | what share of the reasoning's statements do the passages support? | barred |
| Answer relevancy | does the reasoning address the claim? | reported only |
| Context precision (without reference) | were the useful passages ranked first? | reported only |

- **Generator:** DeepSeek-V4.1-Flash, FinVet's configured model.
- **Judge:** Qwen3.8 on the GMU ORC gateway, temperature 0, reasoning mode off, gateway cache
  off. A different model family from the generator, so the grader has no reason to favour its
  writing. Each answer is judged twice; a faithfulness difference above 0.10 is flagged.
- **Embedder** (answer relevancy only): `nomic-embed-text`, FinVet's own retrieval embedder.
- **Inputs:** the claim; the model's reasoning with the summary block FinVet's code appends
  removed; the retrieved passages with FinVet's delimiters removed. Other tool outputs the
  model saw are excluded, which can only lower faithfulness.
- **Exclusions:** a claim with no retrieved passages, or whose reasoning was written by code,
  is not scored and is counted. Not-in-filing claims may legitimately retrieve nothing.
- **Two phases:** answers are collected first and saved privately, then graded. The generator
  and the judge are reachable on different networks, and freezing the answers before grading
  keeps the judge from affecting them and allows re-grading later.

**Judge-scored, and said so.** These figures depend on a model's reading; Tests A and B do not.
The public artifact carries scores, verdicts and passage hashes, not claim text, reasoning or
the judge's reasons, all of which restate held-out claims.

## Thresholds, committed before the run

| test | measure | bar |
|---|---|---|
| A | hit rate@5 | ≥ 26/30 (0.87) |
| A | MRR | ≥ 0.70 |
| B | off-topic returning nothing | 12/12 |
| B | near-miss returning nothing | 8/8 |
| C | faithfulness (RAGAS), mean over scored claims | ≥ 0.85, no claim below 0.50 |

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

### Test C — faithfulness (RAGAS)

*(filled after the run; the bar above was committed before any answer was collected)*

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
.venv/bin/python -m pytest tests/unit/test_rag_heldout_cases.py tests/unit/test_rag_ragas_inputs.py -q
```

Test C runs in two phases; the script carries its own pinned dependencies (`uv run` builds
them, the project environment is untouched):

```
FINVET_GOLDEN_DIR=... uv run scripts/eval_rag_ragas.py answer \
    --extra-claims notes/rag_ragas_claims_v1.jsonl --record notes/rag-ragas-record.json
LITELLM_API_KEY=... uv run scripts/eval_rag_ragas.py grade --record notes/rag-ragas-record.json
```

The runner refuses to score a corpus whose hash differs from the frozen one, and refuses
when the embedder is not answering, since the search would then silently run on keywords
alone.

## Changelog

- **2026-10-01** — Test C re-specified for RAGAS on 50 claims, before any answer was
  collected. An earlier Test C run the same day used DeepEval with DeepSeek as both generator
  and judge, scored FinVet's full explanation including its code-appended summary, and covered
  12 claims. It is superseded and removed from the tree; it remains in git history.
