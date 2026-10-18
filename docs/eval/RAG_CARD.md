# Filing-text retrieval (RAG) — evaluation card

**Frozen 2026-10-01 · 50 queries · corpus of 127 filings, 24,560 passages, 115 companies**
Case file: `tests/accuracy/rag_heldout_cases.json` · corpus SHA-256 `7364dc31…` (hash of every passage id)

## What is being measured

When a claim is about what a filing says, FinVet searches its passage index and hands the
five best passages to the model. This card measures that search on its own, on queries it
has never seen, with labels that need no judge. Separately, it measures with RAGAS whether
the model's reading of the passages stays inside them, on 50 claims where that reading is
the answer. Not measured: retrieval on questions phrased by real users, the embedding
model against alternatives, and numeric claims, whose verdicts come from XBRL and never
from passage text.

## Test A — does the search find the passage? (30 queries)

**Built how.** Thirty passages drawn from the live index by a seeded sample
(`random.Random(20261001)`), stratified 10 risk factors, 8 legal proceedings, 6
management's discussion, 6 financial notes; one company each; none of the five companies
the 70-case calibration set used. For each passage an AI assistant wrote a one-line query
about its subject in different words, avoiding the passage's distinctive terms so the
keyword arm cannot win by string match. The queries have not been independently
reviewed. This is known-item retrieval: the labels are exact, the phrasing is not
user-like.

**Expected ids.** The passage's own id plus its immediate neighbours in the same section
(chunk index ± 1), because passages overlap by 100 tokens and the same sentences can sit
in either.

**No section filter.** Queries pass only the ticker, so the test measures retrieval by
content. Five of the 30 passages sit under a section label that does not match their
content, a property of the index's heading parser; the case file flags them.

**Scored.** hit rate@5 (any expected id in the top five), mean reciprocal rank and
nDCG@5, implemented in `scripts/eval_rag_heldout.py` and cross-checked against
`ranx` 0.3.21.

## Test B — does the search return nothing when it should? (20 queries)

- **12 off-topic** queries (recipes, sport, astronomy …), each asked against a real
  company's filings. Tests the relevance floor (0.55 cosine, calibrated on the 70-case set):
  without it the vector arm always returns a nearest neighbour.
- **8 near-miss** queries, on topic for a company the index holds, scoped by `period_end`
  or `filing_type` to a filing it does not hold. Tests the period and form filters. The
  case builder verified each filter matches zero passages.

**Scored.** Count returning zero results.

## Test C — does the model stay inside the passages? (RAGAS, 50 claims)

**Claims.** The benchmark's 12 `sec/qualitative` claims (all supported) plus 38 further
claims, frozen 2026-10-01 before any answer was collected, SHA-256 `46402ac0…996a2c`,
held privately like the benchmark set:

| kind | n | built how | expected verdict |
|---|---|---|---|
| supported | 16 | restates a drawn passage | SUPPORTS |
| contradicted | 12 | changes one named fact the passage states otherwise; no numbers | REFUTES |
| not in the filing | 10 | a plausible topic none of the company's stored passages mention | NOT_ENOUGH_INFO |

The passages were drawn by seeded sample from 10-K prose of companies outside both the
calibration five and the 30 Test A companies; absence was verified against every stored
passage of the company. The 38 were written by an AI assistant and approved by the author
before freezing.

**Method.** RAGAS 0.4.3. Each claim goes through FinVet's `/verify` unchanged; the model's
reasoning is scored against the passages it retrieved:

| RAGAS metric | question |
|---|---|
| Faithfulness | what share of the reasoning's statements do the passages support? |
| Answer relevancy | does the reasoning address the claim? |
| Context precision (without reference) | were the useful passages ranked first? |

- **Generator:** DeepSeek-V4.1-Flash, FinVet's configured model.
- **Judge:** Qwen3.8 on a university research cluster behind a LiteLLM gateway,
  temperature 0, reasoning mode off, gateway cache off. A different model family from the
  generator, so the grader has no reason to favour its writing. Each answer is judged
  twice; a faithfulness difference above 0.10 is flagged.
- **Inputs:** the claim; the model's reasoning with the summary block FinVet's code appends
  removed; the retrieved passages with FinVet's delimiters removed. Other tool outputs the
  model saw are excluded, which can only lower faithfulness.
- **Exclusions:** a claim with no retrieved passages, or whose reasoning was written by
  code, is not scored and is counted. Answers are collected and saved first, then graded,
  so the judge cannot affect them and re-grading is possible.

**Judge-scored, and said so.** These figures depend on a model's reading; Tests A and B do
not. The public artifact carries scores, verdicts and the number of passages retrieved,
and withholds claim text, reasoning, the judge's reasons and passage identities, since
each would let a reader rebuild a held-out claim.

## Rules, committed before the run

A miss is reported with its id and what came back instead. Nothing is tuned on this set.
If the floor, the chunker or the embedding model changes, this set is retired and a new one
drawn, because its corpus hash will no longer match.

## Results — 2026-10-01

Artifact: `runs/rag-heldout-2026-10-01.json`. Corpus hash matched the frozen one; the
embedder was live. A second run produced identical numbers.

| test | measure | result |
|---|---|---|
| A | hit rate@5 | **0.80** (24/30) |
| A | MRR | **0.68** |
| A | nDCG@5 | 0.49 |
| B | off-topic returning nothing | **12/12** |
| B | near-miss returning nothing | **8/8** |

In every miss the keyword arm scored zero, as the queries were written to ensure, so the
vector arm decided alone. Four of the six returned a different passage on the same subject
from another section of the same filing; two (pos-06, pos-07) returned text off the
subject: paraphrased questions about debt terms and a named dispute are not reliably found
when none of the query's words appear in the passage. nDCG is low by construction, since
the ideal ranking assumes all three overlapping neighbours are returned and the search
returns one. Nothing was re-tuned.

### Test C — faithfulness (RAGAS), 2026-10-05

Artifact: `runs/rag-ragas-2026-10-05.json`. Generator: FinVet's API, version 2.0.9, all
three roles on DeepSeek. Judge: Qwen3.8, two passes per answer. No claim was excluded.

| | all | supported | contradicted | not in the filing |
|---|---|---|---|---|
| claims scored | 49 of 50 | 27 of 28 | 12 | 10 |
| faithfulness, mean | **0.687** | 0.740 | 0.717 | 0.509 |
| faithfulness, lowest claim | **0.125** | 0.125 | 0.167 | 0.167 |
| answer relevancy, mean | 0.908 | 0.894 | 0.915 | 0.938 |
| context precision, mean | 0.793 | 0.921 | 0.765 | 0.482 |
| verdict as expected | 38 of 50 | 28 of 28 | 0 of 12 | 10 of 10 |

**One claim was not graded.** The judge repeated itself until it reached its token limit,
at temperature 0, and a retry produced no score either. It is reported as a judge failure,
not scored as zero.

**The judge was consistent, which is not the same as right.** Both passes gave the same
faithfulness on every graded claim. Whether the shortfall belongs to the model's
explanations or to the judge's reading has not been checked by hand. Not-in-filing claims
score lowest partly by construction: an explanation that a filing does not mention
something cannot be supported by the passages retrieved.

**Contradicted claims: 0 of 12 verdicts, by rule.** In all twelve the model's own
reasoning names the contradiction, and every one was returned as NOT_ENOUGH_INFO, because
the evidence model turns a REFUTES on a claim that states no number into a decline
(`non_corroboration_is_not_contradiction` in `src/finvet/models/evidence.py`). By
construction FinVet never refutes a claim about what a filing says: no false refutations,
and no true ones.

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
    --extra-claims <claims file> --record <record file>
LITELLM_API_KEY=... uv run scripts/eval_rag_ragas.py grade --record <record file>
```

The runner refuses to score a corpus whose hash differs from the frozen one, and refuses
when the embedder is not answering, since the search would then silently run on keywords
alone.
