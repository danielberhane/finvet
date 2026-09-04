# Dataset card — the FinVet golden set (`golden_c.jsonl`)

A held-out evaluation set for the end-to-end FinVet pipeline: 97 financial
claims, each labelled with the verdict the system should reach, the evidence
path it should take, and the parse its claim parser should emit.

**The claims and labels are not published.** This card is the public record of
what the set is, how it was built and sealed, and how its results can be
checked without seeing it. Rationale for the withholding is in
[`docs/EVAL_DATA_POLICY.md`](../EVAL_DATA_POLICY.md); the short version is
that a published test enters training corpora and stops measuring anything.

## Identity and integrity

| | |
|---|---|
| File | `golden_c.jsonl` (JSONL, one claim per line) |
| Rows | **97** — ids run 1–100 with gaps at 35, 36, 97 (range claims removed by design; ids are never renumbered) |
| Frozen | 2026-09-01, in its own git repository |
| SHA-256 | `ae0ba8bf2140cbcf98e2d4aeac003c7695a6403459606340f9f1181dc49335bf` |
| Canary string | **none embedded** — a known gap, disclosed rather than implied; the fingerprint above is the integrity mechanism. A canary would require modifying the frozen file and is deferred to the next revision |
| Benchmark runs pinned to it | `deepseek-c1`, `minimax-c1` (2026-08-31), both executed at this exact hash |

The hash is a commitment: if the set is ever shared privately (see *Access*),
the recipient can verify it is byte-for-byte the file the published results
came from.

## Composition

By pipeline path exercised:

| Category | Rows | What it tests |
|---|---|---|
| `sec/xbrl` | 22 | GAAP figures against filed XBRL facts |
| `reject` | 16 | claims that must be refused (questions, opinions, forecasts, unidentifiable entities) |
| `sec/qualitative` | 12 | what a filing *says*, answered from retrieved text |
| `declined` | 10 | claims the system must decline with a stated limitation |
| `a2a` | 8 | fines/settlements — News→SEC delegation against the primary source |
| `market/quote` | 8 | live price claims (excluded from cross-run comparison: live data) |
| `sec/tolerance` | 6 | values placed just inside / just outside the comparison band |
| `sec/operator` | 6 | `>`, `≥`, `<`, `≤`, `approx` comparators |
| `guard` | 6 | prompt injection and PII — must be blocked before any model call |
| `known-defect` | 3 | recorded, never asserted (`observe`) |

By assertion strength: **66 strict** (must match exactly), **28 safe** (only
the opposite verdict is a failure — the exact answer depends on live data),
**3 observe** (recorded only). Comparison operators covered: `eq` 44, `gt` 8,
`lt` 4, `gte` 1, `lte` 1, `approx` 1.

## Row schema

```
id            stable integer; gaps are deliberate, ids are never reused
claim         the input text — the only thing sent to the system under test
category      pipeline path (table above)
strength      strict | safe | observe
expected      verdict, required limitation, and the evidence sources that
              should be used (xbrl / rag / a2a) — the routing label that
              separates a right answer from a right answer by the wrong path
ground_truth  why the label is correct (filed value and distance, or the
              design rule that forces the outcome)
source        where the label was verified: SEC accession number + XBRL
              concept, or the code/decision record that defines the behavior
gold_parse    the 7-field parse the claim parser should emit, plus
              label_source (below)
```

## Label provenance and quality — stated honestly

Two independent label layers with different confidence:

**Verdict labels (`expected`)** — high confidence. Every numeric SEC label was
read from the filing itself and carries its accession number and XBRL concept;
behavioral labels (rejects, declines, guards) cite the design rule that forces
them. All 97 were exercised by two full benchmark runs.

**Parse labels (`gold_parse`)** — mixed, and the mix is recorded per row in
`label_source`:

| `label_source` | Rows | Meaning |
|---|---|---|
| `derived_from_source` | 34 | derived mechanically from the row's own provenance (ticker, concept, period from the accession; value from the claim text — never from the filed figure, which would reward hallucinating it) |
| `needs_review` | 57 | drafted from the claim text; **not yet human-adjudicated** |
| `n/a` | 6 | guard rows — blocked before the parser runs, so no parse exists to label |

Until the 57 are adjudicated, parse-accuracy numbers from this set should be
treated as provisional; verdict-accuracy numbers are not affected. No row's
`gold_parse` was ever produced by running the parser under test.

## How results are verifiable without the data

1. **Redacted run artifacts** (committed beside this card): the full per-row
   record of each benchmark run — expected verdicts, actual verdicts,
   confidence, tools called, retrieved values, timings — with only the claim
   text withheld. Labels without their stimuli are of little use for training,
   and keeping them is what lets every published table cell recompute from
   these files.
2. **Layer summaries**: the computed metrics (`layers-*.json`) as scored by
   `scripts/eval_layers.py`, which reads saved runs and calls no APIs.
3. **The recipe**: labels come from public primary sources (SEC EDGAR
   accessions, market data, FRED series) under the rules in this card, so an
   equivalent set can be built independently and the pipeline re-scored on it.
4. **The burned sample below**: three complete rows, published here in full.
5. **Access on request**: the full set is available privately to reviewers;
   verify against the SHA-256 above.

## Burned sample — three complete rows

Published in full, and therefore **burned**: per the project's exclusion
policy (`src/finvet/eval/exclusions.py`), a row whose claim text enters public
material never counts in a scored benchmark again. Ids 1, 68 and 88 are
excluded from all runs after `*-c1`. They were chosen from the three largest
categories so the loss of scored coverage is smallest.

**id 1 — the happy path (`sec/xbrl`, strict).** The claimed value is compared
against the filed XBRL fact; provenance pins the exact filing.

```json
{"id": 1,
 "claim": "Apple's total revenue was $391 billion in fiscal year 2024",
 "category": "sec/xbrl", "strength": "strict",
 "expected": {"verdict": "SUPPORTS", "limitation": null, "sources": ["xbrl"]},
 "ground_truth": "filed 391,035,000,000; period_end 2024-09-28; 0.0090% from the claim, inside the 1.5% band",
 "source": "AAPL FY2024 10-K facts (period_end 2024-09-28), accn 0000320193-25-000079, us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
 "gold_parse": {"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",
                "operator": "eq", "value": 391000000000, "period": "FY2024",
                "reject_reason": null, "label_source": "derived_from_source"}}
```

**id 68 — a decline with a stated reason (`declined`, strict).** The metric is
parseable but unservable; the system must say so rather than guess. The
label's authority is a design rule, cited.

```json
{"id": 68,
 "claim": "US CPI inflation was 3.1 percent in July 2025",
 "category": "declined", "strength": "strict",
 "expected": {"verdict": "NOT_ENOUGH_INFO", "limitation": "unsupported_metric", "sources": []},
 "ground_truth": "SERVABLE_METRICS['news'] is an empty frozenset in Release A, so cpi_inflation is whitelisted but unservable and verification_strategy_for returns 'unsupported'. The stated reason: a macro claim names no vintage, and these series are revised, so the same claim is true or false depending on which release you read",
 "source": "src/finvet/config/metrics.py:146 SERVABLE_METRICS['news']; RELEASE_A_DECISIONS.md D10",
 "gold_parse": {"claim_type": "news", "ticker": null, "metric": "cpi_inflation",
                "operator": "eq", "value": 3.1, "period": "July 2025",
                "reject_reason": null, "label_source": "needs_review"}}
```

**id 88 — the tempting reject (`reject`, strict).** A precise number attached
to an unidentifiable entity — the shape most likely to tempt a guess. The
correct behavior is refusal, and the refusal is itself a recorded decision.

```json
{"id": 88,
 "claim": "A large US bank posted $30 billion in net income last year",
 "category": "reject", "strength": "strict",
 "expected": {"verdict": "REJECTED", "limitation": null, "sources": []},
 "ground_truth": "reject_reason 'ambiguous_entity'. A precise value attached to an unidentifiable entity — the shape most likely to tempt a guess. A rejected claim returns HTTP 200 with status 'rejected' and confidence 1.0 — the rejection is itself a decision on the record",
 "source": "src/finvet/agents/prompts/parser_system.txt, reject_reason vocabulary",
 "gold_parse": {"claim_type": "reject", "ticker": null, "metric": null,
                "operator": null, "value": null, "period": null,
                "reject_reason": "ambiguous_entity", "label_source": "needs_review"}}
```

## Known limitations of this dataset

- **57 parse labels await human adjudication** (verdict labels are unaffected).
- **No canary string** — contamination of this set is not detectable from
  model output; the freeze and hash mitigate but do not replace it.
- **Single annotator.** Labels were drafted and verified by one person against
  primary sources; no inter-annotator agreement exists.
- **Small strata.** `a2a` (8), `guard` (6) and `sec/tolerance` (6) are too
  small for per-category rates to carry statistical weight; they exist to
  exercise paths, not to estimate rates.
- **US large-cap equities only**, reflecting the system's own scope.
- `market/quote` rows depend on live data and are excluded from cross-run
  comparison by design.

## Changelog

- **2026-08-30** — 100 rows, frozen as `golden_100.jsonl`.
- **2026-08-31** — `gold_parse` added to every row; `expected.status` removed
  (derivable from the verdict); benchmark runs `deepseek-c1` / `minimax-c1`
  executed.
- **2026-09-01** — three range/"between" claims removed (ids 35, 36, 97) after
  range support was withdrawn (decision D18); renamed `golden_c.jsonl`;
  frozen at the SHA-256 above.
- **2026-09-04** — ids 1, 68, 88 burned by publication in this card.
