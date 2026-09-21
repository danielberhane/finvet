# FinVet Golden Evaluation Set — dataset card

**v1 · 97 claims · frozen 2026-09-01 · held out privately**
SHA-256 `ae0ba8bf2140cbcf98e2d4aeac003c7695a6403459606340f9f1181dc49335bf`

## Summary

A held-out test set for the end-to-end FinVet pipeline. Each of the 97
financial claims carries three labels: the verdict the system should reach,
the evidence path it should take, and the parse its claim parser should emit.

The claims and labels are **not published** — a released test enters training
corpora and stops measuring anything. This card is the public
record: what the set contains, how it was built and sealed, and how its
results can be checked without seeing it. Three rows are published in full
below and permanently retired from scoring.

## Structure

One claim per line (JSONL). Ids run 1–100 with gaps at 35/36/97 — removed
rows are never renumbered or reused.

| Field | Contents |
|---|---|
| `claim` | the input text — the only thing sent to the system under test |
| `category` | pipeline path exercised (table below) |
| `strength` | `strict` (must match) · `safe` (only the opposite verdict fails) · `observe` (recorded only) |
| `expected` | verdict, required limitation, and the evidence sources to use — so a right answer by the wrong path is detectable |
| `ground_truth` | why the label is correct (filed value and distance, or the design rule) |
| `source` | verification pointer: SEC accession + XBRL concept, or the decision record |
| `gold_parse` | the 7-field parse the parser should emit, with a per-row `label_source` |

| Category | Rows | Tests |
|---|---|---|
| `sec/xbrl` | 22 | GAAP figures vs filed XBRL facts |
| `reject` | 16 | claims that must be refused |
| `sec/qualitative` | 12 | what a filing says, from retrieved text |
| `declined` | 10 | claims declined with a stated limitation |
| `a2a` | 8 | fines/settlements via News→SEC delegation |
| `market/quote` | 8 | live prices (excluded from cross-run comparison) |
| `sec/tolerance` | 6 | values just inside / outside the comparison band |
| `sec/operator` | 6 | `>` `≥` `<` `≤` `approx` |
| `guard` | 6 | injection and PII — blocked before any model call |
| `known-defect` | 3 | observed only, never asserted |

Strength: 66 strict · 28 safe · 3 observe.

## Creation

**Verdict labels** were read from primary sources: every numeric SEC label
carries the accession number and XBRL concept it was checked against;
behavioral labels (rejects, declines, guards) cite the design rule that
forces them.

**Parse labels** record their own provenance per row:

| `label_source` | Rows | |
|---|---|---|
| `derived_from_source` | 34 | derived from the row's accession/concept/period; `value` always from the claim text, never the filed figure |
| `needs_review` | 57 | drafted from the claim text, **not yet human-adjudicated** |
| `n/a` | 6 | guard rows — blocked before the parser runs |

No `gold_parse` was ever produced by running the parser under test. Labelled
by a single annotator against primary sources.

## Considerations — biases and limitations

- **Author-designed test of the author's own system.** For numeric rows the
  ground truth is external (the filing decides). For behavioral rows the
  "correct" answer is the design's own rule — those rows test conformance to
  the spec, not the spec itself. This is not independent validation.
- **Easiest tier of filers.** Numeric claims cover a handful of US mega-caps
  with the cleanest XBRL. Results do not generalize to messier filers, and no
  such claim is made.
- **Authored phrasing.** Claims were written by the author: grammatical,
  unambiguous, one fact each. Real user input is messier; this set has no
  naturally-phrased arm.
- **57 parse labels unadjudicated** — parse-accuracy numbers are provisional;
  verdict-accuracy numbers are unaffected.
- **Small strata.** `a2a` (8) and `guard` (6) exercise paths; their
  per-category rates carry no statistical weight.
- **No canary string.** Contamination would not be detectable from model
  output; the freeze and hash mitigate but do not replace one.

## How results are checked without the data

1. **Redacted run artifacts** (beside this card): every per-row record of
   each published run (MiniMax c1; DeepSeek c1–c4) — expected and actual
   verdicts, confidence, tools, retrieved values, timings — with the claim
   text withheld. They are produced by `scripts/redact_run.py`, which also
   reduces the dataset path to its basename, replaces any non-public
   endpoint, and refuses to write a file that still carries a home path or an
   IP address. Every published metric recomputes from these files.
2. **Layer summaries** (`layers-*.json`, including the pooled
   `layers-deepseek-c1-c4.json`) and the benchmark write-up, verbatim.
3. **The recipe**: labels come from public primary sources under the rules
   above, so an equivalent set can be built independently and the pipeline
   re-scored on it.
4. **Hash commitment**: the SHA-256 above pins every published result to one
   immutable file.
5. **Access on request**: available privately to reviewers; verify what you
   receive against the hash.

## Sample rows (published in full, and therefore burned)

Per the exclusion policy (`src/finvet/eval/exclusions.py`), a row whose claim
text becomes public never counts in a scored benchmark again. Ids 1, 68, 88
are excluded from all runs after `*-c1`; they come from the three largest
categories, so the loss of coverage is smallest.

**id 1 — the happy path** (`sec/xbrl`, strict): claimed value vs the filed
XBRL fact, provenance pinning the exact filing.

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

<details>
<summary><b>id 68 — a decline with a stated reason</b> (<code>declined</code>, strict)</summary>

```json
{"id": 68,
 "claim": "US CPI inflation was 3.1 percent in July 2025",
 "category": "declined", "strength": "strict",
 "expected": {"verdict": "NOT_ENOUGH_INFO", "limitation": "unsupported_metric", "sources": []},
 "ground_truth": "SERVABLE_METRICS['news'] is an empty frozenset in Release A, so cpi_inflation is whitelisted but unservable and verification_strategy_for returns 'unsupported'. The stated reason: a macro claim names no vintage, and these series are revised, so the same claim is true or false depending on which release you read",
 "source": "src/finvet/config/metrics.py:146 SERVABLE_METRICS['news']",
 "gold_parse": {"claim_type": "news", "ticker": null, "metric": "cpi_inflation",
                "operator": "eq", "value": 3.1, "period": "July 2025",
                "reject_reason": null, "label_source": "needs_review"}}
```
</details>

<details>
<summary><b>id 88 — the tempting reject</b> (<code>reject</code>, strict): a precise
number attached to an unidentifiable entity</summary>

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
</details>

## Versions

- **v0** (2026-08-30) — 100 rows, frozen.
- **v1** (2026-09-01) — parse labels added on every row; three range claims
  removed after range support was withdrawn (D18); frozen at the hash above.
  Benchmark runs `deepseek-c1` / `minimax-c1` executed against it.
- 2026-09-04 — ids 1, 68, 88 burned by publication here.
