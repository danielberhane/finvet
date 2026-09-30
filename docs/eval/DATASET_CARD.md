# FinVet golden evaluation set — dataset card

**golden_u v1.1 · 349 claims · 76 companies · frozen 2026-09-29 · held out privately**
SHA-256 `fb2f98ca965492558573a489c14d34e735f6ba037b94552cf09c70f4ba3f23c2`

## Summary

A held-out test set for the end-to-end FinVet pipeline. Each claim carries three
labels: the verdict the system should reach, the evidence path it should take, and the
parse its claim parser should emit. The claims and labels are not published, because a
released test enters training corpora and stops measuring anything. This card is the
public record of what the set contains, how it was built and sealed, and how a result
on it can be checked without seeing it.

The set is the union of two parts built at different times and sharing no ticker:

| | Set C | Set G | golden_u |
|---|---|---|---|
| Rows | 97 | 252 | 349 |
| Ids | 1–100 (gaps 35/36/97) | 1001–1257 (gaps 1158/1159/1161/1162/1171) | both |
| Filers | 5 mega-caps (AAPL, AMZN, MSFT, NVDA, TSLA) | 71, chosen to be hard | 76 |
| Claims written by | the author | a script, from the filer's own XBRL | |
| Strength | 66 strict · 28 safe · 3 observe | 210 strict · 31 safe · 11 observe | 276 · 59 · 14 |

Set G's own card, [`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md), is the pre-registration
record: its pass/fail criteria A1–A13 were committed before the set was run, and it is
not edited after the fact.

## Uses

A run of this set measures whether FinVet reaches the labelled verdict by the labelled
path, on claims the system was not developed on. Six rows pause for human review and are
decided through the review endpoint before scoring. Market rows are run after US market
close and labelled from a snapshot quote at run start.

Out of scope: the set does not measure retrieval on its own (that has separate cases),
resistance to novel attacks (guard rows test wiring, not adversaries), or anything about
non-US filers, multi-assertion claims or historical share prices, none of which it
contains.

## Structure

One claim per line (JSONL). Removed ids are never renumbered or reused.

| Field | Contents |
|---|---|
| `id` | stable row id |
| `claim` | the input text — the only thing sent to the system under test |
| `category` | pipeline path exercised |
| `strength` | `strict` (must match) · `safe` (only the opposite verdict fails) · `observe` (recorded only) |
| `expected` | verdict, required limitation, and the evidence sources to use — so a right answer by the wrong path is detectable |
| `ground_truth` | why the label is correct (filed value and distance, or the design rule) |
| `source` | verification pointer: SEC accession + XBRL concept + period, the filing section quoted, or the decision record |
| `gold_parse` | the 7-field parse the parser should emit, with a per-row `label_source` |
| `tags` | set G only: hardness class, the edge case the row exists for, phrasing-twin links |
| `recipe` | set G's 20 market rows: ticker, operator and offset; the runner fills the price at run start |

**Set C by category.** `sec/xbrl` 22 · `reject` 16 · `sec/qualitative` 12 · `declined` 10 ·
`a2a` 8 (fines via News→SEC delegation) · `market/quote` 8 · `sec/tolerance` 6 ·
`sec/operator` 6 · `guard` 6 · `known-defect` 3 (observed, never asserted).

**Set G by stratum.** Generalization 130 (65 filers × one SUPPORTS and one REFUTES, in
five hardness classes: non-calendar fiscal years, banks and insurers, mid-caps,
non-primary revenue concepts, restated periods) · boundary 20 · fines and settlements 16 ·
limitation controls 20 · guard 10 · fail-closed 6 · live-price recipes 20 · phrasing
twins 30. Class membership rules and per-stratum detail are in the G card.

## Creation

**Verdict labels.** Every numeric label carries the accession number, XBRL concept and
period it was checked against. Set C's were read from the filing by the author; set G's
were derived by script from SEC `companyfacts`, with no person choosing the number, and
re-fetched in full before the freeze. Behavioural labels (rejects, declines, guards) cite
the design rule that forces them.

**Parse labels** record their own provenance. No `gold_parse` was ever produced by
running the parser under test.

| `label_source` | Set C | Set G | |
|---|---|---|---|
| `derived_from_source` | 34 | 200 | from the row's accession/concept/period; `value` always from the claim text |
| `filing_transcribed` | — | 16 | the fine or settlement sentence transcribed from the filing |
| `needs_review` | 57 | 26 | drafted from the claim text, not human-adjudicated |
| `n/a` | 6 | 10 | guard rows, blocked before the parser runs |

**The union** is built by `scripts/build_golden_u.py`, which refuses to write on a
duplicate id, a schema deviation, two rows with the same normalised claim text, or two
rows asking the same fact unless one is a phrasing twin of the other.

**Personal data.** None. The PII guard rows use invented identifiers.

## Bias, risks and limitations

- **Author-built test of the author's own system.** For numeric rows the filing decides;
  for behavioural rows the "correct" answer is the design's own rule, so those rows test
  conformance to the spec, not the spec. This is not independent validation. What set G
  adds is that challenge is possible for someone else: no human judgment enters a numeric
  label, the build and scoring scripts are published, and the criteria were committed
  before the run.
- **83 parse labels unadjudicated** (57 in set C, 26 in set G). Parse-accuracy figures on
  those rows are provisional; verdict accuracy is unaffected.
- **Phrasing is tidy.** Set C was written by hand and set G's numeric claims come from
  sentence templates. The 30 phrasing twins are the only naturally-phrased arm.
- **Small strata.** Set C's `a2a` (8) and `guard` (6), and set G's fines stratum (16 rows,
  6 decisive, concentrated on two filers), exercise paths; their rates carry no
  statistical weight.
- **Market labels are live, not frozen.** The 28 live-price rows are labelled from a
  quote at run start and excluded from cross-run comparison.
- **The restated class tests recasts**, not error corrections: all 13 qualifying filers
  are spin-off or divestiture recasts. No 10-K/A appears in the set.
- **Guard rows test wiring, not resistance.** All 16 match documented patterns.
- **Data gaps**: non-US filers; numeric news claims other than fines; historical share
  prices as an answerable question; claims with more than one assertion; news claims that
  name a period; accounting restatements as distinct from recasts.
- **No canary string.** Contamination would not be detectable from model output; the
  freeze and the published hash mitigate but do not replace one.

Set G's card carries the longer form of each point, and the diagnoses behind them.

## Burned rows

A row whose claim text becomes public never counts in a scored run again
(`src/finvet/eval/exclusions.py`). Six rows are burned: **1, 68, 88** (set C, published
below) and **1079, 1151, 1186** (set G, published in its card). A scored run is therefore
343 rows, 329 of them with an expected verdict.

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

## How results are checked without the data

1. **Redacted run artifacts** in [`runs/`](runs/): every per-row record of each published
   run — expected and actual verdicts, confidence, tools, retrieved values, timings — with
   the claim text withheld. `scripts/redact_run.py` produces them, strips the dataset path
   and any non-public endpoint, and refuses to write a file that still carries either.
   Every published figure recomputes from these files, and CI does so on every run.
2. **Layer summaries** (`runs/layers-*.json`) and the [evaluation report](README.md).
3. **The recipe**: labels come from public primary sources under the rules above, so an
   equivalent set can be built independently and the pipeline re-scored on it.
4. **Hash commitment**: the SHA-256 above pins every published result to one file.
5. **Access on request**: available privately to reviewers; verify what you receive
   against the hash.

## Which criteria apply where

- The **strength contract** (`strict`/`safe`/`observe`) applies to all 349 rows and is
  judged by `tests/integration/test_golden.py`.
- The pre-registered **A1–A13** apply to the whole file via `scripts/score_golden_g.py`.
  A3's baseline is computed from the set C `sec/xbrl` rows of the same run
  (`--baseline auto`).

## Run procedure

Run with `--dataset golden_u.jsonl` from the private evaluation repository, with
vendor-free labels (`u-ds-c1`, `u-qw-c1`). Before any paid run,
`scripts/replay_fact_selection.py` replays the 214 XBRL-backed rows through the
production path with no model and must report no unexplained miss. Market rows run after
US market close; the six fail-closed rows are decided through `POST /review/{request_id}`
before scoring. The full procedure is `.claude/skills/benchmark-run/SKILL.md`.

## Versions

- **Set C v1** (2026-09-01) — 97 rows, SHA-256 `ae0ba8bf…9335bf`. Ids 1, 68, 88 burned
  2026-09-04.
- **Set G v1.2** (2026-09-29) — 252 rows, SHA-256 `bd88c1cd…`; history in its card.
- **golden_u v1** (2026-09-28) — 349 rows, the union.
- **golden_u v1.1** (2026-09-29) — built from set G v1.2, same 349 rows and ids, frozen
  at the hash above. The version every reported run uses.
