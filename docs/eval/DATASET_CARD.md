# FinVet golden evaluation set — dataset card

**golden_u · 325 claims · 72 companies · frozen 2026-10-12 · held out privately**
SHA-256 `f602b2fe2dfd210b8e49e28e6ae5a631c026bdd94f93aac11fa0a3f159e3a342`

## Summary

A held-out test set for the end-to-end FinVet pipeline. Each claim carries three
labels: the verdict the system should reach, the evidence path it should take, and the
parse its claim parser should emit. The claims and labels are not published, because a
released test enters training corpora and stops measuring anything. This card is the
public record of what the set contains, how it was built and sealed, and how a result
on it can be checked without seeing it.

The pre-registration card, [`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md), records the pass/fail
criteria A1–A13, committed before the set was run and scored by
`scripts/score_golden_g.py`; it is not edited after the fact.

## Uses

A run of this set measures whether FinVet reaches the labelled verdict by the labelled
path, on claims the system was not developed on. Market rows are labelled from a snapshot
quote at run start.

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
| `strength` | always `strict`: the verdict must match the label |
| `expected` | verdict, required limitation, and the evidence sources to use — so a right answer by the wrong path is detectable |
| `ground_truth` | why the label is correct (filed value and distance, or the design rule) |
| `source` | verification pointer: SEC accession + XBRL concept + period, the filing section quoted, or the decision record |
| `gold_parse` | the 7-field parse the parser should emit, with a per-row `label_source` |
| `tags` | on script-labelled rows: hardness class, the edge case the row exists for, phrasing-twin links |
| `recipe` | the 28 recipe-driven market rows: ticker, operator and offset; the runner fills the price at run start |

**By category.** `sec/xbrl` 149 · `sec/operator` 37 · `market/quote` 28 · `declined` 28 ·
`sec/tolerance` 26 · `guard` 16 · `reject` 15 · `a2a` 14 (fines via News→SEC delegation) ·
`sec/qualitative` 12.

**Hardness classes.** The generalization rows are one SUPPORTS and one REFUTES per filer
in five classes of 13 filers each, membership checked by script from SEC data:

- `non_calendar_fye`: fiscal period-end month is not December.
- `financials`: banks and insurers, whose income statements carry no gross profit or cost
  of revenue.
- `mid_cap`: market-cap tier, where XBRL tagging is sparser.
- `nonprimary_revenue`: revenue filed under `Revenues` rather than the primary concept.
- `restated`: one reporting period carries two filed values differing by more than 0.5%;
  all thirteen are spin-off or divestiture recasts, not error corrections.

## Creation

**Verdict labels.** Every numeric label carries the accession number, XBRL concept and
period it was checked against. Most were derived by script from SEC `companyfacts`, with
no person choosing the number, and re-fetched in full before the freeze; the rest were
read from the filing by the author. Behavioural labels (rejects, declines, guards) cite
the design rule that forces them.

**Parse labels** record their own provenance. No `gold_parse` was ever produced by
running the parser under test.

| `label_source` | rows | |
|---|---|---|
| `derived_from_source` | 232 | from the row's accession/concept/period; `value` always from the claim text |
| `filing_transcribed` | 10 | the fine or settlement sentence transcribed from the filing |
| `needs_review` | 67 | drafted from the claim text, not human-adjudicated |
| `n/a` | 16 | guard rows, blocked before the parser runs |

**The set** is built by `scripts/build_golden_u.py`, which refuses to write on a
duplicate id, a schema deviation, two rows with the same normalised claim text, or two
rows asking the same fact unless one is a phrasing twin of the other.

**Personal data.** None. The PII guard rows use invented identifiers.

## Bias, risks and limitations

- **Author-built test of the author's own system.** For numeric rows the filing decides;
  for behavioural rows the "correct" answer is the design's own rule, so those rows test
  conformance to the spec, not the spec. This is not independent validation. What the
  script-labelled rows add is that challenge is possible for someone else: no human
  judgment enters their numeric labels, the build and scoring scripts are published, and
  the criteria were committed before the run.
- **67 parse labels unadjudicated.** Parse-accuracy figures on those rows are
  provisional; verdict accuracy is unaffected.
- **Phrasing is tidy.** Claims were written by hand or generated from sentence
  templates. The 30 phrasing twins are the only naturally-phrased arm.
- **Small strata.** The fines rows (`a2a`, 14) and guard rows (16) exercise paths; their
  rates carry no statistical weight.
- **Market labels are live, not frozen.** The 28 live-price rows are labelled from a
  quote at run start and excluded from cross-run comparison.

## How results are checked without the data

1. **Redacted run artifacts** in [`runs/`](runs/): every per-row record of each published
   run — expected and actual verdicts, confidence, tools, timings — with the claim text
   withheld, along with the parsed ticker, value and period and the retrieved figure,
   which together would rebuild it. `scripts/redact_run.py` produces them, strips the
   dataset path and any non-public endpoint, and refuses to write a file that still
   carries either. Every published figure recomputes from these files, and CI does so on
   every run.
2. **The recipe**: labels come from public primary sources under the rules above, so an
   equivalent set can be built independently and the pipeline re-scored on it.
3. **The hash and access on request**: the SHA-256 above pins every published result to
   one file, available privately to reviewers, who verify what they receive against it.
