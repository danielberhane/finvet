# FinVet golden evaluation set — dataset card

**golden_u v1.1 · 343 claims · 76 companies · frozen 2026-09-29 · held out privately**
SHA-256 `fb2f98ca965492558573a489c14d34e735f6ba037b94552cf09c70f4ba3f23c2`

## Summary

A held-out test set for the end-to-end FinVet pipeline. Each claim carries three
labels: the verdict the system should reach, the evidence path it should take, and the
parse its claim parser should emit. The claims and labels are not published, because a
released test enters training corpora and stops measuring anything. This card is the
public record of what the set contains, how it was built and sealed, and how a result
on it can be checked without seeing it.

| | |
|---|---|
| Claims | 343 |
| Filers | 76 US-listed companies |
| Scored | 329 carry an expected verdict; 14 are recorded only |
| Strength | 271 strict · 58 safe · 14 observe |

The pre-registration card, [`GOLDEN_G_CARD.md`](GOLDEN_G_CARD.md), records the pass/fail
criteria A1–A13, committed before the set was run; it is not edited after the fact.

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
| `tags` | on script-labelled rows: hardness class, the edge case the row exists for, phrasing-twin links |
| `recipe` | the 20 recipe-driven market rows: ticker, operator and offset; the runner fills the price at run start |

**By category.** `sec/xbrl` 149 · `sec/operator` 37 · `market/quote` 28 · `declined` 28 ·
`sec/tolerance` 26 · `a2a` 18 (fines via News→SEC delegation) · `guard` 16 · `reject` 15 ·
`known-defect` 14 (observed, never asserted) · `sec/qualitative` 12.

**Strata.** Generalization pairs (one SUPPORTS and one REFUTES per filer, in five
hardness classes: non-calendar fiscal years, banks and insurers, mid-caps, non-primary
revenue concepts, restated periods), tolerance boundary, fines and settlements,
limitation controls, guard, fail-closed, live-price recipes and phrasing twins. Class
membership rules are in the pre-registration card.

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
| `filing_transcribed` | 15 | the fine or settlement sentence transcribed from the filing |
| `needs_review` | 80 | drafted from the claim text, not human-adjudicated |
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
- **80 parse labels unadjudicated.** Parse-accuracy figures on those rows are
  provisional; verdict accuracy is unaffected.
- **Phrasing is tidy.** Claims were written by hand or generated from sentence
  templates. The 30 phrasing twins are the only naturally-phrased arm.
- **Small strata.** The fines rows (`a2a`, 18) and guard rows (16) exercise paths; their
  rates carry no statistical weight.
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

The pre-registration card carries the longer form of each point, and the diagnoses
behind them.

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

- The **strength contract** (`strict`/`safe`/`observe`) applies to all 343 claims and is
  judged by `tests/integration/test_golden.py`.
- The pre-registered **A1–A13** apply via `scripts/score_golden_g.py`. A3's baseline is
  computed from the same run (`--baseline auto`).

## Run procedure

Run with `--dataset golden_u.jsonl` from the private evaluation repository, with
vendor-free labels (`u-ds-c1`, `u-qw-c1`). Before any paid run,
`scripts/replay_fact_selection.py` replays the XBRL-backed rows through the production
path with no model and must report no unexplained miss. Market rows run after US market
close; the six fail-closed rows are decided through `POST /review/{request_id}` before
scoring. The full procedure is `.claude/skills/benchmark-run/SKILL.md`.

## Versions

- **golden_u v1.1** (2026-09-29) — 343 evaluated claims, frozen at the hash above. The
  version every reported run uses.
