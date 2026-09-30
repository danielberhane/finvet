# FinVet Golden Evaluation Set U — dataset card

**v1.1 · 349 claims · frozen 2026-09-29 · held out privately**
SHA-256 `fb2f98ca965492558573a489c14d34e735f6ba037b94552cf09c70f4ba3f23c2`

## What this is

The union of set C (`docs/eval/DATASET_CARD.md`, 97 rows, ids 1–100, gaps
35/36/97) and set G (`docs/eval/GOLDEN_G_CARD.md`, 252 rows, ids 1001–1257,
gaps 1158/1159/1161/1162/1171). One file, one schema, one run: `golden_u.jsonl`
is now the run set, and set C and set G are its two components rather than two
separately-run artifacts.

## Why one set

Set C is 97 author-written claims over a handful of US mega-caps with clean
XBRL. Set G is 252 rows over 71 additional filers chosen to be hard in five
named ways, with a natural-phrasing arm. The two share **no ticker in
common**. Running them as one artifact, at one code SHA, in one sitting,
answers the generalization question — does accuracy on the easy tier predict
accuracy on the harder one — inside a single, re-analysable record, instead of
two artifacts a reader has to reconcile by hand.

## How it was built

`scripts/build_golden_u.py` concatenates the two sets (golden_c read through
`$FINVET_GOLDEN_DIR`, golden_g passed as a path — golden_c is never named on a
command line, per `.claude/hooks/protect-artifacts.sh`) and refuses to write
unless every check passes:

- no duplicate ids;
- every row has exactly set C's 7-field schema plus the optional `tags` and
  `recipe` keys, and a well-formed `expected` block;
- no two rows share normalised claim text (market-recipe rows are compared by
  `ticker:offset_pct` instead, since two recipes on one ticker legitimately
  share their claim template);
- no two rows ask the same fact — `(ticker, metric, period, operator,
  value)` — twice, unless one is a phrasing twin of the other or both are
  `observe` rows (reported as a warning, not a failure, since neither is
  asserted).

Before this build, set G was itself trimmed from 256 to 252 rows: four
redundant Chevron threshold-boilerplate `observe` rows quoting the same
"$1.0 million or more" sentence (ids 1158, 1159, 1161, 1162) were dropped;
1157 (a state-regulator proceeding) and 1160 (federal) were kept. See
`docs/eval/GOLDEN_G_CARD.md`, v1.1.

## Burned rows

Six rows, published in full in the two parent cards and permanently excluded
by `src/finvet/eval/exclusions.py["golden_u.jsonl"]`: **1, 68, 88** (set C,
`docs/eval/DATASET_CARD.md`) and **1079, 1151, 1186** (set G,
`docs/eval/GOLDEN_G_CARD.md`).

## Which criteria apply where

- Set C's **strength contract** (`strict`/`safe`/`observe`) applies uniformly
  to all 349 rows and is judged by `tests/integration/test_golden.py`.
- Set G's pre-registered **A1–A13** apply to the whole file via
  `scripts/score_golden_g.py`. Its A3 baseline is no longer a fixed constant:
  it is computed from the golden_c rows (`id < 1000`, `category ==
  "sec/xbrl"`, non-null expected verdict) of the *same* run — `--baseline
  auto` (now the default; a numeric override is still accepted).

## Run procedure

Lives in the `u/` subdirectory of the private evaluation repository, run with
`--dataset golden_u.jsonl`. Labels are vendor-free — `u-ds-c1`, `u-qw-c1` — for
the same reason as set G's: the README recompute test pools every `docs/eval`
artifact whose name contains a vendor. Market rows are run after US market
close. Six rows (set G's fail-closed rows) pause for human review; each is
decided through `POST /review/{request_id}` before the run is scored, and
pending reviews do not survive an API restart.

## What the old golden_c runs are now

**Superseded, not deleted.** The published `golden_c`-only run artifacts
remain exactly as published, at their own recorded SHA, and are not
retroactively wrong. But they are not comparable row-for-row to a
`golden_u` run without filtering `id < 1000` first — a union run's per-row
records for ids 1–100 are the same claims, the same expectations, but sit
inside a larger artifact with set G's rows interleaved by id.

## Composition

| | Set C | Set G | Union |
|---|---|---|---|
| Rows | 97 | 252 | 349 |
| Id range | 1–100 (gaps 35/36/97) | 1001–1257 (gaps 1158/1159/1161/1162/1171) | both |
| Strength | 66 strict · 28 safe · 3 observe | 210 strict · 31 safe · 11 observe | 276 strict · 59 safe · 14 observe |
| Distinct tickers | 5 (AAPL, AMZN, MSFT, NVDA, TSLA) | 71 | 76, none shared |

## Versions

- **v1** (2026-09-28) — 349 rows: 97 from set C + 252 from set G (post-trim).
  Built by `scripts/build_golden_u.py`; all checks passed.
- **v1.1** (2026-09-29) — built from set G v1.2. Same 349 rows and ids. Frozen
  at the hash above; the version every reported run uses.

## Before a paid run

`scripts/replay_fact_selection.py` replays every row whose evidence is an XBRL
fact (214 of them) through the production path with no model: period
resolution, the statement tool, the observation resolver, the comparator. It
costs nothing and takes about five minutes. A run is not started until it
reports no unexplained miss and no row whose result depends on the filing
opened.
