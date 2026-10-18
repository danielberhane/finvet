# FinVet Golden Evaluation Set G — dataset card

**252 claims · frozen 2026-09-29 · held out privately**
SHA-256 `bd88c1cddc94ec793af30a4cdded7332a90761c93b5d5bc9741e847efcb2486c`

Frozen as a component of the combined run set `golden_u.jsonl`; see
[`DATASET_CARD.md`](DATASET_CARD.md) for that set.

## Summary

This card is the pre-registration record for set G, the 252-claim part of the evaluation
set, whose labels were derived from SEC filings by script. The commit that added it fixed
the pass/fail criteria A1–A13 below and the file's hash before any run of the set existed;
those sections are not edited afterwards. Composition, labelling and limitations of the
whole set are in [`DATASET_CARD.md`](DATASET_CARD.md).

## Pre-registered criteria

Computed by `scripts/score_golden_g.py`, whose thresholds are the ones written
here. The **Result** column is blank by design: the criteria were committed
before any run, and the commit is the proof the goalposts were fixed in advance.

**Baseline for A3** — set C's `sec/xbrl` outcome accuracy in the most recent
published DeepSeek artifact, `run-20260906T005622Z-deepseek-c4-redacted.json`
(beside this card): **21/21 = 1.00** over the rows with `category == "sec/xbrl"`
and a non-null expected verdict. So A3's line is **0.90 per hardness class**.
The pipeline code is unchanged from `main` at commit `cdd039e`: the branch
carrying this card touches only `scripts/`, `tests/`, `README.md`, `CLAUDE.md`,
one hook, the `benchmark-run` skill and `docs/`, never `src/finvet/` beyond
registering the burned ids below.

A3, A8 and A13 are pooled over the full run set — one rate across every
labelled run. A4 is scored **per run**: every run must individually clear
18/20, so one weak run cannot hide behind a strong one in the pool.

Burning ids 1079, 1151 and 1186 shrinks three populations below their
pre-registration counts: A3's `nonprimary_revenue` class to 25 rows, A5 to 19,
A7b's corroborated population to 3, and A13's seeded population to 53.

| # | Rule | Population | Passes if | Measured by | Result |
|---|---|---|---|---|---|
| A1 | No dangerous errors | all decisive rows | **0** verdicts assert the opposite of the truth | asymmetric-risk layer | |
| A2 | Every number traceable | all decisive numeric rows | **100%** — one untraceable number means the guard has a hole | grounding layer | |
| A3 | Generalization holds | 130 strict, non-control, non-twin, non-boundary rows — 26 per class | **every** hardness class ≥ **0.90** (baseline − 0.10). Twin and boundary seeds are excluded so they cannot lift a class; an empty class map **fails closed** rather than passing vacuously | outcome, by `tags.class` | |
| A4 | Boundary respected | 20 (`boundary_in` + `boundary_out`) | ≥ **18/20** per run | outcome | |
| A5 | Limitations enforced | 20 | **all** decline with the exact expected `limitation` | actual vs expected limitation | |
| A6 | Guard fires | 10 | **all** return HTTP 400 with zero tool calls | `http` + `tools_called` | |
| A7 | Human gate fires | 6 | **all** escalate (`escalated: true`), and each receives a recorded reviewer decision through `/review` | `escalated` in the run record; `hitl_*` audit events | |
| A7b | News path is never inverted | 4 corroborated, 2 wrong-amount, 5 uncertifiable | **no** corroborated row returns REFUTES **and** no wrong-amount row returns SUPPORTS **and** **all** uncertifiable rows carry `amount_not_certifiable`. The corroborated SUPPORTS rate is reported as `corroborated_supports: k/n` — **a measurement, not a threshold** (see the retrieval collision above) | outcome + limitation check | |
| A8 | Phrasing robustness | 30 twins vs their 30 originals | twins within **5 points** of the originals | paired outcome | |
| A9 | Confidence is honest | all decisive rows | calibration error ≤ **0.05** | calibration layer | |
| A10 | Vendor-independent safety | A1, A2, A5, A6 | hold **identically** under both LLM vendors | per-vendor layer summaries; recorded by hand | |
| A11 | Stable | all rows, k runs | pass^k ≥ **0.90** at k ≥ 3 (not asserted below k = 3) | reliability layer | |
| A12 | Market comparator works both ways | 10 `snapshot_supports` + 10 `snapshot_refutes` | **all** correct against the run's snapshot | outcome, recipe rows | |
| A13 | Comparator edge cases | 50 plain vs 54 seeded (quarter 30, operator 20, loss 4) | seeded within **5 points** of plain | outcome, by `tags.seed` | |

**When to re-run** (ongoing monitoring): on any change to the pipeline code, to
the LLM model name reported by the running service, or to a tolerance or
confidence constant. **A breach** is any rule flipping from pass to fail, or
overall accuracy falling more than 3 points below the recorded baseline.


## Sample rows (published in full, and therefore burned)

Per the exclusion policy (`src/finvet/eval/exclusions.py`), a row whose claim
text becomes public never counts in a scored benchmark again. Ids **1079**,
**1151** and **1186** are excluded from every run of this set. They are one row
from each of the three label methods, so a reader can check all three by hand.
No other row's claim text appears anywhere in this card.

**id 1079 — a derived numeric label** (`sec/xbrl`, strict, `nonprimary_revenue`
class, quarterly seed): the claimed value against the filed XBRL fact, with
provenance pinning the exact filing and the distance that decides the verdict.

```json
{"id": 1079,
 "claim": "Southern Company's total revenue was $7.74 billion in Q1 2025",
 "category": "sec/xbrl", "strength": "strict",
 "expected": {"verdict": "SUPPORTS", "limitation": null, "sources": ["xbrl"]},
 "ground_truth": "filed 7,775,000,000 (USD) period_end 2025-03-31; 0.4502% from the filed value, inside the 1.5% band",
 "source": "SO Q1 2025 10-Q accn 0000092122-25-000042, us-gaap:Revenues, frame None",
 "gold_parse": {"claim_type": "sec", "ticker": "SO", "metric": "revenue",
                "operator": "eq", "value": 7740000000, "period": "Q1 2025",
                "reject_reason": null, "label_source": "derived_from_source"},
 "tags": {"class": "nonprimary_revenue", "seed": "quarter"}}
```

<details>
<summary><b>id 1151 — a filing-transcribed fine label</b> (<code>a2a</code>, safe): the
news→SEC path, labelled from the sentence the 10-K actually states</summary>

```json
{"id": 1151,
 "claim": "Chevron paid $1.5 million in stipulated penalties under the Noble Energy DJ Basin Consent Decree",
 "category": "a2a", "strength": "safe",
 "expected": {"verdict": "SUPPORTS", "limitation": null, "sources": ["a2a"]},
 "ground_truth": "10-K FY2024 Legal Proceedings: 'The associated civil penalty was paid by Noble previously, and Chevron paid $1.5 million in stipulated penalties for noncompliance with the Consent Decree in August 2024.' A paid, resolved amount; claim matches the filed figure.",
 "source": "CVX 10-K accn 0000093410-25-000009, Item 3 Legal Proceedings",
 "gold_parse": {"claim_type": "news", "ticker": "CVX", "metric": "fine_amount",
                "operator": "eq", "value": 1500000, "period": null,
                "reject_reason": null, "label_source": "filing_transcribed"},
 "tags": {"class": "fines", "seed": "corroborated"}}
```

This is one of the two corroborated rows the probe found declining under the
retrieval collision: CVX's Item 3 also carries six threshold sentences, and
pooling across them defeats certification. The row is `safe`, so the decline
is tolerated and recorded; only a REFUTES here would fail A7b.
</details>

<details>
<summary><b>id 1186 — an authored limitation control</b> (<code>declined</code>, strict):
the fail-safe must fire <i>and name its reason</i></summary>

```json
{"id": 1186,
 "claim": "Nike's total revenue was €45 billion in fiscal 2025",
 "category": "declined", "strength": "strict",
 "expected": {"verdict": "NOT_ENOUGH_INFO", "limitation": "non_usd_amount", "sources": []},
 "ground_truth": "_NON_USD_AMOUNT matches € (euro sign) in claim_normalized; declined before any agent runs.",
 "source": "src/finvet/graph/nodes/domain_agents.py:393-404, :467",
 "gold_parse": {"claim_type": "sec", "ticker": "NKE", "metric": "revenue",
                "operator": "eq", "value": 45000000000, "period": "FY2025",
                "reject_reason": null, "label_source": "needs_review"},
 "tags": {"class": "limitation", "seed": "non_usd"}}
```

The probe confirmed this row declines with exactly this limitation in under two
seconds, with zero tool calls — before any agent runs.
</details>


## History

- 2026-09-28 — 256 rows drafted, ids 1001–1257 with one gap at **1171**: the
  fines stratum was resized from 21 rows to 20 after review dropped one
  candidate whose only reachable figure was the size of a penalty *reduction*
  rather than a penalty. The id was retired rather than reused, as ids 35/36/97
  are in set C. Criteria A1–A13 pre-registered in the same commit as this
  card, before any run of this set existed.
- 2026-09-28 — four redundant Chevron threshold-boilerplate observe rows
  (1158, 1159, 1161, 1162) removed before freeze; 1157 and 1160 kept.
- 2026-09-28 — ids 1079, 1151, 1186 burned by publication here.
- 2026-09-29 — frozen at the hash above, as a component of the combined run set
  `golden_u.jsonl` ([`DATASET_CARD.md`](DATASET_CARD.md)). The set every
  reported run uses.
