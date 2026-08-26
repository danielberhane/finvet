# Release A — recorded decisions

Where an acceptance criterion was ambiguous, Release A takes the fail-closed
reading. Each decision below names what the system now declines to do, why, and
what would have to be built to lift the restriction. This file exists so the
narrowing is discoverable rather than buried in a diff.

The general rule: a comparison that *looks* deterministic while resting on an
unverified premise is worse than no comparison, because it produces a decisive
verdict a reader has no reason to doubt.

---

## D1 — Q4 is not derived

**Decision.** Q4 numeric claims are declined. The system does not compute
Q4 from an annual figure minus a nine-month cumulative.

**Why.** The arithmetic is only valid when both figures come from the same
restatement generation, cover the same entity scope, and use the same fiscal
calendar. None of those are checked, and a derived number carries no filing,
no accession, and no XBRL concept — so it cannot be located in a source or
audited afterwards. A derived value entering the comparator is indistinguishable
from a filed one.

**What was removed.** The instruction to derive Q4 appeared in two places that
shaped model behaviour: the verdict-extraction prompt in `agents/base.py` and
the `search_filings` docstring in `tools/sec_tools.py`. Both are gone; the
docstring now states the limitation instead. No public file advertises Q4
support.

**To lift it.** Deterministic derivation with restatement-generation matching,
entity-scope checks, and a provenance record naming both source filings.

---

## D2 — A period-bound claim on a route that resolves no period is declined

**Decision.** When a claim names a period and no canonical period was
resolved for that route, the numeric comparison is skipped and the verdict
fails closed to `NOT_ENOUGH_INFO`. The agent evidence records
`temporal_status: "unresolved_period"`.

**Why.** Only the SEC route runs `period_resolver`, so `canonical_period` is
`None` on the market and news routes and the period check in
`resolve_trusted_observation` never applied there. A claim naming a specific
day was compared against whatever the tool returned for *today*, with nothing
checking that the two referred to the same date.

**Blast radius.** Numeric market and news claims that name a period.
Non-numeric claims never reach the comparator and are unaffected. Current price
claims parse with `period: null` and remain supported. Market-cap, P/E and
other company-overview metrics are declined for a *different* reason — they
carry no source observation time — which is D10, not this decision.

**To lift it.** Per-source temporal matching: trading-day alignment for
historical prices, a freshness window for real-time quotes, and event-date
handling for news — each with its own tolerance, calibrated rather than assumed.

---

## D3 — Evidence must name its period

**Decision.** When a claim is period-bound, a fact carrying no period at all is
not trusted evidence.

**Why.** The guard read "expected and observed and they differ", so it only
rejected evidence already labelled well enough to be checked. A fact with no
period passed unexamined — the least trustworthy case treated as the most.

**A period mismatch skips the record, it does not end the search.** The
resolver continues to the next tool result rather than returning None. Both
reject wrong-period evidence; only the second additionally makes a correct,
available fact unreachable whenever a wrong-period one happens to be scanned
first. Which tools an agent called, and in what order, is not a safety
property, and correctness should not depend on it.

---

## D4 — An explicitly unconsolidated fact is not evidence

**Decision.** A fact the issuer marked `consolidated: false` is not used for
a company-level claim.

**Why.** A segment or subsidiary figure is not the entity-wide number a claim
asks about. `sec_edgar` tracks the distinction per fact; `_format_financial_items`
dropped it before it reached the trust boundary, so the two were
indistinguishable downstream. Absent (rather than false) still passes: XBRL
concepts that are entity-wide by definition do not carry the flag.

---

## D5 — A malformed range is declined, not repaired

**Decision.** A range with inverted bounds, or a range operator with one or
both bounds missing, rejects the claim. Stray bounds on a *non*-range operator
are still dropped.

**Why.** Both former repairs answered a question nobody asked. Swapping the
bounds of "between $100B and $50B" picks one of two readings and then returns
a decisive verdict on the guess. Downgrading a bandless range to `approx`
turns a membership question into a point comparison, so a filed value squarely
inside the intended band can come back `REFUTES`. Dropping stray bounds is
different in kind: nothing was asserted about an interval, so removing them
takes away noise rather than changing the claim.

---

## D6 — A failed comparator does not hand control back to the model

**Decision.** When `_apply_override` raises, the verdict fails closed to
`NOT_ENOUGH_INFO` with confidence capped at the HITL threshold, and the
evidence records `execution_status: "failed"` with the error.

**Why.** The handler used to restore the model's own verdict. That made the one
component whose job is to overrule the model hand control back to it at exactly
the moment it broke — releasing a decisive verdict *because* deterministic
verification had failed.

---

## D7 — Persistent checkpointing is out of scope, and documented as such

**Decision.** Release A keeps `MemorySaver`. Paused HITL claims do not survive
an API restart.

**Why.** Persistent PostgreSQL checkpointing is a separate piece of work with
its own migration and failure modes. The limitation is stated plainly in the
architecture documentation rather than glossed; no public file claims durable
pause/resume.

---

## D8 — Similar-claim memory is experimental and disabled by default

**Decision.** `enable_claim_memory` defaults to `false`. The default
verification flow does not search, reuse, or inject previous verifications.
`ENABLE_CLAIM_MEMORY=true` still enables it, and the implementation, both
routes and the server-side request-ID resolution all remain in the repository.

**Why.** Claim memory is the one subsystem whose output is prior *model* output
rather than a source. Reusing a cached verdict, or feeding a past summary back
to an agent, moves something the system said into the position of something the
system found. Enabling it does not upgrade cached model output into trusted
source evidence, and nothing in the trust boundary treats it as such — but the
safer default is off, and a default is enforcement in a way a paragraph is not.

**What the disabled path does.** `/memory-check` answers `{"matches": []}`, and
the UI routes an empty result straight to verification with no cached-result or
prior-context card. Naming a specific prior episode while memory is disabled is
a 404, not a silent pass: the caller asked for particular context, and
proceeding without it would verify a different question.

---

## D9 — Filing RAG is supporting textual evidence

**Decision.** RAG stays enabled for attributable filing excerpts. A retrieved
passage cannot become a `TrustedObservation` for deterministic numeric
comparison, and an empty result does not prove a disclosure is absent.

**How it is enforced.** `search_filing_text` is in `SUPPORTING_EVIDENCE_TOOLS`,
so `resolve_trusted_observation` skips its records regardless of payload shape;
every result carries `evidence_role="supporting"`. An empty search reports
`reason="no_relevant_evidence"` when the issuer has indexed filings and
`reason="no_corpus"` when it does not — a filing that was read and said
nothing, and no filing to read, are opposite facts.

**What retrieval must carry.** Form, part, item, period, section, chunk index,
a content-derived `evidence_id`, and a `content_sha256` whose `hash_scope`
names the column it covers. A 10-Q restarts item numbering in each part, so an
item number without its part cites two different places.

---

## D10 — Historical market and macro numeric claims are declined

**Decision.** Historical and date-bound market claims, and all macro numeric
claims, fail closed to `NOT_ENOUGH_INFO` until their sources have deterministic
period-selection and freshness contracts. Company-overview metrics — market
cap, P/E — are declined for want of a source observation time.

**How it is enforced.** Three checks, at three layers:

1. *At the producer.* `FinnhubClient.get_quote` uses the source's own `t` and
   nothing else — no fallback to the server clock — converted in UTC, so the
   same quote carries the same trading day wherever the process runs. Mock mode
   supplies no observation time at all and marks itself `source_mode="mock"`.
2. *At the trust boundary.* `_observation_from_field` accepts only
   `get_stock_quote`, and re-validates the `YYYY-MM-DD` shape rather than
   assuming its producer is correct. A missing or malformed day yields no
   observation. `get_company_overview` — market cap, P/E, dividend yield, the
   52-week range — yields none regardless of payload, because that endpoint
   timestamps nothing.
3. *In the record.* The value is carried as
   `TrustedObservation.observed_at`, and `period_end` stays empty: a quote is
   observed at a moment, it does not close a reporting period. Both reach the
   audit metadata.

D2's temporal gate remains the fourth: a claim naming a period on a route that
resolves none skips the comparison and records
`temporal_status: "unresolved_period"`.

**To lift the decision itself.** Trading-day alignment for historical prices, a
freshness window for real-time quotes, and observation-date contracts for macro
series — each calibrated rather than assumed.

---

## D11 — Material-disclosure inference is not a Release-A capability

**Decision.** News-to-SEC delegation may corroborate or contradict two decisive
verdicts. Release A does not infer legal materiality or nondisclosure from
filing silence. Only `CONTRADICTS` creates a source-disagreement HITL trigger.

**Why.** The removed `UNDISCLOSED_MATERIAL_CLAIM` status escalated when a
filing covering the period did not mention a claimed fine or settlement.
Reaching it meant deciding the issuer *should* have disclosed the amount — a
materiality judgment made by testing a metric name against a set, with nothing
calibrating it. The status is deleted rather than left unused, so it cannot be
re-enabled by accident.

**What replaced it.** A narrower, checkable distinction: silence may only be
reported when an applicable filing was *successfully searched*. A delegation
that completed with NOT_ENOUGH_INFO having searched nothing reports
`SOURCE_UNAVAILABLE`, not silence — an absence of evidence is not evidence of
absence.

---

## D12 — Non-terminal review states are not integrity-checkable

**Decision.** A row whose `verdict` column holds `REVIEWING` or
`REVIEW_FINALIZATION_FAILED` returns integrity `status="unavailable"` with a
named reason — never `verified`, never `failed`.

**Why.** Two things diverge from the committed envelope during a review and
neither is tampering: `claim_pending_review` writes the lifecycle marker with a
single atomic UPDATE that deliberately leaves `full_trace` alone (that
atomicity is what makes the reviewer race safe), and the review route logs its
`hitl_*` event straight to the database, outside the envelope. Reporting
`verified` would claim a check that did not happen; reporting `failed` would
allege tampering that did not occur.

**Order matters.** The checksum is recomputed *before* any lifecycle reasoning,
so a genuinely corrupted row under review still reports `failed`. Lifecycle
state excuses the projection comparison, never the checksum.

**To lift it.** A dedicated `review_status` column, so `verdict` stops carrying
two meanings. Release A forbids new columns, which is what forces the
reason-coded `unavailable`.

---

## D13 — Non-corroboration is not contradiction

**Decision.** A claim that names no value is never REFUTED on retrieval
evidence alone. The verdict is declined to `NOT_ENOUGH_INFO` and carries the
limitation `non_corroboration_is_not_contradiction`.

**Why.** Asked whether Apple's annual report describes plans for a theme park
in Ohio, the pipeline answered **REFUTES at 0.95**, reproducibly, in 4 of 4
runs. The claim routes to the News agent, which called `search_financial_news`
four times. Every call succeeded and returned ten real articles — "Ohio Rich in
New Travel Experiences for 2014", "apple hospitality reit, inc." — none about a
theme park. The model meant *"I could not confirm this"* and said *"this is
false"*.

`_apply_override` is the component whose job is to overrule the model, but its
only guard was numeric — `claimed_val is not None and observation is None` — and
this claim names no value, so nothing applied and the model's verdict was
released as given.

**Why not a count check.** The first fix asked whether retrieval returned
anything. It does: ten articles. No count distinguishes an article that
contradicts a claim from one that merely fails to mention it, and nothing
deterministic reads relevance out of prose. Both guards now exist — an empty
result set is also refused — but the count gate alone would not have closed
this.

**Asymmetric on purpose.** Confirming a claim means having found text that
asserts it, which retrieval supplies. Refuting one on absence is the fallacy,
and the failure actually observed. SUPPORTS still stands.

**This is D9 generalized.** Filing silence does not become weaker evidence
because it arrived by the news route rather than a nested delegation.

**Declared, not queued.** The limitation suppresses the low-confidence
escalation. A reviewer opening this claim sees exactly the nothing the system
saw, which is the mistake the readiness report already names in §7.

**To lift it.** A verdict contract in which the model must cite the specific
retrieved passage that contradicts the claim, verifiable against the retrieved
set by id. Release A has no citation field on `VerdictOutput`.

---

## D14 — Filing retrieval is a subsystem, not a claim path — **LIFTED, same day**

**Status: superseded by parser rule R2b.** The decision below is kept in full
because the boundary moving is the point: it was written in the morning, was
accurate when written, and was lifted the same evening once the cost of it
became visible. What follows is the original, then what changed.

### The original decision

Release A advertised hybrid filing retrieval as a *tested retrieval subsystem*,
measured at the tool boundary, and did not claim a qualitative filing claim
could be verified end to end.

**Why.** `METRIC_WHITELIST["sec"]` holds only numeric GAAP metrics, so "Apple
discussed supplier concentration risk in its annual report" was rejected by the
parser as `non_financial` and never reached an agent. The SEC prompt separately
instructs the model not to call `search_filing_text` to re-confirm an XBRL
number. Together they left no route: **0 of 322 recorded executions carried a
`rag` data source.**

**To lift it,** the original text said: *"A qualitative SEC claim type routed to
the SEC agent. That is a parser taxonomy change, and it is Release B."*

### What lifted it

That estimate was wrong in one respect, and the error is worth recording.
`verification_strategy_for` (`config/metrics.py:185-219`) **already returned
`filing_rag`** for a `sec` claim with a null metric. The strategy existed and
was wired; nothing in the taxonomy needed changing. The only thing standing in
the way was the parser calling such claims `non_financial`.

So the change was one rule in `agents/prompts/parser_system.txt` — **R2b**: a
claim about what a filing *says* is a `sec` claim with `metric: null`. No code
change, and no touch to `METRIC_WHITELIST`, which is vendored from the parser
project and conformance-tested.

**Nothing was loosened.** These claims name no number, so the numeric guard has
nothing to demand of them. A claim that *does* name a number still needs an
XBRL fact or a market quote, and filing prose still cannot become one
(`SUPPORTING_EVIDENCE_TOOLS`, D9). D13 still forbids refuting on absence.

**Observed after the change:**

```
Apple's annual report discusses risks from supplier concentration
  -> SUPPORTS, 95%, status=success, data_sources.rag.used=true, 8 chunks
Nvidia's 10-K describes dependence on a limited number of suppliers
  -> SUPPORTS, 95%, 10 chunks
```

The reject boundary was re-verified at the same time: all seven cases still
reject with their original reasons, including the one R2b could plausibly have
swallowed — "Apple's CEO enjoys sailing on weekends" is about the company and
is in no filing, and stays `non_financial`.
`tests/integration/test_qualitative_filing_claims.py` pins both halves.

### The drift guard worked

The original text said `scripts/release_gate_evidence.py` *"asserts the
unreachability so it cannot drift unnoticed — the check fails the day a
qualitative route is added, which is when the documentation must change with
it."* That is exactly what happened, and this rewrite is the consequence. The
check is now inverted to assert reachability.


---

## D15 — A review is never finalized from an event read that failed

**Decision.** `submit_hitl_review` assembles its envelope through
`AuditLogger.events_for_finalization`, which reads strictly and reconciles the
persisted rows with the in-memory buffer by `event_id`. A read failure marks the
row `REVIEW_FINALIZATION_FAILED` and returns 503; it never finalizes.

**Why.** The route finalized with `audit.get_events(request_id) or []`.
`AuditDatabase.get_events` catches bare `Exception` and returns `[]`, so a
database read failure was indistinguishable from a run that genuinely produced
no events. The execution committed with `full_trace.events = []`, the checksum
was computed over that empty envelope, and `/audit/{id}` then reported
integrity **`verified`** — a wiped audit trail presented as an intact one. For a
system whose thesis is auditability that is worse than an outright failure,
because nothing downstream can tell.

**The second half.** `log_event` writes each row best-effort and keeps a
buffered copy. `commit_execution` reconciles from that buffer, which is why the
verify path never had this defect; the review path re-read the database instead,
so any event whose write had failed was absent from the envelope even when the
read succeeded.

**Ordering.** The merge sorts by `(timestamp, event_id)` — the same tiebreak
`get_events` applies — so the envelope and the queried trail cannot disagree.

---

## D16 — A range is an interval, in the gold data as in the code

**Decision.** `operator="range"` requires both `range_min` and `range_max`
everywhere, including the evaluation gold. 32 of the 383 rows in `test.jsonl`
carried only a midpoint `value`; all 32 were migrated to explicit bounds.

**Why.** The midpoint encoding is the one that caused a defect: collapsing
"between $50B and $150B" to $100B and comparing it as equality refuted a filed
$149B at a 32.89% difference, though it sits plainly inside the stated band. The
contract is right and the gold was stale.

**How the migration was validated.** Every affected row states both bounds in
its own input text, and the recorded midpoint checks the reading: bounds were
written only where `(min + max) / 2` reproduces the stored `value`, with the
scale word settled the same way. A row that did not reconcile would have been
reported and left alone. All 32 reconciled;
`scripts/migrate_gold_range_bounds.py` is idempotent and keeps a backup.

**The check that hid it.** `test_parser_contract_conformance` resolved the gold
directory at *import* time, so whether it ran depended on which conftest had
already loaded `.env`: `pytest tests/unit` skipped it and `pytest tests` ran it.
The failure was invisible to every unit-only run and to CI. The directory is now
resolved per test, and both bounds are in the round-trip field list.

---

## D17 — A disclosure is scoped forward from its event, not onto a period

**Decision.** `search_filing_text` passes the resolved period as `period_start`
— a lower bound on which filings could carry the disclosure — rather than as an
exact `period_end`. Numeric retrieval through `sec_tools` is unchanged and keeps
exact matching.

**This partially reverses D-era commit `5b86cff`** ("reject unscoped or
irrelevant filing evidence"), which introduced the scoping. That commit's
reasoning was:

> Retrieval ignored the resolved period, so a FY2024 query could fill its
> candidate set with chunks from other years — wrong evidence, not weak
> evidence, and nothing downstream could tell.

That is correct, and the example it reaches for says what it was aimed at: a
*numeric* query, where a FY2023 figure satisfying a FY2024 claim is genuinely
the wrong evidence. The mistake was applying it to narrative text as well.

**Why the two differ.** A number belongs to exactly one period. A disclosure
describes an *event*, and appears in whichever filings were current while the
matter was live — often across several, often years later. Apple's March 2024
European Commission investigation is disclosed in the **FY2025** 10-K and
carried across three filings. Measured against the live corpus:

| scope | chunks |
|---|---|
| `period_end = 2024-12-31` (what the resolver produces) | 0 |
| `period_end = 2024-09-28` (even with the fiscal mapping corrected) | 0 |
| `period_start = 2024-01-01` | 4 |

Note the second row: this was not the fiscal-vs-calendar bug wearing a
disguise. Exact matching fails on a correct fiscal date too.

**What it cost while it stood.** The News → SEC delegation failed 100% of the
time. The nested agent searched, got nothing, reworded, got nothing, and died at
its recursion limit of 7 — reported honestly as `FAILED`, with no provenance at
all. A/B against the real delegation:

```
A  exact period_end   : success=False  status=FAILED   provenance=0  (recursion limit)
B  forward range      : success=True   status=…        provenance=2  (found=3 each)
```

End to end after the change, the same claim returns `status=NO_MATCHING_
DISCLOSURE`, `retrieved_value=500,000,000` against a claimed 500,000,000, and 6
filing sources. The status is not `CORROBORATES` because the nested agent still
declines to assert a number read from prose — the trust boundary holding, which
is the point.

**Why relaxing this is safe.** `search_filing_text` is in
`SUPPORTING_EVIDENCE_TOOLS`, so filing prose can never become a trusted
observation and no numeric verdict can rest on it. "Wrong-period narrative text"
and "wrong-period figure" are therefore not the same hazard, which is exactly
why the constraint was misplaced here. The ticker filter — the one that actually
prevents cross-company evidence — is untouched.

**The imprecision this accepts, stated rather than discovered later.** A claim
naming a specific filing will also match later filings that carry the same
disclosure forward. There is no signal distinguishing "the period of the filing"
from "when the event happened"; the parsed `period` field means both. Each chunk
carries its own `period_end`, `filing_type` and `evidence_id` in the response, so
the attribution stays visible to a reader. Bounded, visible imprecision was
preferred over a feature that returned nothing.

**Diagnostic note.** Two theories were investigated and disproved before this
one: that the nested agent's toolbox was too wide, and that
`A2A_MAX_ITERATIONS = 3` was too small. Under forward-range scoping the agent
still calls the same "wasteful" preamble tools and still finishes inside three
iterations, because the first search succeeds and the retry loop never starts.
Both were symptoms.

**To lift it.** A parsed claim that distinguishes an event date from a filing
period would allow exact scoping where a filing is named and forward scoping
where an event is. That is a parser change, and it is Release B.
