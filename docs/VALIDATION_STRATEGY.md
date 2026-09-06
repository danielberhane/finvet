# FinVet Validation Strategy

How FinVet is tested and measured, and how that work maps to model-risk
principles from three jurisdictions: SR 26-2 (US), OSFI Guideline E-23
(Canada), and the EU AI Act. FinVet is a research system built by one
person; this document claims alignment with principles, verified against
the code and published artifacts — it claims no compliance status, and
its validation is not independent (see Known gaps).

## Test taxonomy, as built

### Unit tests (`tests/unit/` — 84 files, ~1,550 tests, mocked LLMs)

Pure logic: tolerance thresholds, the verdict override, routing, guards,
consensus adjustments, response assembly. One rule shapes the suite:
**tests must drive the producer.** Any test covering a verdict, an
escalation, or a persistence path enters through a route callable, a
graph node, or a decorated tool — never a hand-built dict. Three defects
once survived a 500-test suite because their tests constructed their own
inputs; a fixture asserts the shape you remembered, not the shape the
system emits.

Run: `.venv/bin/python -m pytest tests/unit -q`

### Integration tests (`tests/integration/`)

End-to-end against the running API and live data sources, including:

| Test | What it establishes |
|---|---|
| `test_xbrl_retrieval.py` | XBRL values vs SEC primary sources — 198/199, the one miss returns NOT_ENOUGH_INFO |
| `test_rag_release_a.py` | the 70-case retrieval manifest (`tests/accuracy/rag_release_a_manifest.json`) |
| `test_qualitative_filing_claims.py` | text-sourced verdicts cite retrieved passages |
| `test_review_race.py` | HITL resume semantics under contention |
| `test_golden.py` | asserts outcome expectations over recorded benchmark artifacts |

### The golden benchmark harness

The end-to-end measure is a frozen 97-claim set, run through the public
API exactly as a user would submit claims. Separation of duties within
the harness:

- `scripts/run_golden.py` **records and asserts nothing** — it posts
  claims, saves every response, and writes the artifact after each row.
- `tests/integration/test_golden.py` **judges** recorded artifacts.
- `scripts/eval_layers.py` computes the measurement layers offline from
  saved artifacts: no API calls, no spend, indefinitely re-runnable.

Composition, labelling rules, provenance, and disclosed biases are in
the public [dataset card](eval/DATASET_CARD.md); redacted per-row
artifacts for the published runs are beside it in [`docs/eval/`](eval/),
and every published metric recomputes from them.

## Measurement layers — the agentic evaluation

An agentic system can be right for the wrong reason: an agent can reach
the correct verdict from the wrong source, without the required tool, or
by skipping a step whose absence no accuracy number reveals. The proof
is in this repo's own history — the routing layer, backtested over
recorded runs, found two claims that returned the expected verdict in
every run while the News→SEC delegation they were built to exercise
never fired. Outcome scoring saw nothing. That is why the evaluation is
seven layers rather than one:

| Layer | Question | Scoring |
|---|---|---|
| 1 Outcome | right verdict? | deterministic, vs labels |
| 2 Trajectory | required tools called, no out-of-lane calls, zero spend where none is allowed? | DeepEval ToolCorrectnessMetric for required-tool checks; the rest deterministic |
| 3 Grounding | every decisive number traceable to a trusted observation? | deterministic |
| 4 Calibration | confidence ≈ accuracy (ECE)? | deterministic |
| 5 Asymmetric risk | confidently-wrong verdicts, counted alone? | deterministic |
| 6 Reliability | same result on repeated runs (pass@1, pass^k)? | deterministic |
| 7 Reachability | can the failure occur at all? | argued from construction, not measured |
| — Routing | evidence path used vs the path the label expects? | deterministic, reported not asserted |

**Controls and measurements are kept apart.** Some properties are
enforced by construction: each agent binds only its own tools, so an
out-of-lane call cannot happen; reject/guard/declined claims
short-circuit before any agent, so they cannot spend; a decisive numeric
verdict is unreachable without a source-traced observation. Those are
controls — the layers verify they held, not whether they might. Other
properties are observed in runs — accuracy, calibration, escalation
behavior — and hold only for the runs measured. The published results
state which kind each number is, because a validator weighs the two
differently.

**Assertions are graded.** Each dataset row carries a strength:
`strict` (the expectation must hold), `safe` (only the opposite verdict
fails — an escalation is acceptable), or `observe` (recorded, never
asserted — the known-defect rows). Behavioral rows test conformance to
the design without over-claiming, and observe rows cannot fail a run.

**The scoring map cannot drift.** The per-agent tool lanes the
trajectory layer scores against are pinned to the agent modules by a
unit test — if an agent's toolset changes, the evaluation's ground
truth fails loudly instead of silently scoring against a stale map.

**Repeated runs are part of the method.** pass@1 hides a claim that
flickers between right and wrong; pass^k (passed on *all* k attempts)
exposes it. Repeated series isolate a small stable set of flickering
rows — tolerance-boundary claims that oscillate between a verdict and
an escalation — which single-run accuracy cannot distinguish from solid
behavior. Validation also includes a matched cross-model comparison:
the same frozen claims through two models on identical code, with each
run's model label taken from the serving process. The method lives
here; the numbers live in the evidence pack ([`docs/eval/`](eval/)).

Verdict correctness is **never scored by an LLM judge.** The system's
core commitment is that the LLM is untrusted and a deterministic
comparator decides verdicts; using a model to grade verdicts would
reintroduce at evaluation time the assumption the design rejects. The
one framework dependency (DeepEval, tool-correctness only) is isolated
behind an adapter with a deterministic fallback, enforced by test.

Layer 7 findings are properties of the code: A2A delegation cannot
recurse because the SEC agent holds no delegation tool, and a decisive
numeric verdict is unreachable without a source-traced observation —
the path fails closed.

## Evaluation integrity controls

Each control exists because the failure it prevents happened:

- **Model identity from the serving process.** The runner reads the
  model from the API's `/health` — not its own environment — and refuses
  to start on a mismatch. 56 rows were once labelled with a model that
  did not produce them; the client shell's belief is not evidence.
- **Freeze rule.** Runs are comparable only when produced at recorded
  SHAs of both the code and the dataset; the commit message of every
  run artifact states both.
- **Frozen set, committed hash.** The dataset is sealed by SHA-256
  (published in the dataset card) and held out privately: a released
  test set enters training corpora and stops measuring anything.
- **Burned rows.** A row whose claim text becomes public is permanently
  excluded — recorded in `src/finvet/eval/exclusions.py` and enforced
  where the runner selects rows, so a burned claim cannot be executed,
  let alone scored.
- **Artifacts are append-only spend.** Paid run artifacts are never
  regenerated; scoring reads them. The output-path flag of the layers
  script refuses to overwrite a run artifact (it did once; the guard is
  the scar).

## Runtime controls relevant to validation

- **Deterministic verdict override**: numeric comparisons are recomputed
  in Python with source-appropriate tolerances; when the model
  disagrees, the override applies and **both** verdicts are recorded
  (`llm_original_verdict`, `override_applied`) — the disagreement rate
  is itself measurable from the audit trail.
- **Trusted observations**: for numeric claims the comparison input is a
  structured observation (XBRL fact, quote field, or the deterministic
  fine/settlement extraction); model prose never becomes the number.
- **Bounded delegation**: one hop, News→SEC only; where the filing
  states an amount, the verdict follows the filing rather than the
  press.
- **Human oversight**: low confidence, unsafe output, or a
  press-vs-filing conflict pauses at a LangGraph checkpoint; the
  reviewer's decision is merged in on resume. An unresumable review is
  refused, not answered from the reviewer's own submission.
- **Audit trail**: every tool call and verdict persists to Postgres with
  a checksum the API re-verifies on read. Honest scope: it detects a
  record altered without its checksum recomputed; it is not tamper-proof
  against a writer who can change both.
- **Input/output guardrails**: regex/PII checks always on, optional
  Llama Guard layer; the free-form memory-context field was removed as a
  prompt-injection channel and must not return.

## Regulatory alignment

None of the three texts imposes obligations on this system — FinVet is
not a regulated institution's production model and is not placed on the
EU market. The mappings below say which principle each piece of
validation work serves, so a reader from any of the three regimes can
locate the evidence.

### SR 26-2 (Federal Reserve / OCC / FDIC, 2026) — United States

> **Scope note.** SR 26-2 (17 Apr 2026) supersedes the Fed's earlier
> model-risk letters, replacing annual revalidation with risk-based
> oversight tied to model materiality. It places generative and agentic
> AI **outside** its formal scope as "novel and rapidly evolving",
> directing institutions to apply the underlying principles —
> materiality, ongoing monitoring, effective challenge — to systems it
> does not cover. FinVet is such a system: the alignment claimed here is
> with those principles, not with a compliance obligation SR 26-2
> imposes.

- *Conceptual soundness* — design decisions are recorded with their
  rationale and revisited on evidence
  ([`RELEASE_A_DECISIONS.md`](RELEASE_A_DECISIONS.md)); limitations are
  published in the README rather than discovered by the reader.
- *Outcomes analysis* — the golden benchmark and layer measurements
  above, with recomputable artifacts.
- *Ongoing monitoring* — the model-drift refusal, the freeze rule, and
  the checksummed audit trail.
- *Effective challenge* — **not satisfied**: the author validates their
  own system. Disclosed here and in the dataset card, not papered over.

### OSFI Guideline E-23 (Canada, 2024 revision) — model lifecycle

E-23 expects risk management across the model lifecycle — design,
review, deployment, monitoring, decommission — with intensity scaled to
model risk, and its revised scope explicitly reaches AI/ML models.

- *Design* — explicit contracts (the 7-field parser output; `range` is
  legal input that fails closed rather than being approximated), named
  constants over magic numbers, fail-closed defaults.
- *Review* — the test taxonomy and measurement layers above.
- *Deployment* — locked dependencies (`uv.lock`), containerized stack,
  CI running lint, tests and builds on every push.
- *Monitoring* — per-request audit rows with tool calls, sources and
  verdicts; drift refusal at the benchmark boundary.
- *Decommission* — not applicable to a research system; noted for
  completeness rather than claimed.

### EU AI Act (Regulation (EU) 2024/1689) — used as a design checklist

FinVet is not a deployed high-risk system and no conformity is claimed;
the high-risk requirements are used as a checklist because they name
concrete engineering obligations:

- *Art. 9 (risk management)* — the asymmetric-risk layer counts
  confidently-wrong verdicts separately; fail-closed paths prefer
  declining to guessing.
- *Art. 10 (data governance)* — the dataset card documents composition,
  label provenance and known biases; label sources are pinned to SEC
  accessions and concepts.
- *Art. 12 (record-keeping)* — the audit trail records every tool call,
  data source and verdict per request.
- *Art. 14 (human oversight)* — the HITL checkpoint, with its stated
  limitation: pending reviews do not survive an API restart.
- *Art. 15 (accuracy, robustness)* — published accuracy with evidence,
  calibration measured by ECE, injection patterns blocked before any
  model call.

## Prompt and configuration versioning

System prompts are text files in `src/finvet/agents/prompts/`
(`parser_system.txt`, `sec_system.txt`, `market_system.txt`,
`news_system.txt`) — every change is a commit, and benchmark results are
tied to the SHA that produced them. Thresholds and tolerances live in
`src/finvet/config/constants.py`, not inline.

## Known gaps

Stated so they are findings, not surprises:

- **No independent validation.** Every label, test and measurement was
  produced by the system's author. This is the largest gap and no
  internal control closes it.
- **57 of 97 parse labels await human adjudication** (verdict labels are
  unaffected; the dataset card tracks the split).
- **No canary string** in the golden set — contamination would not be
  detectable from model output; the freeze and hash mitigate but do not
  replace one.
- **HITL state is in-memory** (MemorySaver); production use would need a
  persistent checkpointer.
- **The consensus step is heuristic, not learned**, and single-agent
  routing means it adjusts confidence rather than reconciling opinions.
- **Reliability coverage is partial**: pass^k is measured over repeated
  runs per model; series beyond the published runs are added as they
  are produced.
