# `Finvet_engineering.png` — corrections

Verified against the compiled graph on 2026-08-20. The PNG has no source file in
the repo, so apply these in whatever tool produced it.

Ground truth, dumped from `create_verification_graph(checkpointer=MemorySaver())`:

```
12 nodes: input_guardrails, claim_parser, period_resolver, sec_agent,
          market_agent, news_agent, reject_handler, consensus,
          output_guardrails, hitl_checkpoint, apply_hitl_decision,
          response_generator

__start__            -> input_guardrails
input_guardrails     -> claim_parser                        (static, unconditional)
claim_parser         -> period_resolver     [sec]           (conditional)
claim_parser         -> market_agent        [market]        (conditional)
claim_parser         -> news_agent          [news]          (conditional)
claim_parser         -> reject_handler      [reject]        (conditional)
period_resolver      -> sec_agent
sec_agent            -> consensus
market_agent         -> consensus
news_agent           -> consensus
reject_handler       -> response_generator
consensus            -> output_guardrails
output_guardrails    -> hitl_checkpoint     [needs_hitl]    (conditional)
output_guardrails    -> response_generator  [no_hitl]       (conditional)
hitl_checkpoint      -> apply_hitl_decision
apply_hitl_decision  -> response_generator
response_generator   -> __end__
```

## 1. Factually wrong — fix these first

**The PostgresSaver band and `checkpointer=pg`.** `main.py:35` is `MemorySaver()`.
The `PostgresStore` at `main.py:53` is a LangGraph *store* backing claim memory,
not a checkpointer. So all three claims in that band are false as drawn:

- "checkpoint written after every node" — they are written to memory, not Postgres
- "time-travel replay" — not available; checkpoints die with the process
- "immutable audit trail" — that is the separate SQLAlchemy `audit/` tables with
  the SHA-256 `execution_hash`, which have nothing to do with the checkpointer

Replace with `MemorySaver (in-process)` and note the real consequence: **an API
restart destroys any paused HITL review**, while its `audit_executions` row stays
`PENDING` forever. Draw the immutable audit trail as its own band pointing at the
`audit/` tables.

**The Invariant panel** — "After the agent band, no LLM writes to `verdict`. The
value is recomputed in Python from evidence." Both sentences overstate:

- The verdict comes from a *second LLM call*, `_extract_verdict` (`base.py:248`),
  deliberately run on a filtered message history to avoid anchoring.
- `_apply_override` (`base.py:306`) only wins when both a claimed and a retrieved
  numeric value exist. A news claim with no number keeps the LLM's verdict.
- `_simple_consensus` — the box drawn as `aggregate_evidence` with the caption
  "deterministic recompute · no LLM writes the verdict" — does **not** recompute
  the verdict. It passes `agent_evidence["verdict"]` straight through and adjusts
  only *confidence* (`workflow.py:185-215`).

Accurate wording: *"A second LLM call extracts the verdict on a filtered history
(anti-anchoring). Python overrides it whenever a numeric comparison is available.
Consensus adjusts confidence only."*

**The `unsafe / off-topic → reject` edge conflates two different paths.** Only
off-topic is a graph edge. `input_guardrails` has no conditional edge at all — it
is a static edge to `claim_parser` (confirmed above). An unsafe input **raises**
`GuardrailViolation` (`input_guardrails.py:42`), which unwinds out of LangGraph,
is caught in `verify.py:173-203`, and returns HTTP 400 via `main.py:80`.

Draw these as two separate things:
- `claim_parser --[reject]--> reject_handler --> response_generator` (in-graph)
- `input_guardrails ==X==> HTTP 400` exiting the graph boundary entirely

This is intentional, and worth labelling as such: unsafe input never receives a
verdict, because a verdict asserts that a financial claim was evaluated. The
tradeoff is that guardrail violations produce a `guardrail_violation` audit event
but **no `audit_executions` row and no execution_hash**.

## 2. Structural

- **`period_resolver` is missing.** `"sec"` routes to `period_resolver`, which
  then goes to `sec_agent`. The diagram shows `"sec"` reaching `sec_agent` directly.
- **`human_review` is two nodes**: `hitl_checkpoint` (where `interrupt_before`
  pauses) and `apply_hitl_decision` (which applies approve/override/reject). The
  compile box should read `interrupt_before=["hitl_checkpoint"]`.
- **Node names should match the registered names** so a reader can grep from box
  to code: `parse_claim`→`claim_parser`, `aggregate_evidence`→`consensus`,
  `respond`→`response_generator`.
- **"8 nodes" is wrong** — there are 12. The diagram itself draws 10 boxes.
  "2 conditional edges" and "1 interrupt point" are both correct.
- **The HITL branch has two triggers, not one.** The label reads
  `confidence < 0.70 → review`, but `output_guardrails.py:61` also sets
  `hitl_required=True` on `output_safety_violation` (financial-advice guard /
  Llama Guard S6). Label the branch `confidence < 0.70 OR output_safety_violation`.
- **The `human_review → respond` arrow is three arrows** — approve, override,
  and reject — each producing a different disposition.
- Optional: the agent badges show only MCP / Tavily / Finnhub. All three agents
  also carry `search_past_verifications` (claim memory), and the SEC agent carries
  `search_filing_text` (hybrid RAG). Neither RAG nor memory appears anywhere on
  the diagram despite both being in the architecture summary.

## 3. Correct as drawn — leave alone

`reject → respond`; the agent band into `consensus → output_guardrails`;
2 conditional edges; 1 interrupt point; the 0.70 threshold (`settings.py:80`);
`≤ 5 iterations` (`constants.py:31`, now wired through all three agents);
the A2A arrow direction sec→news (only `SECAgent` gets `corroborate_with_news`,
`sec_agent/react_agent.py:37`).

## 4. Worth adding

Every terminal path now writes a `disposition` into
`final_response["metadata"]`, persisted to the audit `full_trace` JSONB column:

| disposition | written by |
|---|---|
| `released` | `response_generator` (auto-release) |
| `pending_review` | `response_generator` (paused at checkpoint) |
| `rejected_parser` | `reject_handler` |
| `rejected_human` | `apply_hitl_decision` |
| `approved_human` | `apply_hitl_decision` |
| `overridden_human` | `apply_hitl_decision` |
| `rejected_input_guard` | reserved — the input guard currently fails closed at the API boundary and never reaches `response_generator` |

Annotating `response_generator` as the single point where disposition is recorded
makes the "why did this run end this way" story legible on the diagram.
