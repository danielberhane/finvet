"""LangGraph state definition for the FinVet verification pipeline.

This is the single most important file in the codebase. Every node in the
LangGraph pipeline reads from and writes to this state dictionary.

How it works:
    - VerificationState is a TypedDict (a regular Python dict with typed keys).
    - LangGraph requires TypedDict for state — not Pydantic, not dataclass.
    - total=False means ALL fields are optional. The state starts nearly empty
      (only 4 input fields) and grows as each node adds its own fields.
    - Each node receives the full state, reads upstream fields, and returns a
      dict with new fields. LangGraph merges the returned dict into the state.
    - Nodes must use state.get("field") not state["field"] to avoid KeyError
      on fields that haven't been set yet by upstream nodes.

Pipeline flow (which node writes which fields):
    /verify route      → claim_raw, user_id, request_id, timestamp_received,
                          execution_start_time, audit_trail, total_tokens_used,
                          memory_context
    Node 1 (input_guardrails)   → claim_normalized, guardrails_passed/failed,
                                   guard_result_input, audit_events
    Node 2 (claim_parser)       → parsed_claim, parser_used, total_tokens_used
    Node 3 (period_resolver)    → canonical_period
    Node 4 (domain_agent)       → agent_type, agent_evidence,
                                   rag_chunks_retrieved, corroboration_result
    Node 5 (consensus)          → verdict, confidence, confidence_label,
                                   consensus_reasons, confidence_adjustments
    Node 6 (output_guardrails)  → hitl_required, hitl_triggers,
                                   guard_result_output, audit_events
    Node 7 (hitl_checkpoint)    → hitl_checkpoint_passed
    Node 8 (apply_hitl_decision)→ hitl_applied, verdict (override),
                                   confidence (override)
    Node 9 (response_generator) → final_response, execution_end_time

    /review route (external)    → hitl_decision, hitl_override_verdict,
                                   hitl_reviewer_notes (injected via
                                   graph.update_state())
"""

from typing import Any, Literal, Optional, TypedDict
from .claim import ParsedClaim, CanonicalPeriod


class AgentEvidence(TypedDict):
    """Structured evidence returned by a domain agent (SEC, Market, or News).

    This is the output of the ReAct loop + verdict extraction in base.py.
    It contains everything needed to explain and audit the agent's conclusion:
    the verdict, the numeric evidence, the tools used, and the reasoning.

    Stored in state as state["agent_evidence"] and later written to the
    audit_executions PostgreSQL table in the full_trace JSONB column.
    """

    # Which agent produced this evidence: "sec", "market", or "news".
    # Used by response_generator to label the data source and by consensus
    # to log which agent ran.
    agent: str

    # The agent's conclusion about the claim.
    # SUPPORTS = claim is true, REFUTES = claim is false,
    # NOT_ENOUGH_INFO = couldn't find enough data to decide.
    # May be overridden by the Python deterministic check in base.py
    # (e.g., LLM says SUPPORTS but numbers don't match → REFUTES).
    verdict: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    # Agent's confidence in the verdict, 0.0 to 1.0.
    # Comes from the verdict extraction LLM call, then may be adjusted
    # by the consensus node (close_match_bonus, large_diff_penalty, etc.).
    confidence: float

    # The actual numeric value the agent found in the data.
    # Example: claimed $94B → retrieved $94.2B → retrieved_value = 94200000000.0
    # None if the agent couldn't find a specific number (qualitative claims,
    # or when the LLM verdict call fails to populate it — see fallback in base.py).
    # This is the key field for the Python verdict override comparison.
    retrieved_value: Optional[float]

    # Human-readable description of where the data came from.
    # Examples: "SEC EDGAR XBRL (10-K filing)", "Finnhub API", "Financial News (Tavily)"
    # Shown in the UI under the verdict card.
    source_description: str

    # Direct URL to the source document, if available.
    # Example: SEC filing URL. None for most API-based lookups.
    source_url: Optional[str]

    # How far the claimed value is from the retrieved value, as a percentage.
    # Example: claimed $94B, retrieved $94.2B → 0.21%.
    # Used by consensus for confidence adjustments (only for equality claims).
    # Used by response_generator to show "Difference from claimed value: 0.21%".
    # None if either claimed or retrieved value is missing.
    magnitude_difference_percent: Optional[float]

    # List of tool names the agent called during the ReAct loop.
    # Examples: ["get_company_info", "get_income_statement"]
    # Used by: consensus (thoroughness bonus if >= 3 tools),
    #          response_generator (source citations),
    #          _format_metadata (data_sources provenance: xbrl/rag/a2a).
    tools_called: list[str]

    # Full input/output of every tool call for audit trail.
    # Each entry: {"tool": "get_income_statement", "input": {...},
    #              "result": "...", "success": True/False}
    # Used by base.py _extract_retrieved_value() fallback when the verdict
    # LLM doesn't populate retrieved_value.
    # Written to PostgreSQL full_trace for complete auditability.
    tool_calls_detail: list[dict[str, Any]]

    # The LLM's natural language explanation of its verdict.
    # Example: "Apple reported revenue of $94.9B for Q4 2024, within 1% of
    #           the claimed $94B."
    # Shown in the UI explanation section.
    # Checked by output_guardrails (Llama Guard + financial guard) for safety.
    reasoning: str

    # How long the agent took in milliseconds (ReAct loop + verdict extraction).
    # Used for performance monitoring in the UI and audit trail.
    execution_time_ms: int


class VerificationState(TypedDict, total=False):
    """Complete state object for the FinVet verification pipeline.

    This state dictionary travels through all 12 nodes in the LangGraph DAG.

    TypedDict: Makes this a regular Python dict with typed keys. LangGraph
    requires TypedDict — not Pydantic BaseModel, not dataclass.

    total=False: ALL fields are optional. The state starts with only 4 fields
    (claim_raw, user_id, request_id, timestamp_received) and grows to ~30
    fields by the end. Without total=False, creating a state with missing
    fields would raise TypeError. Always use state.get("field") to read.
    """

    # ===================================================================
    # INPUT FIELDS — Set by /verify route BEFORE the pipeline starts
    # These are the seed values. Every downstream node can read them.
    # ===================================================================

    # The exact string the user typed or submitted via the API.
    # Example: "Apple's revenue was $94 billion in Q4 2024"
    # Never modified by any node — preserved for audit trail and final response.
    # Read by: input_guardrails (to classify), claim_parser (to parse),
    #          output_guardrails (passed to Llama Guard for context),
    #          response_generator (included in final_response["claim"]).
    claim_raw: str

    # Who submitted the claim. From request body or defaults to "anonymous".
    # Used for audit logging and future per-user rate limiting.
    # Read by: input_guardrails (audit event).
    user_id: str

    # UUID-based unique identifier generated in /verify route.
    # Format: "req_7f3a2b1c9d4e" (req_ prefix + 12 hex chars).
    # This is the PRIMARY KEY that ties everything together:
    # - Audit events and execution records in PostgreSQL
    # - MemorySaver checkpointer slot for HITL resume
    # - Memory store entries for future cache lookups
    # - All log messages include this for correlation
    # Read by: every single node (for logging and audit).
    request_id: str

    # ISO format timestamp of when the HTTP request hit the API.
    # Example: "2026-02-24T14:30:00.000000"
    # Set once, never modified. Used to calculate execution_time_ms at the end.
    timestamp_received: str

    # ===================================================================
    # NODE 1: INPUT GUARDRAILS
    # Written by: input_guardrails node (src/finvet/graph/nodes/input_guardrails.py)
    # Runs: RegexGuardProvider → LlamaGuardProvider (if enabled)
    # ===================================================================

    # The claim text after cleaning: Unicode NFKC normalization, whitespace
    # collapsed, curly quotes replaced, PII redacted (emails → [REDACTED_EMAIL]).
    # Often identical to claim_raw unless the user pasted weird characters.
    # Read by: claim_parser (uses this instead of claim_raw for parsing).
    claim_normalized: str

    # Names of guardrail checks that passed.
    # Example: ["composite_guard"]
    # Read by: audit trail only.
    guardrails_passed: list[str]

    # Names of guardrail checks that failed.
    # Always empty if the pipeline continues (failures raise GuardrailViolation
    # which aborts the pipeline with a 400 error).
    guardrails_failed: list[str]

    # Full GuardResult from input classification, serialized as dict.
    # Contains: safe, categories, flags, violation_type, provider, latency_ms.
    # Example: {"safe": true, "provider": "composite", "latency_ms": 12.5,
    #           "flags": ["llama_guard_unavailable"]}
    # Read by: audit trail for debugging guard behavior.
    guard_result_input: Optional[dict[str, Any]]

    # Full GuardResult from OUTPUT classification (set by output_guardrails, not here).
    # Same structure as guard_result_input but for the agent's response.
    # None until Node 6 (output_guardrails) runs.
    guard_result_output: Optional[dict[str, Any]]

    # ===================================================================
    # NODE 2: CLAIM PARSER
    # Written by: claim_parser node (src/finvet/graph/nodes/claim_parser.py)
    # Runs: DeepSeek LLM with structured output → ParsedClaim
    # ===================================================================

    # The structured extraction of the claim. This is the MOST IMPORTANT
    # field in the entire state — it determines:
    #   - claim_type → which agent runs (sec/market/news/reject)
    #   - ticker → which company to look up ("AAPL")
    #   - value → the number to verify (94000000000.0)
    #   - comparison → how to compare: eq/gt/gte/lt/lte
    #   - period → the time reference ("Q4 2024")
    # Read by: period_resolver, domain agents, consensus, response_generator,
    #          base.py (verdict override), helpers.py (build_preliminary_analysis).
    parsed_claim: ParsedClaim

    # Which LLM parsed the claim. Currently always "deepseek".
    # Exists for future flexibility (e.g., fallback to a different parser).
    parser_used: Literal["deepseek"]

    # ===================================================================
    # NODE 3: PERIOD RESOLVER
    # Written by: period_resolver node (src/finvet/graph/nodes/period_resolver.py)
    # Runs: Regex parsing of period string → calendar date range
    # Only runs for SEC claims. Market/News claims get "current" period.
    # ===================================================================

    # The resolved time period with exact start/end dates.
    # Example: "Q4 2024" → CanonicalPeriod(start="2024-10-01", end="2024-12-31")
    # IMPORTANT: Currently assumes CALENDAR year, not fiscal year. The agent
    # compensates by calling get_company_info to get the real fiscal_year_end.
    # Read by: domain agents (to know which filing period to fetch),
    #          response_generator (assumptions become disclosures).
    canonical_period: CanonicalPeriod

    # ===================================================================
    # NODE 4: DOMAIN AGENT (SEC, Market, or News)
    # Written by: run_sec_agent / run_market_agent / run_news_agent
    #             (src/finvet/graph/nodes/domain_agents.py)
    # Runs: BaseVerificationAgent.execute() — ReAct loop + verdict extraction
    # Only ONE agent runs per claim, determined by parsed_claim.claim_type.
    # ===================================================================

    # Which agent executed: "sec", "market", or "news".
    # Matches parsed_claim.claim_type. Used in audit trail and response metadata.
    agent_type: Literal["sec", "market", "news"]

    # The full structured output of the ReAct loop. See AgentEvidence above.
    # Contains: verdict, confidence, retrieved_value, magnitude_difference_percent,
    #           tools_called, tool_calls_detail, reasoning, execution_time_ms.
    # Read by: consensus (to produce final verdict/confidence),
    #          output_guardrails (reasoning checked for safety),
    #          response_generator (to build explanation, sources, metadata),
    #          /verify route (for memory storage and audit commit).
    agent_evidence: AgentEvidence

    # ===================================================================
    # NODE 5: CONSENSUS
    # Written by: _simple_consensus in workflow.py
    # Runs: Single-agent pass-through with confidence adjustments.
    # If multi-agent is added later, this is where voting/merging happens.
    # ===================================================================

    # The FINAL verdict after consensus. For single-agent, same as
    # agent_evidence.verdict. For multi-agent (future), could differ
    # if agents disagree and voting resolves the conflict.
    # Can be overwritten by apply_hitl_decision if human overrides.
    verdict: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    # The FINAL confidence after adjustments. Starts from agent_evidence.confidence,
    # then consensus applies:
    #   - close_match_bonus (magnitude_diff < threshold, eq claims only)
    #   - large_diff_penalty (magnitude_diff > threshold, eq claims only)
    #   - thoroughness_bonus (tools_called >= threshold)
    # Clamped to [0.0, CONSENSUS_MAX_CONFIDENCE].
    # This value determines HITL routing: < 0.70 → human review required.
    confidence: float

    # Human-readable confidence label derived from confidence score:
    #   >= 0.85 → "HIGH"
    #   >= 0.70 → "MODERATE"
    #   <  0.70 → "LOW"
    # Shown in the UI verdict card. "LOW" claims typically trigger HITL.
    confidence_label: Literal["HIGH", "MODERATE", "LOW"]

    # Explanations of why this verdict was chosen.
    # Currently: the agent's reasoning text passed through.
    # For multi-agent (future): ["2 of 3 agents agree on SUPPORTS"].
    consensus_reasons: list[str]

    # Specific confidence adjustments applied by consensus.
    # Each entry: {"reason": "close_match", "amount": 0.03}
    # Empty list means no adjustments were needed.
    # Visible in audit trail for transparency.
    confidence_adjustments: list[dict[str, Any]]

    # ===================================================================
    # NODE 6: OUTPUT GUARDRAILS
    # Written by: output_guardrails node (src/finvet/graph/nodes/output_guardrails.py)
    # Runs: Confidence threshold check + Llama Guard + Financial advice guard
    # ===================================================================

    # List of conditions that require human review. Possible values:
    #   "low_confidence" — confidence < settings.confidence_threshold_hitl (0.70)
    #   "output_safety_violation" — Llama Guard or Financial guard flagged the output
    # Empty list means no triggers fired → no HITL needed.
    # Multiple triggers can fire on the same claim (they're independent).
    hitl_triggers: list[str]

    # The gate: should the pipeline pause for human review?
    # True if ANY trigger in hitl_triggers fired.
    # Determines routing: True → hitl_checkpoint (pause), False → response_generator (skip).
    hitl_required: bool

    # ===================================================================
    # NODES 7 & 8: HITL CHECKPOINT + APPLY DECISION
    # Written by: _hitl_checkpoint and _apply_hitl_decision in workflow.py
    #
    # HITL flow:
    #   1. output_guardrails sets hitl_required=True
    #   2. _route_after_guardrails sends to hitl_checkpoint
    #   3. LangGraph pauses BEFORE hitl_checkpoint (interrupt_before config)
    #   4. /verify returns status="pending_review" to the client
    #   5. Human reviews via POST /review/{request_id}
    #   6. /review calls graph.update_state() to inject hitl_decision
    #   7. /review calls graph.invoke(None, config) to resume
    #   8. hitl_checkpoint runs (logs audit), then apply_hitl_decision runs
    #   9. Pipeline continues to response_generator → END
    # ===================================================================

    # Set to True by _hitl_checkpoint when it runs (after resume).
    # Bookkeeping flag — confirms the checkpoint node executed.
    # During the first /verify call (before resume), this field doesn't exist yet
    # because the graph pauses BEFORE the checkpoint node runs.
    hitl_checkpoint_passed: bool

    # Set to True by _apply_hitl_decision when a human decision was applied.
    # False or missing means no human reviewed this claim.
    hitl_applied: bool

    # The human reviewer's choice, injected via graph.update_state() from /review:
    #   "approve" — accept the agent's verdict as-is
    #   "override" — replace the verdict with hitl_override_verdict
    #   "reject" — discard the claim entirely (verdict → REJECTED)
    # None until a human submits a review. For non-HITL claims, stays None forever.
    hitl_decision: Optional[Literal["approve", "override", "reject"]]

    # The new verdict chosen by the human when decision="override".
    # Must be one of SUPPORTS, REFUTES, NOT_ENOUGH_INFO.
    # None for "approve" and "reject" decisions.
    # When set, _apply_hitl_decision overwrites state["verdict"] with this value
    # and sets confidence to 0.95 (human override = high confidence).
    hitl_override_verdict: Optional[Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]]

    # Free-text notes from the human reviewer.
    # Example: "Verified against 10-K filing page 47 — agent missed restatement"
    # Stored in the audit trail for compliance documentation.
    # Optional for all three decision types.
    hitl_reviewer_notes: Optional[str]

    # ===================================================================
    # NODE 9: RESPONSE GENERATOR
    # Written by: response_generator (src/finvet/graph/nodes/response_generator.py)
    # Runs: Assembles the final JSON response from all upstream state fields.
    # ===================================================================

    # The complete JSON response returned to the API client.
    # Contains: status, request_id, claim, verdict, confidence, confidence_label,
    #           summary, explanation, sources, disclosures, metadata.
    # Built by response_generator using all upstream fields.
    # This is what /verify returns to the Streamlit UI or curl client.
    final_response: dict[str, Any]

    # ===================================================================
    # CROSS-CUTTING FIELDS — Set by tools/agents during execution
    # These can be written by any node, not tied to a specific pipeline step.
    # ===================================================================

    # Filing text chunks retrieved via hybrid RAG search (pgvector cosine +
    # tsvector BM25 with RRF fusion). Set by run_sec_agent when the SEC agent
    # calls the search_filing_text tool.
    # Each chunk: {"section": "mda", "filing_type": "10-K",
    #              "period_end": "2024-09-28", "text": "...", "search_query": "..."}
    # Empty or missing if the agent used XBRL tools directly (most common case).
    # Read by: response_generator (_format_sources adds RAG provenance badge),
    #          _format_metadata (data_sources["rag"] section).
    rag_chunks_retrieved: list[dict[str, Any]]

    # Result of cross-source verification via the News agent.
    # Set by run_sec_agent when the SEC agent calls corroborate_with_news tool.
    # Contains: {"news_verdict": "CONFIRMED", "news_confidence": 0.85,
    #            "news_reasoning": "Bloomberg reported...", "finding": "..."}
    # None if the agent didn't use A2A corroboration.
    # Read by: response_generator (_format_sources adds A2A provenance badge),
    #          _format_metadata (data_sources["a2a"] section).
    corroboration_result: Optional[dict[str, Any]]

    # Prior verification result injected as context for the agent.
    # Set by /verify route when the user chose "Verify With Context" after
    # /memory-check found a similar past claim (similarity >= 0.95).
    # Contains: {"request_id": "req_abc", "verdict": "SUPPORTS",
    #            "confidence": 0.92, "similarity": 0.97, "summary": "..."}
    # The agent sees this in its context: "A similar claim was previously
    # verified as SUPPORTS with 0.92 confidence."
    # None for fresh verifications (most common case).
    # Read by: base.py _build_context() (adds to agent prompt).
    memory_context: Optional[dict[str, Any]]

    # ===================================================================
    # AUDIT FIELDS — Accumulated throughout the pipeline
    # Multiple nodes append to these. Written to PostgreSQL at the end.
    # ===================================================================

    # Events accumulated specifically by guardrail nodes.
    # Each guardrail node appends its audit event to this list:
    #   state.get("audit_events", []) + [new_event]
    # Read by: audit commit at the end of the pipeline.
    audit_events: list[dict[str, Any]]

    # Complete chronological log of all events across all nodes.
    # Initialized as [] by /verify route. Nodes append events throughout.
    # Written to PostgreSQL audit_events table and audit_executions.full_trace
    # at the end of the pipeline via audit.commit_execution().
    audit_trail: list[dict[str, Any]]

    # Running counter of LLM tokens consumed across all LLM calls.
    # Initialized as 0 by /verify route. Incremented by claim_parser and
    # agent execution. Typical values: 3000-5000 tokens per verification.
    # Written to response metadata for cost tracking.
    total_tokens_used: int

    # ISO timestamp of when processing started. Set by /verify route.
    # Same value as timestamp_received — duplicated here so pipeline nodes
    # can compute elapsed time without knowing about the API layer.
    # Read by: response_generator (to compute execution_time_ms).
    execution_start_time: str

    # ISO timestamp of when processing completed. Set by response_generator.
    # None until the pipeline finishes. Used with execution_start_time to
    # calculate total execution time.
    execution_end_time: Optional[str]
