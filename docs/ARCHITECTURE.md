# FinVet -- System Architecture

## 1. High-Level Architecture

> **See**: `docs/diagrams/finvet-linkedin.png` for the visual pipeline diagram.

```
Streamlit UI (:8501)
       |  HTTP
FastAPI API (:8000)
       |
LangGraph Pipeline (12-node DAG + HITL checkpoint)
       |
  SEC Agent ------> SEC EDGAR MCP (:9870) ---> XBRL/Filing Data
  ^    |                                            |
  |    +---------> RAG (pgvector + tsvector + RRF) <+
  |                                                 |
  | A2A: corroborate_with_filing                    |
  |                                                 |
  News Agent -----> Tavily News Search              |
                                                    |
  Market Agent ---> Finnhub REST                    |
                                                    |
                   PostgreSQL 16 + pgvector (:5432) |
                   [audit_events, audit_executions, |
                    claim_memory, filing_chunks] <--+
```

---

## 2. Component Breakdown

### 2.1 API Layer (FastAPI, port 8000)

**Entry**: `src/finvet/main.py` -- creates app, compiles LangGraph, registers routers.

| Method | Path | Purpose | Audit Events |
|--------|------|---------|-------------|
| `GET` | `/health` | Health check | -- |
| `GET` | `/stats` | Service info, agents, truth regimes | -- |
| `POST` | `/verify` | Main claim verification | `input_received`, `commit_execution` |
| `GET` | `/reviews` | List pending HITL reviews | -- |
| `POST` | `/review/{id}` | Submit HITL decision | `hitl_{decision}` |
| `POST` | `/memory-check` | Search similar past verifications | -- |
| `POST` | `/memory-accept` | Log cache acceptance | `memory_cache_accepted` |
| `GET` | `/audit/{id}` | Full audit trail for a request | -- |
| `POST` | `/search-history` | Find past verifications (7-day window) | -- |

**Request/Response Models** (Pydantic v2, `src/finvet/api/models.py`):
- `VerifyClaimRequest`: claim (10-2000 chars), optional user_id, optional memory_context
- `MemoryCheckRequest`: claim (10-2000 chars)
- `MemoryAcceptRequest`: original_request_id, claim, similarity
- `HITLReviewRequest`: decision (approve/override/reject), optional override_verdict, optional reviewer_notes

**`/verify` Flow**:
1. Generate `request_id = "req_<12-hex>"`
2. Audit: log `input_received`
3. If memory_context provided: Audit: log `memory_context_injected`
4. Build initial `VerificationState` dict
5. `graph.invoke(state, config={thread_id: request_id})`
6. If HITL interrupted: return `status="pending_review"` + `preliminary_analysis`
7. If success: Audit `commit_execution()`, Memory `store()`, return `final_response`
8. `GuardrailViolation`: return 400 with `error_code`

**HITL Resume** (`POST /review/{id}`):
1. Validate decision (approve/override/reject)
2. Audit: log `hitl_{decision}`
3. Update state: `hitl_decision`, `hitl_override_verdict`, `hitl_reviewer_notes`
4. Resume: `graph.invoke(None, config)` -- graph continues from checkpoint
5. Fallback: if resume fails, compute verdict directly from audit DB

---

### 2.2 LangGraph Pipeline (12-Node DAG)

**File**: `src/finvet/graph/workflow.py`

Twelve nodes are registered; at most nine execute for a given claim, since the router
selects one domain agent and the two HITL nodes run only below the confidence threshold.

Compiled with `MemorySaver` checkpointer for HITL. `interrupt_before=["hitl_checkpoint"]`.

#### Node Map

| # | Node | Source | Destination | Conditional | Purpose |
|---|------|--------|------------|-------------|---------|
| 1 | `input_guardrails` | ENTRY | claim_parser | No | Regex + Llama Guard safety validation |
| 2 | `claim_parser` | 1 | 3/5/6/7 | Yes (claim_type) | LLM extracts 7-field ParsedClaim |
| 3 | `period_resolver` | 2 (sec only) | 4 | No | Convert period text to canonical dates |
| 4 | `sec_agent` | 3 | 8 | No | ReAct SEC verification |
| 5 | `market_agent` | 2 (market) | 8 | No | ReAct market verification |
| 6 | `news_agent` | 2 (news) | 8 | No | ReAct news verification |
| 7 | `reject_handler` | 2 (reject) | 12 | No | Set verdict=REJECTED |
| 8 | `consensus` | 4/5/6 | 9 | No | Adjust confidence |
| 9 | `output_guardrails` | 8 | 10/12 | Yes (hitl_required) | Check confidence + output safety |
| 10 | `hitl_checkpoint` | 9 (needs_hitl) | 11 | No | Graph pauses (interrupt_before) |
| 11 | `apply_hitl_decision` | 10 | 12 | No | Apply reviewer decision |
| 12 | `response_generator` | 7/9/11 | END | No | Format final response |

#### Routing Logic

**After claim_parser** (`_route_after_parsing`):
- `claim_type == "sec"` --> `period_resolver`
- `claim_type == "market"` --> `market_agent`
- `claim_type == "news"` --> `news_agent`
- `claim_type == "reject"` --> `reject_handler`

**After output_guardrails** (`_route_after_guardrails`):
- `hitl_required == True` --> `hitl_checkpoint`
- `hitl_required == False` --> `response_generator`

#### Consensus Adjustments

| Condition | Adjustment | Constant |
|-----------|-----------|----------|
| `magnitude_diff >= 20%` | -0.10 | `CONSENSUS_LARGE_DIFF_PENALTY` |
| `magnitude_diff <= 2%` | +0.05 | `CONSENSUS_CLOSE_MATCH_BONUS` |
| `len(tools_called) >= 3` | +0.05 | `CONSENSUS_THOROUGH_BONUS` |
| Always | cap at 0.95 | `CONSENSUS_MAX_CONFIDENCE` |

Only applies magnitude adjustments for equality claims (`comparison == "eq"`).

---

### 2.3 State Object: VerificationState

**File**: `src/finvet/models/state.py` -- `TypedDict` with `total=False` (all fields optional).

| Pipeline Stage | Fields Written |
|---|---|
| **API init** | `claim_raw`, `user_id`, `request_id`, `timestamp_received`, `execution_start_time`, `memory_context` |
| **input_guardrails** | `claim_normalized`, `guard_result_input`, `guardrails_passed`, `guardrails_failed` |
| **claim_parser** | `parsed_claim` (ParsedClaim), `parser_used`, `total_tokens_used` |
| **period_resolver** | `canonical_period` (CanonicalPeriod), `period_assumptions` |
| **domain_agents** | `agent_evidence` (AgentEvidence), `agent_type`, `rag_chunks_retrieved`, `corroboration_result` |
| **consensus** | `verdict`, `confidence`, `confidence_label`, `confidence_adjustments`, `consensus_reasons` |
| **output_guardrails** | `hitl_required`, `hitl_triggers`, `guard_result_output` |
| **hitl_checkpoint** | `hitl_checkpoint_passed` |
| **apply_hitl_decision** | `hitl_applied`, `hitl_decision`, `hitl_override_verdict` |
| **response_generator** | `final_response`, `execution_end_time` |

#### ParsedClaim (7 fields)

| Field | Type | Example |
|-------|------|---------|
| `claim_type` | `"sec"/"market"/"news"/"reject"` | `"sec"` |
| `ticker` | `str or null` | `"AAPL"` |
| `value` | `float or null` | `94000000000` |
| `comparison` | `"eq"/"gt"/"gte"/"lt"/"lte" or null` | `"gt"` |
| `period` | `str or null` | `"Q4 2024"` |
| `currency` | `str or null` | `"USD"` |
| `reject_reason` | `"non_financial"/"question"/"incomplete" or null` | `null` |

#### AgentEvidence

| Field | Type |
|-------|------|
| `agent` | `"sec"/"market"/"news"` |
| `verdict` | `"SUPPORTS"/"REFUTES"/"NOT_ENOUGH_INFO"` |
| `confidence` | `float (0-1)` |
| `retrieved_value` | `float or null` |
| `magnitude_difference_percent` | `float or null` |
| `tools_called` | `list[str]` |
| `tool_calls_detail` | `list[dict]` |
| `provenance` | `list[dict]` (full results for RAG/A2A) |
| `reasoning` | `str` |
| `execution_time_ms` | `int` |

---

### 2.4 Agent Architecture (ReAct Loop)

**Base class**: `src/finvet/agents/base.py` -- `BaseVerificationAgent`

#### Execution Flow

```
1. Build context (claim + parsed info + period + company + memory context)
2. Initialize messages: [SystemMessage, HumanMessage(context)]
3. LOOP (max 5 iterations):
   |-- LLM.invoke(messages)
   |-- IF tool_calls:
   |     Execute each tool, truncate result to 50K chars
   |     If provenance tool: capture full result
   |     Append ToolMessage to messages
   |-- ELSE (no tool_calls): capture reasoning, break
4. VERDICT EXTRACTION (separate LLM call):
   |-- create_llm("verdict").with_structured_output(VerdictOutput, method="json_mode")
   |-- Skip agent's final reasoning (prevent anchoring bias)
   |-- Returns: VerdictOutput(verdict, confidence, reasoning, retrieved_value)
5. PYTHON VERDICT OVERRIDE (deterministic):
   |-- magnitude_diff = |claimed - retrieved| / max(|claimed|, |retrieved|) * 100
   |-- For equality: diff <= tolerance -> SUPPORTS, else REFUTES
   |-- For directional (gt/gte/lt/lte): boolean comparison
   |-- Tolerances: SEC >$1B: 1%, SEC <$1B: 2%, Market: 5%, News: 5%
6. Return AgentEvidence dict
```

#### Three Domain Agents

| Agent | File | Tools | Data Sources | Provenance |
|-------|------|-------|-------------|------------|
| **SECAgent** | `agents/sec_agent/react_agent.py` | `get_company_info`, `get_recent_filings`, `get_income_statement`, `get_balance_sheet`, `get_cash_flow`, `search_filing_text` | SEC EDGAR (XBRL), RAG | `{"search_filing_text"}` |
| **MarketAgent** | `agents/market_agent/react_agent.py` | `get_stock_quote`, `get_daily_prices`, `get_company_overview`, `get_earnings` | Finnhub | (none) |
| **NewsAgent** | `agents/news_agent/react_agent.py` | `search_financial_news`, `verify_news_source`, `corroborate_with_filing` | Tavily, A2A | `{"corroborate_with_filing"}` |

System prompts: git-tracked text files in `src/finvet/agents/prompts/`.

---

### 2.5 Tools & External Services

#### SEC Tools --> MCP Server (port 9870)

**Transport**: Streamable-HTTP (JSON-RPC 2.0). Generic `MCPClient` handles session init, tool calls, SSE parsing.

**Adapter**: `src/finvet/mcp/sec_edgar.py` -- typed wrapper. Uses `get_xbrl_concepts` MCP tool (accepts accession_number for period-specific data).

**XBRL Concept Mappings**:

| Statement | Key Concepts |
|-----------|-------------|
| Income | Revenues, CostOfRevenue, GrossProfit, OperatingIncomeLoss, NetIncomeLoss, EPS |
| Balance | Assets, Liabilities, StockholdersEquity, Cash, LongTermDebt, PPE |
| Cash Flow | NetCashFromOperations, CapEx, Dividends, NetCashFromInvesting/Financing |

#### Market Tools --> Finnhub

Direct REST API via `src/finvet/mcp/finnhub.py`. Rate limit: 60 req/min. Historical candles (`get_daily_prices`) require a paid Finnhub tier; the free tier returns 403, which surfaces as an error rather than falling back to another provider. Mock mode available (`FINNHUB_MOCK_MODE=true`).

#### News Tools --> Tavily

SDK wrapper in `src/finvet/tools/tavily_search.py`. Source credibility tiers:
- **Tier 1** (0.95): Reuters, Bloomberg, WSJ, FT, AP, SEC.gov
- **Tier 2** (0.80): CNBC, MarketWatch, Yahoo Finance
- **Tier 3** (0.50): Unknown sources

#### RAG (Hybrid Search)

**File**: `src/finvet/tools/filing_search.py` + `src/finvet/rag/service.py`

For narrative filing text NOT in structured XBRL (segment revenue, risk factors, MD&A, footnotes).

```
Query --> OpenAI Embed (1536-dim)
            |
     +------+------+
     |             |
  pgvector      tsvector
  cosine        BM25
  (HNSW)        (GIN)
  top 20        top 20
     |             |
     +------+------+
            |
  Reciprocal Rank Fusion (k=60)
  score = 1/(60+rank_vec) + 1/(60+rank_kw)
            |
       top_k results
```

**Indexable sections**: business, risk_factors, mda, market_risk, financial_statements_and_notes, cybersecurity, controls_and_procedures, executive_compensation, legal_proceedings.

**Chunking**: tiktoken cl100k_base, max 500 tokens, 100 token overlap.

#### A2A (Agent-to-Agent Corroboration)

**Files**: `src/finvet/tools/corroborate_sec.py`, `src/finvet/models/a2a.py`

Delegation runs **News -> SEC only**. A news claim about a fine or settlement is checked
against the issuer's own filing: a 10-K's Legal Proceedings section is the primary source and
press coverage is secondary. The reverse direction existed once and was removed — it fired 0
times in 496 benchmark runs, because an audited filing is already the strongest source and the
claims a filing cannot settle parse as `news` and never reach the SEC agent.

Two triggers, one field. `trigger_mode="agent"` when the model called the tool itself (the
result is lifted from ReAct provenance); `"policy"` when `run_news_agent` invoked it after the
loop for a `CORROBORATION_METRICS` claim with a ticker — that call is absent from the message
history, so the node attaches the result explicitly.

```
News Agent ReAct loop
  +-- Verifies the event via search_financial_news()
  +-- Calls corroborate_with_filing(finding, ticker, claimed_value, operator, period)
       +-- Temporal gate: a filing that closed before the event cannot cover it
       |    -> NOT_APPLICABLE_YET, no nested run
       +-- Builds a real ParsedClaim carrying claimed_value, so _apply_override runs
       +-- run_sec_agent_scoped(state, max_iterations=3)  <- same period targeting
       +-- Returns A2AResult.model_dump(): status, verdict, retrieved_value, sources
  +-- run_news_agent writes it to corroboration_result
```

Recursion is structurally impossible: the SEC agent holds no delegation tool, so News -> SEC
terminates by construction. `status=CONTRADICTS` adds the `source_disagreement` HITL trigger;
`NO_MATCHING_DISCLOSURE` and `NOT_APPLICABLE_YET` do not — filing silence is expected from a
point-in-time document.

---

### 2.6 PostgreSQL Schema (4 Tables)

**Image**: `pgvector/pgvector:pg16` via Docker Compose.

#### `audit_events` (Append-Only Event Log)

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | `VARCHAR(50)` PK | `"evt_<8-hex>"` |
| `request_id` | `VARCHAR(50)` | Indexed |
| `parent_event_id` | `VARCHAR(50)` | FK to self |
| `event_type` | `VARCHAR(50)` | Indexed |
| `timestamp` | `VARCHAR(50)` | Indexed |
| `agent` | `VARCHAR(50)` | sec/market/news |
| `data` | `JSONB` | Event payload |
| `created_at` | `VARCHAR(50)` | |

**Event types**: `input_received`, `period_resolved`, `output_guardrails_checked`, `hitl_checkpoint_reached`, `hitl_approved`, `hitl_overridden`, `hitl_rejected`, `memory_cache_accepted`, `memory_context_injected`, `guardrail_violation`

#### `audit_executions` (Execution Summary)

| Column | Type | Notes |
|--------|------|-------|
| `execution_id` | `INTEGER` PK | Auto-increment |
| `request_id` | `VARCHAR(50)` UNIQUE | |
| `claim_text` | `TEXT` | |
| `claim_hash` | `VARCHAR(64)` | SHA256, indexed |
| `verdict` | `VARCHAR(50)` | Indexed |
| `confidence` | `FLOAT` | |
| `agents_run` | `JSONB` | `["sec"]` |
| `execution_time_ms` | `INTEGER` | |
| `execution_hash` | `VARCHAR(64)` | SHA-256 integrity checksum over `full_trace`; re-verified on read |
| `full_trace` | `JSONB` | The canonical execution envelope — exactly the object the checksum covers |
| `data_sources` | `JSONB` | `{xbrl, rag, a2a}` GIN indexed |

#### `claim_memory` (Vector Store)

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` PK | |
| `request_id` | `VARCHAR(50)` UNIQUE | |
| `claim_text` | `TEXT` | |
| `ticker` | `VARCHAR(10)` | Indexed |
| `agent_type` | `VARCHAR(20)` | Indexed |
| `verdict` | `VARCHAR(30)` | Indexed |
| `confidence` | `FLOAT` | |
| `retrieved_value` | `FLOAT` | |
| `summary` | `TEXT` | |
| `embedding` | `JSONB` | 1536 floats (OpenAI) |

#### `filing_chunks` (RAG Store)

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` PK | |
| `ticker` | `VARCHAR(10)` | Composite index |
| `cik` | `VARCHAR(20)` | |
| `filing_type` | `VARCHAR(10)` | |
| `period_end` | `VARCHAR(20)` | |
| `section` | `VARCHAR(50)` | Indexed |
| `chunk_text` | `TEXT` | |
| `token_count` | `INTEGER` | |
| `embedding` | `Vector(1536)` | HNSW index (m=16, ef=64) |
| `tsv` | `tsvector` GENERATED | GIN indexed |

---

### 2.7 Memory System (Episodic Claim Store)

**File**: `src/finvet/memory/service.py`

Embeddings: OpenAI `text-embedding-3-small` (1536 dims). Similarity: NumPy cosine.

| Mode | Threshold | Use Case |
|------|-----------|----------|
| **Cache** | >= 0.95 | Near-identical claim, skip pipeline |
| **Context** | >= 0.75 | Inject prior result as agent context |
| **Similar** | >= 0.60 | "People Also Verified" UI display |

**Write**: After successful verification, embed claim, store in `claim_memory`. Skips PENDING/REJECTED/ERROR.

**Read** (`/memory-check`): Embed query, fetch 1000 candidates, NumPy cosine similarity, return sorted matches.

---

### 2.8 Guardrail System

**File**: `src/finvet/guards/`

| Provider | Type | Speed | Checks |
|----------|------|-------|--------|
| **RegexGuardProvider** | Always on | <1ms | Injection patterns, PII (SSN/CC/email/phone), length (10-2000), NFKC normalization, English detection |
| **LlamaGuardProvider** | Optional | ~200ms | Llama Guard 3 via Ollama. S1-S13 taxonomy. S6 customized for financial domain |
| **FinancialGuardProvider** | Output only | <1ms | Investment advice patterns ("you should buy", "price target", etc.) |
| **CompositeGuardProvider** | Chain | Sum | Runs providers in order, short-circuits on failure |

**Graceful degradation**: If Ollama is down, LlamaGuard returns safe=True with flag `"llama_guard_unavailable"`.

**Violation types**: `INJECTION_DETECTED`, `PII_DETECTED`, `CLAIM_TOO_SHORT`, `CLAIM_TOO_LONG`, `UNSUPPORTED_LANGUAGE`, `LLAMA_GUARD_UNSAFE`

---

### 2.9 LLM Configuration

**File**: `src/finvet/llm/factory.py`

Three independently configurable endpoints:

| Purpose | Default Model | Temperature | Used By |
|---------|--------------|-------------|---------|
| `parser` | deepseek-chat | 0.0 | Claim parser |
| `agent` | deepseek-chat | 0.0 | ReAct loop |
| `verdict` | deepseek-chat | 0.0 | Verdict extraction |

DeepSeek does NOT support `json_schema` response_format. Uses `method="json_mode"` with schema in prompt text.

Configurable via `LLM_PARSER__MODEL`, `LLM_AGENT__TEMPERATURE`, etc.

---

### 2.10 Constants

**File**: `src/finvet/config/constants.py`

| Category | Constant | Value |
|----------|----------|-------|
| **Tolerances** | `TOLERANCE_SEC_LARGE` | 1% (>$1B) |
| | `TOLERANCE_SEC_SMALL` | 2% (<=$$1B) |
| | `TOLERANCE_MARKET` | 5% |
| | `TOLERANCE_NEWS` | 5% |
| **Consensus** | `CONSENSUS_LARGE_DIFF_PENALTY` | -0.10 |
| | `CONSENSUS_CLOSE_MATCH_BONUS` | +0.05 |
| | `CONSENSUS_THOROUGH_BONUS` | +0.05 |
| | `CONSENSUS_MAX_CONFIDENCE` | 0.95 |
| **Agent** | `AGENT_MAX_ITERATIONS` | 5 |
| | `AGENT_MAX_RESULT_CHARS` | 50,000 |
| **Confidence** | `CONFIDENCE_HIGH_THRESHOLD` | 0.85 |
| | `CONFIDENCE_MODERATE_THRESHOLD` | 0.70 |
| **RAG** | `EMBEDDING_MODEL` | text-embedding-3-small |
| | `EMBEDDING_DIMS` | 1536 |
| | `RRF_K` | 60 |
| **Memory** | `MEMORY_CACHE_THRESHOLD` | 0.95 |
| | `MEMORY_CONTEXT_THRESHOLD` | 0.75 |
| | `MEMORY_SIMILAR_THRESHOLD` | 0.60 |

---

## 3. Key Design Decisions

### ReAct + Separate Verdict + Python Override

Three-phase verdict extraction prevents LLM number comparison errors:
1. **ReAct loop**: Agent reasons freely, calls tools
2. **Verdict LLM**: Separate structured call (skips agent's final reasoning to prevent anchoring)
3. **Python override**: Deterministic comparison with tolerance thresholds

### Hybrid RAG (pgvector + tsvector + RRF)

Combines semantic similarity with keyword matching. Prevents pure-semantic false positives on financial terms (e.g., "revenue" matching "cost of revenue").

### HITL Checkpoint with MemorySaver

LangGraph `interrupt_before` pauses graph. State persisted. API returns `pending_review`. Human review resumes via `update_state()` + `invoke(None, config)`.

### Provenance for Audit Compliance

The SEC agent captures full untruncated RAG results and the News agent its A2A result. These flow through as `rag_chunks_retrieved` and `corroboration_result` into response metadata and audit DB `data_sources` JSONB.

### Non-Critical Memory and Similar Claims

All memory operations wrapped in try/except. Failures log warnings but never block verification.

---

## 4. Infrastructure

### Docker Compose

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | Audit, Memory, RAG |
| `sec-edgar-mcp` | Custom | 9870 | SEC EDGAR MCP server |

### Startup Order

1. PostgreSQL (`docker compose up -d postgres`)
2. SEC EDGAR MCP server (port 9870)
3. FastAPI (`uvicorn finvet.main:app --port 8000 --app-dir src`)
4. Streamlit UI (`streamlit run ui/app.py --server.port 8501`)

### Dependencies

Core: `langchain>=0.3`, `langgraph>=0.2`, `fastapi>=0.115`, `pydantic>=2.9`, `sqlalchemy>=2.0`, `pgvector`, `numpy`, `tiktoken`, `httpx`, `tavily-python>=0.5`

---

## 5. Module Dependency Graph

```
config/
  settings.py      <-- .env
  constants.py     <-- (standalone)
  database.py      <-- settings

llm/
  factory.py       <-- config/settings

models/
  claim.py         <-- (standalone Pydantic)
  state.py         <-- models/claim

agents/
  base.py          <-- config/constants, llm/factory, models/state
  prompts/*.txt    <-- (loaded at import time)
  sec_agent/       <-- base, tools/sec_tools, tools/filing_search, tools/memory_tools
  market_agent/    <-- base, tools/market_tools
  news_agent/      <-- base, tools/news_tools, tools/corroborate_sec

graph/
  nodes/           <-- agents, models/state
  workflow.py      <-- nodes, config/constants

tools/
  market_tools.py  <-- mcp/finnhub
  sec_tools.py     <-- mcp/sec_edgar
  news_tools.py    <-- tools/tavily_search
  filing_search.py <-- rag/service
  corroborate_sec.py <-- graph/nodes/domain_agents (runtime import)

rag/
  service.py       <-- config/constants, config/database
  parser.py        <-- (standalone)

memory/
  service.py       <-- config/constants, config/database

audit/
  logger.py        <-- audit/database
  database.py      <-- config/database

guards/
  composite.py     <-- regex.py, llama_guard.py, financial.py

api/
  routes/*.py      <-- api/deps, api/models, audit, memory, graph
  deps.py          <-- (module-level graph reference)
  models.py        <-- (standalone Pydantic)

main.py            <-- api/routes, graph/workflow, config
```

---

## 6. Extension Guide

### Adding a New Agent

1. Create `src/finvet/agents/<name>_agent/react_agent.py` extending `BaseVerificationAgent`
2. Add prompt to `src/finvet/agents/prompts/<name>_system.txt`
3. Create tools in `src/finvet/tools/<name>_tools.py`
4. Add wrapper to `src/finvet/graph/nodes/domain_agents.py`
5. Register node + routing in `src/finvet/graph/workflow.py`
6. Update `_route_after_parsing` with new claim_type

### Adding a New Tool

1. Create tool function with `@tool` decorator and Pydantic return model
2. Add to the agent's tool list in `tools/__init__.py`
3. If provenance needed, add tool name to `_provenance_tool_names`

### Adding a Guard

1. Implement `GuardProvider` protocol in `src/finvet/guards/`
2. Add to `CompositeGuardProvider` chain in the guardrails node

---

## 7. Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | -- | DeepSeek API key for LLM |
| `TAVILY_API_KEY` | Yes | -- | Tavily API key for news search |
| `POSTGRES_PASSWORD` | Yes | -- | PostgreSQL password |
| `FINNHUB_API_KEY` | No | -- | Finnhub API key (mock mode if absent) |
| `OPENAI_API_KEY` | No | -- | OpenAI key for embeddings (RAG + memory) |
| `FINNHUB_MOCK_MODE` | No | false | Use mock market data |
| `ENABLE_LLAMA_GUARD` | No | false | Enable Llama Guard semantic safety |
| `ENABLE_CLAIM_MEMORY` | No | true | Enable claim memory |
| `CONFIDENCE_THRESHOLD_HITL` | No | 0.70 | Below this triggers HITL |
| `LOG_LEVEL` | No | INFO | Logging level |
| `LLM_PARSER__MODEL` | No | deepseek-chat | Override parser LLM |
| `LLM_AGENT__MODEL` | No | deepseek-chat | Override agent LLM |
| `LLM_VERDICT__MODEL` | No | deepseek-chat | Override verdict LLM |

---

## 8. UI Output (Brief)

The Streamlit UI at port 8501 renders:
- **Verify page**: Claim input, example claims, pipeline progress, verdict cards (white bg + colored left border), evidence panel, data comparison strip, tool call details, source badges (XBRL/RAG/A2A), raw API expander
- **Memory match card**: Similar verification found (>=0.95), Use/Fresh/Context buttons
- **Pending Reviews page**: HITL-queued claims list
- **Review Detail page**: Preliminary analysis + approve/override/reject submission

---

## Diagrams

Architecture diagrams are in `docs/diagrams/`:
- `finvet-linkedin.png` / `finvet-linkedin.svg` -- LangGraph pipeline overview
