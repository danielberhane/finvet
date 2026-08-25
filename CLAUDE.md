# FinVet — Project Guide for Claude Code

## What This Is
Agentic financial claim verification system. FastAPI backend + Streamlit UI + LangGraph pipeline.
Built as a reference implementation of an agentic AI system in a regulated domain: multi-agent orchestration, tool use, guardrails, HITL, and a full audit trail.

## Architecture
- **API**: `src/finvet/main.py` → routes in `src/finvet/api/routes/` (FastAPI, port 8000)
- **UI**: `ui/app.py` → pages in `ui/views/`, components in `ui/components/` (Streamlit, port 8501)
- **Pipeline**: LangGraph — input_guardrails → claim_parser → period_resolver → domain_agents → consensus → output_guardrails → response_generator
- **Agents**: SEC, Market, News in `src/finvet/agents/` — all extend `base.py` (ReAct loop + structured verdict)
- **Prompts**: Text files in `src/finvet/agents/prompts/` (sec_system.txt, market_system.txt, news_system.txt, parser_system.txt)
- **Constants**: All magic numbers in `src/finvet/config/constants.py` (tolerances, consensus, agent limits, thresholds)
- **State**: `VerificationState` TypedDict in `src/finvet/models/state.py`
- **RAG**: Hybrid search (pgvector cosine + tsvector BM25, RRF k=60) in `src/finvet/rag/`
- **A2A**: News -> SEC only. `corroborate_with_filing` in `src/finvet/tools/corroborate_sec.py`, contract in `src/finvet/models/a2a.py`. Two triggers: the model calls it, or `run_news_agent`'s policy path does for `CORROBORATION_METRICS`.
- **Memory**: `src/finvet/memory/` — episodic claim memory via LangGraph `PostgresStore` with pgvector HNSW indexing; `nomic-embed-text` (768-dim) served locally by Ollama, similarity computed in Postgres. `main.py` passes `rag.service._embed_texts` into the store, so RAG and memory share one embedding path. Gated on `enable_claim_memory`; degrades to disabled if unavailable. No hosted embedding provider is used.
- **Audit**: PostgreSQL via SQLAlchemy in `src/finvet/audit/`. Tables created via `init_db()`. Thread-safe request-scoped event buffer.
- **HITL**: MemorySaver checkpointer, `interrupt_before=["hitl_checkpoint"]`, resume via `update_state()` + `invoke(None, config)`

## Project Structure
```
src/finvet/
  main.py                    # App creation, startup (~80 lines)
  api/
    models.py                # Request/response Pydantic models
    deps.py                  # Shared verification_graph reference
    routes/
      health.py              # /, /health, /stats
      verify.py              # /verify
      review.py              # /review/{id}, /reviews
      memory.py              # /memory-check, /memory-accept
      audit.py               # /audit/{id}, /search-history
  config/
    settings.py              # Pydantic Settings from .env
    constants.py             # All named constants (tolerances, thresholds, limits)
  agents/
    base.py                  # BaseVerificationAgent (ReAct + verdict override)
    prompts/                 # System prompt text files (git-tracked, versionable)
    sec_agent/               # SEC domain agent
    market_agent/            # Market domain agent
    news_agent/              # News domain agent
  graph/
    nodes/                   # Pipeline node functions
    workflow.py              # LangGraph compilation + routing
  tools/                     # Agent tools (market, sec, news, filing_search, corroborate)
  rag/                       # Hybrid RAG (pgvector + tsvector + RRF); ingest via `python -m finvet.rag.ingest`
  eval/                      # Eval harness — xbrl_retrieval, reject_classification, dataset, gold-fill
  memory/                    # Claim memory (embeddings + cosine similarity)
  audit/                     # PostgreSQL audit trail (thread-safe)
  guards/                    # Input guardrails (regex + Llama Guard)
  llm/                       # LLM factory
  mcp/                       # MCP clients (SEC EDGAR, Finnhub)
  models/                    # Pydantic models (claim, state, verdict, audit)
  utils/                     # Logging, exceptions, helpers

ui/
  app.py                     # Streamlit entry point, page routing
  styles.py                  # CSS styles constant
  api_client.py              # HTTP wrapper for FinVet API
  components/
    formatting.py            # format_value, _escape, _md_inline
    source_badges.py         # _data_source_badges_html
    evidence.py              # render_evidence, _parse_reasoning_sections
  views/
    verify.py                # Verify Claim page
    reviews.py               # Pending Reviews list
    review_detail.py         # Single review detail + HITL submission
    audit_list.py            # Audit trail browser
    audit_detail.py          # Single run, node by node

tests/
  unit/                      # Mocked tests (480 tests, no .env or DB needed)
  integration/               # End-to-end tests
  golden/                    # Regression tests (scaffold only)
```

## Development Environment
- Python venv: `.venv/` — always use `.venv/bin/python` to run
- Start API: `.venv/bin/uvicorn finvet.main:app --host 0.0.0.0 --port 8000 --app-dir src`
- Start UI: `.venv/bin/streamlit run ui/app.py --server.port 8501`
- Run tests: `.venv/bin/python -m pytest tests/unit/ -v`
- Start all: `./start.sh` (PostgreSQL, SEC MCP, API, UI)
- Version constant: `__version__` in `src/finvet/__init__.py` (currently 2.0.9)
- Architecture docs: `docs/ARCHITECTURE.md`
- Validation strategy: `docs/VALIDATION_STRATEGY.md`

## Critical Gotchas (DO NOT FORGET)
1. **DeepSeek API** (`deepseek-chat`) does NOT support `json_schema` response_format. Use `method="json_mode"` with `.with_structured_output()` and include schema in prompt text.
2. **Streamlit HTML**: `st.markdown(html, unsafe_allow_html=True)` breaks on **blank lines** inside HTML blocks. Always build HTML with list-append + `"\n".join()`, never f-string templates with embedded newlines.
3. **Streamlit imports**: `ui/` files must use **absolute imports** (`from api_client import ...`, `from components.formatting import ...`), NOT relative imports (`from .api_client`, `from ..components`). `streamlit run ui/app.py` adds `ui/` to `sys.path` but does NOT treat it as a package, so relative imports fail.
4. **SQLAlchemy + pgvector**: `::vector` cast conflicts with SQLAlchemy. Use `CAST(:query_vec AS vector)` instead.
5. **Settings model**: `extra="ignore"` in pydantic-settings — unknown env vars are silently ignored.
6. **Tests must drive the producer.** Any test covering a verdict, an escalation, or a
   persistence path needs at least one case whose entry point is a route callable, a graph
   node, or a decorated tool — not a hand-built dict. Three defects survived a 500-test suite
   because their tests constructed their own inputs: the streaming route never committed an
   audit row, delegation compared a verdict with itself, and an escalation could not fire on
   the only path that reaches it. A fixture you wrote asserts the shape you remembered, not
   the shape the system emits.

## Code Style
- Keep changes minimal. Don't refactor code you didn't change.
- No unnecessary comments, docstrings, or type annotations on untouched code.
- Pydantic v2 models for request/response.
- SQLAlchemy ORM only — never raw SQL string interpolation.
- HTML in Streamlit: always list-append pattern, never f-string templates.

## Data Sources & Provenance
- `_provenance_tool_names` set on BaseVerificationAgent captures full tool results.
- Response metadata keys: `xbrl`, `rag`, `a2a` in `data_sources` dict.
- Audit DB: `data_sources` JSONB column on execution records.
- UI badges: blue XBRL, purple RAG, amber A2A.

## Pre-Pipeline Claim Memory
- `/memory-check` searches for similar past verifications (threshold >= 0.95).
- Three user choices: Use This Result (cache hit), Verify Fresh, Verify With Context.
- `memory_context` field on VerifyClaimRequest and VerificationState.
- Audit events: `memory_cache_accepted`, `memory_context_injected`.

## Containers
- `docker/Dockerfile` builds both API and UI; `docker/Dockerfile.sec` builds the SEC MCP server.
- `docker compose --profile sec up --build` brings up Postgres, API, UI, and SEC MCP.
- Optional profiles: `sec` (SEC EDGAR MCP), `guards` (Ollama for Llama Guard).
