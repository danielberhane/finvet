# FinVet — Project Guide for Claude Code

## What This Is
Agentic financial claim verification: FastAPI + Streamlit + LangGraph.
Reference implementation of an agentic system in a regulated domain.
The design commitment: **the LLM is untrusted** — a deterministic comparator
decides verdicts, and a claim with no source-traced observation declines
rather than publishing the model's reading.

## Architecture
- **Pipeline**: input_guardrails → claim_parser → period_resolver →
  domain_agents → consensus → output_guardrails → response_generator
- **API** `src/finvet/main.py` + `api/routes/` (port 8000) ·
  **UI** `ui/app.py` (port 8501, absolute imports only)
- **Agents**: SEC, Market, News in `src/finvet/agents/`, all extend `base.py`
  (ReAct + deterministic verdict override). Prompts are text files in
  `agents/prompts/`.
- **A2A**: News → SEC only; SEC holds no delegation tool, so recursion is
  impossible by construction. Two triggers: the model calls
  `corroborate_with_filing`, or the policy path fires for
  `CORROBORATION_METRICS`.
- **State**: `VerificationState` in `models/state.py`.
  All named constants in `config/constants.py`.
- **HITL**: MemorySaver checkpointer, `interrupt_before=["hitl_checkpoint"]`,
  resume via `update_state()` + `invoke(None, config)`. Pending reviews do
  NOT survive an API restart (409 `checkpoint_unavailable`).
- **Memory**: episodic claim memory, off by default (`enable_claim_memory`;
  rationale in RELEASE_A_DECISIONS.md D8). The API takes
  `memory_context_request_id` — an identifier, never free-form context; the
  free-form `memory_context` field was removed as a prompt-injection channel.
  Do not reintroduce it.
- Provenance: `_provenance_tool_names` on BaseVerificationAgent captures full
  tool results; response metadata reports `xbrl` / `rag` / `a2a` under
  `data_sources`.

## Commands
- Always `.venv/bin/python`, never bare `python`.
- API:   `.venv/bin/uvicorn finvet.main:app --host 0.0.0.0 --port 8000 --app-dir src`
- UI:    `.venv/bin/streamlit run ui/app.py --server.port 8501`
- Tests: `.venv/bin/python -m pytest tests/unit -q`
- All:   `./start.sh` (Postgres, SEC MCP, API, UI)
- Containers: `docker compose --profile sec up --build`
  (docker/Dockerfile = API+UI, docker/Dockerfile.sec = SEC MCP; `sec` is the
  only compose profile)

## Critical Gotchas (each exists because something broke)
1. **DeepSeek** does not support `json_schema` — use `method="json_mode"`
   with the schema in the prompt text.
2. **Streamlit HTML** breaks on blank lines inside `st.markdown` blocks —
   build HTML with list-append + `"\n".join()`, never f-string templates
   with embedded newlines.
3. **`ui/` needs absolute imports** (`from api_client import ...`) —
   `streamlit run` adds `ui/` to sys.path but does not treat it as a package.
4. **pgvector + SQLAlchemy**: `CAST(:query_vec AS vector)`, never `::vector`.
5. **pydantic-settings** has `extra="ignore"` — misspelled env vars vanish
   silently instead of erroring.
6. **Tests must drive the producer.** Any test covering a verdict, an
   escalation, or a persistence path needs at least one case whose entry
   point is a route callable, a graph node, or a decorated tool — not a
   hand-built dict. Three defects survived a 500-test suite because their
   tests constructed their own inputs: a fixture asserts the shape you
   remembered, not the shape the system emits.
7. **The parser contract is 7 fields** (`claim_type, ticker, metric,
   operator, value, period, reject_reason`). `range` is a legal operator
   with no code path — it fails closed to NOT_ENOUGH_INFO. Never add
   midpoint comparison: it refutes true claims whose band exceeds the
   tolerance (RELEASE_A_DECISIONS.md D18).
8. **Sourcing `.env.minimax` then running pytest gives ~8 false failures**
   (`LLM_*__MODEL` leaks into the test env and breaks tests that assert the
   DeepSeek defaults). Run tests from a clean shell.

## Evaluation
- Golden dataset: `~/Projects/Active/finvet-golden/golden_c.jsonl`
  (97 rows, frozen, its own git repo). Ids have gaps at 35/36/97 — never
  renumber.
- **`run-*.json` files there are PAID artifacts — never overwrite or
  delete.** `scripts/eval_layers.py --json` is an OUTPUT path, not a
  selector; use `--label` to choose runs.
- Layers live in `src/finvet/eval/measures/` (reliability, calibration,
  grounding, risk, trajectory, routing). `scripts/run_golden.py` records and
  asserts nothing; `tests/integration/test_golden.py` judges artifacts.
  `tests/golden/` is an empty scaffold — the real harness is the above.
- Benchmark runs: both repos clean at recorded SHAs; the model label comes
  from `/health` (the serving process, not the client shell) and the runner
  refuses to start on model drift.

## Hooks (deterministic guards — a block is intended, not a malfunction)
Three PreToolUse/PostToolUse hooks in `.claude/hooks/`, each written after a
real incident (history in each script's header and in git log):
- `protect-artifacts.sh` — blocks writes/moves/deletes touching the golden
  repo's `run-*.json` or dataset. Sanctioned path: copy to the scratchpad,
  or ask the user to run the command.
- `protect-secrets.sh` — blocks editing env/credential files and printing
  them raw. Sourcing them is fine; to inspect, pipe through
  `sed 's/=.*/=REDACTED/'`.
- `ruff-on-edit.sh` — lints every edited `.py` and feeds findings back; fix
  them before moving on.
Never attempt to route around a block — the guard firing means the action
was in the exact class that caused the original incident.

## Code Style
- Keep changes minimal. Don't refactor code you didn't change.
- No unnecessary comments, docstrings, or type annotations on untouched code.
- Pydantic v2 models; SQLAlchemy ORM only — never raw SQL interpolation.
- HTML in Streamlit: always the list-append pattern.
