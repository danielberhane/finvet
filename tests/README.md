# FinVet test suite

## Run

```bash
.venv/bin/python -m pytest tests/unit -q          # ~1,560 tests, well under a minute, no services
.venv/bin/python -m pytest tests -q               # same: integration is excluded by default
.venv/bin/python -m pytest tests/integration -q -m integration   # opt in; needs live services
```

Always use `.venv/bin/python`, never bare `python`. Run from a clean shell:
sourcing `.env.minimax` (or any non-default env file) leaks `LLM_*__MODEL`
into the test environment and fails the tests that assert the DeepSeek
defaults.

The CI command, which is also the release gate:

```bash
uv run pytest tests/unit -q --cov=src/finvet --cov-report=json:coverage.json --cov-fail-under=75
uv run python scripts/check_critical_coverage.py coverage.json   # per-file floors on six modules
```

## Layout

```
tests/
├── conftest.py          # sets DEEPSEEK_API_KEY / TAVILY_API_KEY / POSTGRES_PASSWORD
│                        # defaults so Settings constructs without a .env; no fixtures
├── unit/                # 84 files; no network, no database
├── integration/         # 11 files, all marked `integration`; each self-skips when
│   └── conftest.py      # its service is absent. Loads the repo .env (override=True)
│                        # so real keys beat the root conftest's dummies
├── golden/              # empty scaffold — the real harness is described below
├── accuracy/            # RAG relevance cases + the Release A manifest (data only)
├── fixtures/            # XBRL concept table, filing chunks, two sample filings
├── performance/         # empty
└── security/            # empty
```

Registered markers: `integration` only. `addopts = -m 'not integration'` in
`pyproject.toml` keeps it out of a default run.

## What each integration file needs

| File | Needs |
|---|---|
| `test_database_connection.py`, `test_review_race.py` | Postgres |
| `test_rag_retrieval.py`, `test_rag_release_a.py` | Postgres with ingested chunks + Ollama embedder |
| `test_api.py`, `test_qualitative_filing_claims.py` | a running API (`FINVET_API_URL`, default `http://localhost:8000`) |
| `test_claim_matrix.py` | a running API with the full live stack — it re-spends on every run |
| `test_output_guard_false_positive.py` | Ollama + Llama Guard |
| `test_reject_classification.py` | DeepSeek key + Ollama + `FINVET_EVAL_DATA_DIR` |
| `test_xbrl_retrieval.py` | the SEC EDGAR MCP server + network + `FINVET_EVAL_DATA_DIR` |
| `test_golden.py` | nothing live: a recorded run artifact via `FINVET_GOLDEN_RUN` or `FINVET_GOLDEN_DIR` |

Two `xfail(strict=True)` markers document one live defect (a Llama Guard
S6 false positive on a factual refutation).

## The golden benchmark

`scripts/run_golden.py` runs the held-out set through a live API and
records; it asserts nothing. `tests/integration/test_golden.py` reads the
recorded artifact and judges it under the strict / safe / observe strength
contract. `scripts/eval_layers.py` reports the measurement layers over
recorded runs. The dataset is private; redacted run artifacts and the
dataset card live in `docs/eval/`.

## The one rule

**Tests must drive the producer.** A test that covers a verdict, an
escalation, or a persistence path needs at least one case whose entry point
is a route callable, a graph node, or a decorated tool, not a hand-built
dict. Three defects once survived a suite of hundreds of tests because their
tests constructed their own inputs; a fixture asserts the shape you
remembered, not the shape the system emits. The timeout tests in
`test_mcp_timeout.py` and `test_llm_factory.py` are recent examples: the
original test passed an explicit value and never saw the default path that
was broken.
