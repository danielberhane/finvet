# Contributing

FinVet is a reference implementation maintained by one person. Bug reports, reproductions
and small focused fixes are the most useful contributions. Open an issue before starting
anything larger.

## Setup

```bash
uv sync                                        # dev tools come from the dev group
cp .env.example .env                           # fill in the required keys
.venv/bin/python -m pytest tests/unit -q       # no services needed
```

Use `.venv/bin/python`, never bare `python`. Run tests from a clean shell: an env file
that overrides `LLM_*__MODEL` leaks into the test environment and fails the tests that
assert the default provider.

## Running natively

The README's `docker compose` quickstart runs everything in containers. For hot reload,
run the API and UI from the checkout and keep only the services in containers:

```bash
docker compose --profile sec up -d postgres sec-edgar-mcp      # pgvector, SEC EDGAR MCP (:9870)
.venv/bin/uvicorn finvet.main:app --port 8000 --app-dir src    # API → :8000
.venv/bin/streamlit run ui/app.py --server.port 8501           # UI  → :8501
```

Embeddings come from a local Ollama with `nomic-embed-text` pulled. The keys each feature
needs are listed in `.env.example` and in `docs/ARCHITECTURE.md` under the configuration
reference.

### Loading filings

The filing-text index starts empty; filings are not distributed with the repo. XBRL
verification works without it. Claims about a filing's narrative sections need an ingest
pass:

```bash
.venv/bin/python scripts/fetch_filings.py --tickers AAPL,MSFT --years 2024   # downloads to data/filings/
.venv/bin/python -m finvet.rag.ingest                                        # defaults to data/filings
```

SEC requires a contact address on automated requests; set `SEC_EDGAR_USER_AGENT` in `.env`
first.

## What CI enforces

- `ruff check .`
- `mypy` on six trust-boundary modules
- a check that every `src/finvet/**.py` path named in the docs exists
- `pytest tests/unit` on Python 3.11, 3.12 and 3.13, with coverage at or above 75% overall
  and per-file floors on six modules
- a build of all three Docker images

Integration tests are excluded by default and skip themselves when their service is
absent, so you do not need Postgres or the MCP server to contribute.

## Tests must drive the producer

A test covering a verdict, an escalation or a persistence path enters through a route
callable, a graph node or a decorated tool, never a hand-built dictionary. A fixture
asserts the shape you remembered, not the shape the system emits.

## Changes that will be declined

- **Midpoint comparison for range claims.** It refutes true claims whose band exceeds the
  tolerance. Range claims decline on purpose.
- **A free-form context field on the verify request.** It was removed as a prompt-injection
  channel; the API takes an identifier the server resolves itself.
- **Letting the model decide a numeric verdict.** The comparison is computed from a
  structured observation, and a claim without one declines.

## Evaluation

The claim set is held out privately, so benchmark runs are not reproducible from a fork.
The published artifacts in `docs/eval/runs/` are what every reported number recomputes
from. If a change touches anything the evaluation measures, say so in the PR.

## Commits

Say why, not just what. Most of the odd-looking code here exists because something broke;
name the incident or the behaviour that motivated the change.
