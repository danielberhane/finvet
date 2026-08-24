# FinVet

**Agentic financial claim verification.**

<!-- Restore after the first push, once CI has run once:
[![ci](https://github.com/danielberhane/finvet/actions/workflows/ci.yml/badge.svg)](https://github.com/danielberhane/finvet/actions/workflows/ci.yml)
-->
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

Give it a claim — *"Apple's fiscal 2024 revenue was $391 billion"* — and it returns a verdict
(`SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`), a confidence score, the evidence chain, and an
audit trail of every step that produced the answer.

> **Not financial advice.** A research and demonstration system; outputs may be wrong and must
> be independently verified. Not affiliated with any data provider named here.

---

## Deterministic verdict override

LLMs are unreliable at numeric comparison — a frontier model will confidently claim `0.98%`
exceeds `1%`. In claim verification, the task *is* comparing a claimed number to a filed one.
When both the claimed and retrieved values exist, FinVet recomputes the comparison in Python
with source-appropriate tolerances and overrides the model when they disagree. Both results
are recorded:

```python
{"verdict": "REFUTES", "llm_original_verdict": "SUPPORTS", "override_applied": True}
```

See [`_apply_override`](src/finvet/agents/base.py). This also limits prompt injection: injected
text can sway the model's prose, but not a numeric verdict.

---

## Architecture

A 12-node LangGraph `StateGraph`. Three domain agents fan out by claim type, each a ReAct loop
with its own tools; their evidence is reconciled, guarded, and — when confidence is low —
paused for human review.

<p align="center">
  <img src="docs/diagrams/finvet-linkedin.png" alt="Architecture" width="800">
</p>

| Agent | Handles | Sources |
|---|---|---|
| **SEC** | GAAP financials — revenue, income, EPS, balance sheet, cash flow | SEC EDGAR (XBRL) via MCP, hybrid RAG over filings |
| **Market** | Prices, valuation, market cap | Finnhub |
| **News** | Events, announcements, macro indicators | Tavily search, FRED |

Deeper dive: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

The whole system — Postgres, API, UI, and the SEC MCP server — comes up with one command. The
API creates its schema on first boot.

```bash
git clone https://github.com/danielberhane/finvet.git && cd finvet
cp .env.example .env            # add DEEPSEEK_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD
docker compose --profile sec up --build
# UI → http://localhost:8501   API → http://localhost:8000
```

Prebuilt images: `docker pull ghcr.io/danielberhane/finvet-api:latest` (and `finvet-ui`).

The system verifies against live financial data, so real API keys are required. Only
`DEEPSEEK_API_KEY`, `TAVILY_API_KEY`, and `POSTGRES_PASSWORD` are mandatory; when an optional
key is absent, the corresponding feature is disabled rather than crashing.

<details>
<summary><b>Native dev · compose profiles · which key powers what</b></summary>

### Native (hot reload)

```bash
uv sync
cp .env.example .env
docker compose up -d postgres                        # pgvector
uvicorn finvet.main:app --port 8000 --app-dir src    # API → :8000
streamlit run ui/app.py --server.port 8501           # UI  → :8501
```

### Populating the RAG index

The vector store starts empty — SEC filings are not distributed with the repo. XBRL
verification works without it; hybrid RAG over filing text needs an ingest pass:

```bash
# Place filings as data/filings/<TICKER>/<TICKER>_<FORM>_<PERIOD-END>.html
#   e.g. data/filings/AAPL/AAPL_10-K_2024-09-28.html
python -m finvet.rag.ingest                    # defaults to data/filings
python -m finvet.rag.ingest --dir data/filings/AAPL
```

Requires `OPENAI_API_KEY` (embeddings) and a running Postgres.

### Profiles

| Profile | Adds | For |
|---|---|---|
| *(default)* | Postgres + pgvector, API, UI | always |
| `--profile sec` | SEC EDGAR MCP (self-contained, from PyPI) | SEC claims (most demos) |
| `--profile guards` | Ollama | optional Llama Guard (~5 GB, opt-in) |

### Keys

| Key | Powers | Without it |
|---|---|---|
| `DEEPSEEK_API_KEY` | claim parsing, agents, verdicts | **required** — nothing runs |
| `TAVILY_API_KEY` | news search | news claims → NOT_ENOUGH_INFO |
| `FINNHUB_API_KEY` | market quotes, tickers | market claims → NOT_ENOUGH_INFO |
| `OPENAI_API_KEY` | embeddings → RAG + claim memory | both disabled; SEC still verifies via XBRL |
| *(none)* | FRED macro, SEC XBRL | free public endpoints |

The LLM is pluggable ([`llm/factory.py`](src/finvet/llm/factory.py)) — point any
OpenAI-compatible endpoint (self-hosted via vLLM/LiteLLM, or another vendor) at it via env
var, no code change.

</details>

---

## Features

- **Deterministic verdict override** — Python recomputes numeric comparisons; the LLM never
  has the final say on a number.
- **Hybrid RAG over filings** — pgvector dense retrieval + Postgres `tsvector` BM25, fused by
  reciprocal rank fusion.
- **Layered guardrails** — always-on regex/PII checks, plus an optional Llama Guard semantic
  layer, on both input and output. Safety is the guards' job; verifiability is the parser's.
- **Human-in-the-loop** — LangGraph `interrupt_before` pauses low-confidence or flagged
  verdicts for a reviewer, then resumes from the checkpoint with the decision merged in.
- **Full audit trail** — every node's I/O, tool calls, and data provenance persisted to
  Postgres with a SHA-256 hash per run; any past verification can be reconstructed node by
  node.
- **Agent-to-agent** — the SEC agent can call the news agent when filing data is ambiguous.
- **Claim memory** — embedding search over past verifications for caching and context.

---

## Evaluation

Measured against SEC primary-source values, not self-reported. *Silently wrong* — a confident
number that disagrees with the filing — is treated as the failure that matters.

| Metric | Result |
|---|---|
| XBRL retrieval accuracy | **198/199 (99.5%)**, zero silently-wrong (the one miss returns NOT_ENOUGH_INFO) |
| Reject classification (refusing unverifiable claims) | **64.2% recall at 100% precision** — never rejects a verifiable claim |

Retrieval falls back from SEC's period-targeted `companyconcept` read to the `frames`
endpoint when the first is empty for a company/concept — that fallback took accuracy from
~91% to 99.5%.

---

## Security & operations

Packaged for **local / demo use, not public hosting as-is.**

- **Unauthenticated API** — no auth, rate limiting, or CORS. Run on a trusted network; put an
  authenticating proxy in front before any exposure.
- **Graceful degradation** — with SEC MCP, Ollama, or OpenAI down, the API stays up and
  returns NOT_ENOUGH_INFO or routes to review rather than 500-ing.
- **Reproducible builds** — images build from a committed `uv.lock`, so CI, Docker, and dev
  resolve identical versions.
- **CI/CD** — [`ci.yml`](.github/workflows/ci.yml) runs lint + tests + Docker builds on every
  push; [`release.yml`](.github/workflows/release.yml) publishes versioned images to GHCR on
  a `v*` tag. There is intentionally no deploy-to-live stage.

*Production would add:* an auth gateway + rate limits, a secrets manager, managed/HA
Postgres, and an LLM cost budget.

---

## Limitations

- **US equities only** · **point-in-time claims** (no time series) · **latency 15–40s/claim**
  (three ReAct agents making real tool calls).
- The consensus step is **heuristic**, not learned.
- **Historical prices need a paid Finnhub tier**; on a free key the market agent reports
  NOT_ENOUGH_INFO rather than scraping around it.
- **Not a compliance product** — it applies model-risk-management *principles*; it certifies
  nothing.

---

## Origins

FinVet evolved from [FinVet v1](https://github.com/danielberhane/finvet-acl-demo), which
combined two RAG pipelines and an external fact-check path through vote-based verdicts. This
version is a ground-up redesign as a multi-agent LangGraph system; what carried over is the
design thesis — evidence-backed verdicts with source attribution and explicit uncertainty.

## License & third-party

[Apache-2.0](LICENSE). Relies on external components it does not distribute — notably the
**AGPL-3.0** SEC EDGAR MCP server, run as its own container (installed from PyPI). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Users must honor each provider's terms,
including the [SEC Fair Access policy](https://www.sec.gov/os/webmaster-faq#developers)
(`SEC_EDGAR_USER_AGENT` needs a real name and email).
