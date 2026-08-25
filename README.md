# FinVet

**Agentic financial claim verification.**

<!-- Restore after the first push, once CI has run once:
[![ci](https://github.com/danielberhane/finvet/actions/workflows/ci.yml/badge.svg)](https://github.com/danielberhane/finvet/actions/workflows/ci.yml)
-->
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

Give it a claim — *"Tesla's 2024 annual revenue was $150 billion"* — and it returns a verdict
(`SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`), a confidence score, the evidence chain, and an
audit trail of every step that produced the answer.

<p align="center">
  <img src="docs/diagrams/finvet-screenshot.png" alt="FinVet refuting a claim, with retrieved value, tolerance, and tool calls" width="800">
</p>

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

A 12-node LangGraph `StateGraph`. Three domain agents route by claim type, each a ReAct loop
with its own tools. They are not isolated: an agent can delegate to another when the answer
lies outside its sources: the News agent asks SEC whether the issuer's own filing confirms a
reported fine or settlement, checking press coverage against the primary source.
Evidence is reconciled, guarded, and paused for human review on any of three triggers: low
confidence, unsafe output, or **two sources disagreeing**.

<p align="center">
  <img src="docs/diagrams/finvet-linkedin.png" alt="Architecture" width="800">
</p>

| Agent | Handles | Sources | Can delegate to |
|---|---|---|---|
| **SEC** | GAAP financials — revenue, income, EPS, balance sheet, cash flow | SEC EDGAR (XBRL) via MCP, hybrid RAG over filing text | — |
| **Market** | Prices, valuation, market cap | Finnhub | — |
| **News** | Events, announcements, macro indicators | Tavily search, FRED | SEC |

Delegation runs one way only, so it terminates by construction. When both sides reach a
decisive but opposite verdict the claim escalates to a reviewer rather than shipping — filing
*silence* is not treated as disagreement, since a periodic report is a point-in-time
document.

Retrieval over filing text is hybrid: pgvector dense search and Postgres `tsvector` BM25 fused
by reciprocal rank fusion, embedded locally with `nomic-embed-text` (no API key in the
embedding path).

Deeper dives: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/RAG_AND_AGENTIC_RAG_GUIDE.md`](docs/RAG_AND_AGENTIC_RAG_GUIDE.md).

---

## Quickstart

The whole system — Postgres, Ollama, API, UI, and the SEC MCP server — comes up with one
command. The API creates its schema on first boot.

```bash
git clone https://github.com/danielberhane/finvet.git && cd finvet
cp .env.example .env            # add DEEPSEEK_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD
docker compose --profile sec up --build
docker exec finvet-ollama ollama pull nomic-embed-text   # embeddings, one-time
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

Requires a running Postgres and Ollama with `nomic-embed-text` pulled. **No API key** — the
embedding model is served locally, so ingest and search cost nothing per call.

### Profiles

| Profile | Adds | For |
|---|---|---|
| *(default)* | Postgres + pgvector, Ollama, API, UI | always |
| `--profile sec` | SEC EDGAR MCP (self-contained, from PyPI) | SEC claims (most demos) |

Ollama runs by default because it serves the embedding model that RAG and claim memory both
use. Pull `nomic-embed-text` once (~270 MB). The optional Llama Guard semantic guardrail uses
the same container but stays off unless you set `ENABLE_LLAMA_GUARD=true` and pull
`llama-guard3:8b` (~5 GB); the regex guard runs standalone either way.

### Keys

| Key | Powers | Without it |
|---|---|---|
| `DEEPSEEK_API_KEY` | claim parsing, agents, verdicts | **required** — nothing runs |
| `TAVILY_API_KEY` | news search | news claims → NOT_ENOUGH_INFO |
| `FINNHUB_API_KEY` | market quotes, tickers | market claims → NOT_ENOUGH_INFO |
| *(none)* | embeddings → RAG + claim memory | served locally by Ollama — no key, no per-call cost |
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
- **Agent-to-agent** — the news agent can ask SEC whether the issuer's own filing confirms a
  reported fine or settlement; a contradiction escalates to a human.
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
- **Graceful degradation** — with SEC MCP or Ollama down, the API stays up and returns
  NOT_ENOUGH_INFO or routes to review rather than 500-ing. Losing the embedder costs the
  semantic half of RAG; keyword search keeps working.
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
