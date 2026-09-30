# FinVet

**Agentic financial claim verification.**

[![ci](https://github.com/danielberhane/finvet/actions/workflows/ci.yml/badge.svg)](https://github.com/danielberhane/finvet/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)

FinVet verifies one financial claim at a time. Given *"Microsoft's fiscal 2025 revenue was
$282 billion,"* it returns a verdict of supported, refuted, or not enough information, a
confidence score, and the number it compared the claim against. Each response also names the
agent that answered, lists every tool call it made in order, and keeps an audit trail of the
run.

<p align="center">
  <a href="docs/diagrams/finvet-linkedin.png"><img src="docs/diagrams/finvet-linkedin.png" alt="Architecture" width="1000"></a>
  <br><sub>Click the diagram for full resolution.</sub>
</p>

> **Not financial advice.** A research and demonstration system; outputs may be wrong and must
> be independently verified. Not affiliated with any data provider named here.

---

## Deterministic verdict override

Models are unreliable at comparing numbers, and that comparison decides the verdict. FinVet
recomputes it in Python, with tolerances that depend on the source. Python's result stands
when the two disagree, and both verdicts are kept in the response.

The comparison uses only a structured value returned by a tool. A claim with no such value
returns not enough information.

---

## Evaluation

**Claim set.** 349 financial claims across 76 US-listed companies, from mega-caps to mid-caps,
including banks and insurers, non-calendar fiscal years and restated periods. Claims cover
annual and quarterly GAAP figures, tolerance boundaries, fines and settlements, live share
prices and filing text, plus cases that must decline, escalate or be blocked. Each row carries
its expected verdict, evidence path and a pointer to the filing, concept and period. Full
composition in the [golden_u card](docs/eval/GOLDEN_U_CARD.md).

**Protocol.** Two models, identical code, four runs each on 2026-09-29, scored on seven layers:
outcome, tool trajectory, grounding, calibration, asymmetric risk, reliability, reachability.
Populations and method are in the [benchmark write-up](docs/eval/BENCHMARK_2026-09-29.md).

### Cross-model benchmark

| Layer | DeepSeek-V4.1-Flash | Qwen3.8 |
|---|---|---|
| 1. Outcome — verdict accuracy | **97.0%** | **96.7%** |
| 2. Tool trajectory — tools called | 99.2% | 97.7% |
| 3. Grounding — numbers traced | **100%** | **100%** |
| 4. Calibration — decisive ECE | 0.049 | 0.050 |
| 5. Asymmetric risk, confidently-wrong | 1 | **0** |
| 5. Asymmetric risk, declined | 2.7% | 3.6% |
| 6. Reliability — pass^4 | **94.8%** | **91.5%** |
| 7. Reachability — expected path | 94.8% | 97.9% |

First run of each model; pass^4 uses all four. Outcome excludes the 28 live-price claims.

**Stability.** Four runs per model. DeepSeek-V4.1-Flash: pass@1 97.0%, **pass^4 94.8%**;
17 misses, one a wrong verdict (a net loss parsed as a gain). Qwen3.8: pass@1 96.4%,
**pass^4 91.5%**; 28 misses, none a wrong verdict.

**Models.** DeepSeek-V4.1-Flash via DeepSeek's API (`deepseek-flash`); Qwen3.8 on vLLM behind a
LiteLLM gateway, response cache off.

**Retrieval.** XBRL lookups against SEC values: **198/199**, on a gold set held privately and not
recomputable here.

**Evidence.** [`docs/eval/`](docs/eval/): redacted artifacts of all eight runs, layer summaries
and dataset cards. Every table figure recomputes from them in CI.

---

## Architecture

One agent runs per claim, chosen by claim type.

| Agent | Handles | Sources |
|---|---|---|
| **SEC** | GAAP financials, filing text | SEC EDGAR XBRL via MCP, hybrid RAG over filings |
| **Market** | Prices, valuation, market cap | Finnhub |
| **News** | Events, fines, settlements | Tavily search, and one-way delegation to SEC (cannot recurse) |

<p align="center">
  <a href="docs/diagrams/finvet-delegation-run.png"><img src="docs/diagrams/finvet-delegation-run.png" alt="FinVet verifying a news claim: the pipeline steps, the delegation to the SEC agent, and the verdict" width="860"></a>
  <br><sub>A news claim settled by delegation: the SEC agent found the same &euro;500 million in Apple's filing.</sub>
</p>

**Review and audit.** Regex and PII guards always run, with optional Llama Guard. Low
confidence, unsafe output or a press-versus-filing conflict pauses the graph at a checkpoint
until a reviewer decides. Every tool call and verdict is persisted with a checksum the API
re-verifies on read ([SECURITY.md](SECURITY.md)).

Mechanics — retrieval fusion, delegation states, review recovery:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

```bash
git clone https://github.com/danielberhane/finvet.git && cd finvet
cp .env.example .env            # add DEEPSEEK_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD
docker compose --profile sec up --build
docker exec finvet-ollama ollama pull nomic-embed-text   # embeddings, one-time
# UI → http://localhost:8501   API → http://localhost:8000
```

Prebuilt images are published on tagged releases: `docker pull ghcr.io/danielberhane/finvet-api:latest`
(and `finvet-ui`). Every port binds to `127.0.0.1`; read [SECURITY.md](SECURITY.md) before
exposing the stack. Real API keys are required, since the system verifies against live data.
Missing optional keys disable their feature rather than crashing.

<details>
<summary><b>Native dev · RAG ingest · compose profiles · which key powers what</b></summary>

### Native (hot reload)

```bash
uv sync
cp .env.example .env
docker compose up -d postgres                        # pgvector
uvicorn finvet.main:app --port 8000 --app-dir src    # API → :8000
streamlit run ui/app.py --server.port 8501           # UI  → :8501
```

### Populating the RAG index

The vector store starts empty — filings are not distributed with the repo. XBRL verification
works without it; RAG over filing text needs an ingest pass:

```bash
# Place filings as data/filings/<TICKER>/<TICKER>_<FORM>_<PERIOD-END>.html
python -m finvet.rag.ingest                    # defaults to data/filings
```

Requires Postgres and Ollama with `nomic-embed-text` pulled (~270 MB, no API key — embedding
is local). The optional Llama Guard layer needs `ENABLE_LLAMA_GUARD=true` and
`llama-guard3:8b` (~5 GB).

### Profiles

| Profile | Adds | For |
|---|---|---|
| *(default)* | Postgres + pgvector, Ollama, API, UI | always |
| `--profile sec` | SEC EDGAR MCP (self-contained, from PyPI) | SEC claims (most demos) |

### Keys

| Key | Powers | Without it |
|---|---|---|
| `DEEPSEEK_API_KEY` | claim parsing, agents, verdicts | the default provider fails to start; point the roles elsewhere and it is not needed |
| `TAVILY_API_KEY` | news search | **required** — nothing runs |
| `FINNHUB_API_KEY` | market quotes, tickers | market claims → NOT_ENOUGH_INFO |
| *(none)* | embeddings, SEC XBRL | local Ollama / free public endpoints |

The LLM is pluggable ([`llm/factory.py`](src/finvet/llm/factory.py)): point any
OpenAI-compatible chat-completions endpoint — Ollama, vLLM, LiteLLM, or a hosted vendor — at
any of the three roles via env vars, no code change.

### Tracing (LangSmith)

Off by default. Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`
in `.env` and restart the API. Every verification then appears as one trace carrying its
`request_id`, so it joins to its audit row at `GET /audit/{request_id}`. Trace names, what a
trace contains, and sample timings:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#7-configuration-reference).

**Tracing is data egress.** The claim text, tool outputs (filing excerpts, quotes, news
snippets), and model prompts leave the machine for LangSmith. Do not enable it on claims you
would not send to a third party.

</details>

---

## Limitations

Packaged for local and demo use. With SEC MCP or Ollama down, the API degrades to
NOT_ENOUGH_INFO or review rather than failing.

- **Scope**: US large-cap equities, one claim at a time — a single English sentence stating one
  fact. Compound claims are not decomposed; fourth-quarter figures and unservable metrics are
  declined with a stated reason; value ranges fail closed to human review; relative changes
  ("doubled since 2020") parse to no value and are declined. Latency on the default provider,
  p50 11s and p95 76s per claim; the slow tail is the fines and restated-year claims, which
  read filing prose.
- **Numeric verdicts need a structured source**: an XBRL fact, a market quote field, or the
  deterministic fine/settlement extraction. The other 45 metrics the parser accepts, analyst
  price targets among them, are declined up front with a stated reason.
- **Market data is current-price only.** Finnhub's free tier provides a quote delayed 15–20
  minutes and no price history, so only current-price claims with no stated period reach a
  verdict; historical price claims, and figures like market cap and P/E with no observation
  time, fail closed.
- **A stated loss can be read as a gain.** The parser turns "a net loss of $3.63 billion" into a
  positive value in some runs, and the comparison then refutes a true claim against the filed
  negative figure. This is the one wrong verdict the benchmark produced (row 1073, in two of
  four DeepSeek runs) and the parser's sign handling is the next fix.
- **Model reasoning is not stored.** The verdict's basis — the number, its concept, period and
  filing, and what it replaced — is recorded in the artifact and the audit trail; the prose the
  model wrote about it is returned in the response and not persisted.
- **Not a compliance product.** It applies model-risk-management principles; it certifies
  nothing.
- **Not a service.** The API has no authentication, authorization, rate limiting or tenant
  isolation ([SECURITY.md](SECURITY.md)). Review checkpoints live in process memory, so a
  pending review does not survive an API restart. It is single-process and has not been
  load-, failover- or sustained-degradation-tested; the latency figures above are from
  benchmark runs, one claim at a time.

Next: decomposition of multi-assertion claims, a paid market-data tier with historical price
alignment, and Q4 derivation.

---

## Contributing

Bug reports and focused fixes are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup,
what CI enforces, and the changes that will be declined. For vulnerability reports, and for
what this system does and does not defend against, see [SECURITY.md](SECURITY.md).

## Citation

To cite this software, use [CITATION.cff](CITATION.cff).

**Prior work.** FinVet v1, joint work with Duoduo Liao, verified claims with two retrieval
pipelines and an external fact-check source, deciding verdicts by confidence-weighted vote
([IEEE BigData 2025](https://ieeexplore.ieee.org/document/11400848);
[code](https://github.com/danielberhane/finvet-v1)). This release replaces that design: one
agent per claim, and the verdict settled by a deterministic comparator rather than a vote.

## License

[Apache-2.0](LICENSE). FinVet runs third-party components without distributing them, notably
the AGPL-3.0 SEC EDGAR MCP server, which runs as its own container; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Users must honor each provider's terms,
including the [SEC Fair Access policy](https://www.sec.gov/os/webmaster-faq#developers).
