# FinVet

**Agentic financial claim verification.**

[![ci](https://github.com/danielberhane/finvet/actions/workflows/ci.yml/badge.svg)](https://github.com/danielberhane/finvet/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)

Give it a claim — *"Tesla's 2024 annual revenue was $150 billion"* — and it returns a verdict
(`SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`), a confidence score, the evidence chain, and an
audit trail of every step that produced the answer.

<p align="center">
  <img src="docs/diagrams/finvet-screenshot.png" alt="FinVet refuting a claim, with retrieved value, tolerance, and tool calls">
</p>

> **Not financial advice.** A research and demonstration system; outputs may be wrong and must
> be independently verified. Not affiliated with any data provider named here.

---

## Deterministic verdict override

LLMs are unreliable at numeric comparison, and in claim verification the task *is* comparing
a claimed number to a filed one. When both values exist, FinVet recomputes the comparison in
Python with source-appropriate tolerances and overrides the model when they disagree. Both
results are recorded:

```python
{"verdict": "REFUTES", "llm_original_verdict": "SUPPORTS", "override_applied": True}
```

See [`_apply_override`](src/finvet/agents/base.py). For numeric claims, the final comparison
uses only a trusted structured observation, so model-generated prose cannot override the
arithmetic.

---

## Evaluation

### Cross-model benchmark

A matched single-run comparison: the same 97 claims through two models on identical code and
a frozen dataset. The model behind each run is recorded from the serving process, not the
client — which is how the run caught `deepseek-chat` silently resolving to a different model:

| | DeepSeek | MiniMax |
|---|---|---|
| requested as | `deepseek-chat` (alias) | `MiniMax-M2.7` |
| actually served | `deepseek-v4-flash` | `MiniMax-M2.7` |
| endpoint | hosted API | self-hosted LiteLLM gateway |
| wall clock, 97 claims | **17.2 min** | 47.7 min |

| Measure | DeepSeek | MiniMax |
|---|---|---|
| Verdict accuracy (86 claims scored after excluding 8 live-market rows and 3 observe-only rows) | **96.5%** | **96.5%** |
| Confidently-wrong verdicts (of 94 scored incl. live market) | **0** | **0** |
| Observation-source attribution (42 decisive numeric verdicts) | **100%** | **100%** |
| Required-tool coverage | 98.4% | 94.8% |
| Calibration, decisive-verdict ECE | 0.039 | 0.046 |
| Evidence-path match | 97.9% | 96.8% |

Two kinds of result, kept apart. **Enforced by construction**: zero out-of-lane tool calls
(each agent binds only its own tools), zero tool spend on the 26 claims that must spend
nothing (they short-circuit before any agent), and observation-source attribution (a decisive
numeric verdict is unreachable without a trusted observation — the path fails closed).
**Observed in these runs**: identical accuracy, zero confidently-wrong verdicts, and perfect
accuracy in the top confidence bin. The first kind is a code property; the second held twice,
and for DeepSeek it has since been measured across four runs on the frozen set (c1 at
`94f1ec9`, c2–c4 at `0cdce16`): pass@1 98.9%, **pass^4 94.5%** — 86 of the 91 rows every run
scores passed on *all four* attempts. The four that flickered are ids 17 and 22 (XBRL rows
that a Llama Guard false positive sometimes escalates instead of answering), 51 (the
News→SEC corroboration row) and 62 (a live market quote); the population includes the
live-market rows. MiniMax has one complete run, so its column is pass@1 only. Single-run
differences of a few claims sit inside this measured variation. Pooled summary:
[`layers-deepseek-c1-c4.json`](docs/eval/layers-deepseek-c1-c4.json).

Raw accuracy including live-market rows reads 96.8% vs 93.5% (n=93: both columns drop the
one row that errored at the 600s timeout under MiniMax, so the two models are scored on the
same rows); the gap is a market-data outage during the MiniMax run, not the model.

**Evidence:** [`docs/eval/`](docs/eval/) holds five redacted per-claim run artifacts (claim
text withheld by [`scripts/redact_run.py`](scripts/redact_run.py); every table cell and the
pass^4 figure recompute from them), the layer summaries, the benchmark write-up, and the
[dataset card](docs/eval/DATASET_CARD.md) — composition, SHA-256,
labelling rules, disclosed biases, and three fully published sample rows. The dataset itself
is held out privately — a published test set enters training corpora and stops measuring
anything — and is available to reviewers against the published hash.

### Retrieval accuracy

XBRL retrieval, measured against SEC primary-source values: **198/199 (99.5%)** with zero
silently-wrong results — the one miss returns NOT_ENOUGH_INFO. A fallback from the
period-targeted `companyconcept` endpoint to `frames` took accuracy from ~91% to 99.5%.
This figure comes from `tests/integration/test_xbrl_retrieval.py` against a retrieval gold
set held privately with the golden claims, so unlike the table above it is not recomputable
from this repository (see [Known gaps](docs/VALIDATION_STRATEGY.md#known-gaps)).

Filing-text retrieval, on the 70-case calibration set that also set the relevance floor: all
30 on-topic queries retrieved, all 30 off-topic queries rejected, and 10 near-misses (right
topic, wrong period or form) returned nothing. Calibration fit, not held-out performance;
every case is recorded in
[`tests/accuracy/rag_release_a_manifest.json`](tests/accuracy/rag_release_a_manifest.json).

---

## Architecture

A 12-node LangGraph `StateGraph`. Three domain agents route by claim type, each a ReAct loop
with its own tools: routing, period resolution, consensus and guardrails are fixed pipeline
stages, and within the selected agent the model chooses its own tool calls. One agent runs
per claim, so consensus passes its verdict through and adjusts confidence rather than
reconciling several opinions.

<p align="center">
  <a href="docs/diagrams/finvet-linkedin.png"><img src="docs/diagrams/finvet-linkedin.png" alt="Architecture" width="800"></a>
  <br><sub>Click the diagram for full resolution.</sub>
</p>

| Agent | Handles | Sources | Can delegate to |
|---|---|---|---|
| **SEC** | GAAP financials, and claims about what a filing says | SEC EDGAR (XBRL) via MCP, hybrid RAG over filing text | — |
| **Market** | Prices, valuation, market cap | Finnhub | — |
| **News** | Events and announcements | Tavily search | SEC |

**Delegation** runs one way — News asks SEC whether the issuer's own filing discloses a
reported fine or settlement — and the SEC agent holds no delegation tool, so the call
terminates by construction. Where the filing states an amount, the verdict follows the filing
rather than the press; only the two sources *contradicting* each other sends the claim to a
person. Filing silence does not: deciding an issuer *should* have disclosed something is a
materiality judgment, and this release does not make one.

**Trust boundary.** Retrieved filing text is supporting evidence: it shows what a company
said, and a model's reading of it never becomes the number a verdict rests on — XBRL is the
authoritative numeric source. A claim about what a filing *says* is answered from retrieved
text with passages cited; absence of a passage is never treated as refutation. One narrow
exception: for fines and settlements (`fine_amount`, `settlement_amount` — no XBRL
concept exists), Python — not the model —
extracts the amount from Legal Proceedings text, and only when exactly one unambiguous
candidate is present.

**Guardrails and review.** Always-on regex/PII checks plus an optional Llama Guard layer on
input and output; the guards decide safety, and the parser — not the safety layer — decides
whether a claim is verifiable. Human review triggers on low confidence, unsafe output, or a
press-vs-filing conflict, pausing at a LangGraph checkpoint and resuming with the reviewer's
decision merged in. Every tool call and verdict is persisted to Postgres with a checksum the
API re-verifies on read — it detects a record altered without its checksum being recomputed;
it is not tamper-proof against a writer who can change both.

Mechanics — retrieval fusion, delegation states, review recovery:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/RAG_AND_AGENTIC_RAG_GUIDE.md`](docs/RAG_AND_AGENTIC_RAG_GUIDE.md).

---

## Quickstart

```bash
git clone https://github.com/danielberhane/finvet.git && cd finvet
cp .env.example .env            # add DEEPSEEK_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD
docker compose --profile sec up --build
docker exec finvet-ollama ollama pull nomic-embed-text   # embeddings, one-time
# UI → http://localhost:8501   API → http://localhost:8000
```

Prebuilt images, published on tagged releases: `docker pull ghcr.io/danielberhane/finvet-api:latest`
(and `finvet-ui`).
Every published port binds to `127.0.0.1`; the API is unauthenticated and Postgres ships a
dev password, so expose the stack deliberately (`FINVET_BIND_ADDR=0.0.0.0`) only after
changing `POSTGRES_PASSWORD`. Real API keys are required — the system verifies against live
data. Missing optional keys disable their feature rather than crashing.

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

</details>

---

## Limitations & operations

Packaged for **local / demo use, not public hosting as-is** — no auth, rate limiting, or
CORS. With SEC MCP or Ollama down, the API degrades to NOT_ENOUGH_INFO or review rather than
500-ing. Python dependencies are locked (`uv.lock`); [CI](.github/workflows/ci.yml) runs
lint, tests and Docker builds on every push, and [release](.github/workflows/release.yml)
publishes images to GHCR on version tags.

- **US large-cap equities, point-in-time claims only.** Latency measured on the benchmark
  runs: p50 13s / p95 23s per claim (DeepSeek), p50 28s / p95 43s (MiniMax, excluding the
  one row that errored at the 600s timeout during a market-data outage; p95 48s with it).
- **A numeric verdict requires a structured source** — an XBRL fact, a market quote field,
  or the deterministic fine/settlement extraction. The remaining 45 metrics the parser can
  accept (analyst price targets among them) are declined up front with a stated limitation.
- **Range claims are declined.** "Between X and Y" parses, but only the band's midpoint
  survives parsing, and comparing a midpoint refutes true claims — so these return
  NOT_ENOUGH_INFO and typically route to review.
- **Q4-derived numeric claims are unsupported** and return NOT_ENOUGH_INFO; the retrieval
  contract does not combine annual and nine-month facts.
- **Pending reviews do not survive an API restart** (in-memory checkpoints); an unresumable
  review is refused rather than answered from the reviewer's own submission.
- **Claim memory ships disabled** — its output is prior model output, not a source.
- The consensus step is **heuristic, not learned**; historical prices need a paid Finnhub
  tier.
- **Not a compliance product** — it applies model-risk-management *principles*; it certifies
  nothing.

---

## Contributing, security, origins & license

Bug reports and small focused fixes are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, what CI enforces, and the changes
that will be declined. To report a vulnerability, or to read what this system does and
does not defend against before deploying it, see [SECURITY.md](SECURITY.md).


Evolved from [FinVet v1](https://github.com/danielberhane/finvet-acl-demo) (two RAG pipelines
and vote-based verdicts); this version is a ground-up redesign as a multi-agent LangGraph
system. [Apache-2.0](LICENSE). Relies on external components it does not distribute — notably
the **AGPL-3.0** SEC EDGAR MCP server, run as its own container. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); users must honor each provider's terms,
including the [SEC Fair Access policy](https://www.sec.gov/os/webmaster-faq#developers).
