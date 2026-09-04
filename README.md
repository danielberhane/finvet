# FinVet

**Agentic financial claim verification.**

<!-- Restore after the first push, once CI has run once:
[![ci](https://github.com/danielberhane/finvet/actions/workflows/ci.yml/badge.svg)](https://github.com/danielberhane/finvet/actions/workflows/ci.yml)
-->
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)

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
lies outside its sources: the News agent asks SEC whether the issuer's own filing discloses a
reported fine or settlement, checking press coverage against the primary source.
One agent runs per claim, so consensus passes its verdict through and adjusts confidence
rather than reconciling several opinions. The result is guarded, and paused for human review on
any of three triggers: low confidence, unsafe output, or **the press reading disagreeing with
the filed figure**. Filing silence is not a
fourth: deciding an issuer *should* have disclosed something is a materiality judgment, and
this release does not make one.

<p align="center">
  <img src="docs/diagrams/finvet-linkedin.png" alt="Architecture" width="800">
</p>

| Agent | Handles | Sources | Can delegate to |
|---|---|---|---|
| **SEC** | GAAP financials, and claims about what a filing says | SEC EDGAR (XBRL) via MCP, hybrid RAG over filing text | — |
| **Market** | Prices, valuation, market cap | Finnhub | — |
| **News** | Events and announcements | Tavily search | SEC |

Delegation runs one way only, so it terminates by construction. Exactly one outcome escalates:
the two sides reaching opposite answers — what the news agent read in the press against the
figure the SEC agent took from the filing. Those are not equally strong, and the asymmetry is
the point: the filing figure decides the verdict, and the press reading only decides whether a
person is asked to look. Silence does not escalate — a periodic report omits
most things — and neither does silence from a filing that closed before the event. A filing
that was never successfully searched is recorded as such rather than counted as silence: a
claim about what a document says requires having read one.

Retrieved filing text is **supporting evidence**: it shows what a company said, and a model's
reading of it can never become the number a verdict rests on. Every passage is tagged
`evidence_role="supporting"`, and the trust boundary rejects a retrieval result as a numeric
observation regardless of shape. XBRL remains the authoritative numeric source for GAAP figures.

One narrow exception, because a penalty has no XBRL concept: for `fine_amount` and
`settlement_amount`, **Python** — not the model — extracts the amount from Legal Proceedings
text, and only when exactly one unambiguous candidate sits beside penalty language. What
changes is who reads the filing, not whether prose is trusted.

Retrieval over filing text is hybrid (full-text relevance + local embeddings, rank-fused),
scoped to the resolved period, with a relevance floor calibrated against a labelled set —
below it the tool returns nothing rather than the nearest available passage. Mechanics in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**What retrieval may and may not decide.** A claim about what a filing *says* —
"Apple's annual report discusses risks from supplier concentration" — is
answered from retrieved filing text and returns a verdict with the passages
cited. A claim naming a **number** is not: it falls back to XBRL or declines.
Absence of a passage is never treated as refutation.

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

Every published port binds to `127.0.0.1`. The API has no authentication and Postgres ships a
development default password, so neither belongs on a shared network by accident. Containers
still reach each other normally — that traffic goes over the compose network, not a published
port. To expose the stack deliberately, set `FINVET_BIND_ADDR=0.0.0.0`, and change
`POSTGRES_PASSWORD` first.

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
| `DEEPSEEK_API_KEY` | claim parsing, agents, verdicts | the default provider fails to start; point the roles elsewhere and it is not needed |
| `TAVILY_API_KEY` | news search | **required** — nothing runs |
| `FINNHUB_API_KEY` | market quotes, tickers | market claims → NOT_ENOUGH_INFO |
| *(none)* | embeddings → RAG + claim memory | served locally by Ollama — no key, no per-call cost |
| *(none)* | SEC XBRL | free public endpoints |

The LLM is pluggable ([`llm/factory.py`](src/finvet/llm/factory.py)) — point any
endpoint speaking the OpenAI-compatible chat-completions protocol — Ollama, vLLM, LiteLLM, or
a hosted vendor — at it via env var, no code change. That protocol name is the wire format,
not a dependency on any provider.

</details>

---

## Features

- **Model-directed ReAct agents in a deterministic workflow** — routing, period resolution,
  consensus and guardrails are fixed pipeline stages; within the selected agent the model
  chooses its own tools and iterations.
- **Measured retrieval** — 30 positive and 30 negative queries over a 1,398-chunk corpus:
  zero irrelevant results accepted at full recall, and ten near-miss queries (right topic,
  wrong period or form) return nothing. Every case recorded in
  [`tests/accuracy/rag_release_a_manifest.json`](tests/accuracy/rag_release_a_manifest.json).
- **Layered guardrails** — always-on regex/PII checks, plus an optional Llama Guard semantic
  layer, on both input and output. Safety is the guards' job; verifiability is the parser's.
- **Human-in-the-loop** — LangGraph `interrupt_before` pauses low-confidence or flagged
  verdicts for a reviewer, then resumes from the checkpoint with the decision merged in. A
  review is claimed atomically, so a second reviewer gets a conflict rather than overwriting
  the first. Checkpoints are held in memory: **pending reviews do not survive an API
  restart**, and an unresumable claim is refused rather than answered.
- **Audit trail with an integrity checksum** — every tool call, verdict decision, and data
  source persisted to Postgres, with a SHA-256 checksum over the stored execution envelope
  that the API re-verifies on read. It detects a record altered without its checksum being
  recomputed; it is not tamper-proof against a writer who can change both.
- **Bounded agent delegation** — the News agent can ask SEC whether the issuer's own filing
  discloses a reported fine or settlement; one hop, one direction, in process, and the SEC
  agent holds no delegation tool, so the call terminates by construction. Where the filing
  states an amount the verdict follows the filing rather than the press. Only a conflict between the two sends the claim to
  a human; filing silence does not, and silence is only reported when an applicable filing was
  actually searched *and came back empty*. When passages were read but no amount could be
  extracted from them, that is recorded as `FOUND_UNCERTIFIED` rather than as silence.
- **Claim memory** *(experimental, off by default)* — embedding search over past
  verifications. Its output is prior model output, not a source, so it ships disabled;
  `ENABLE_CLAIM_MEMORY=true` turns it on.

---

## Evaluation

### Cross-model benchmark

The same 97 claims through two different models, on identical code and dataset — so any
difference is the model, not the scaffolding. Each of the three LLM roles (parser, agent,
verdict) is pointed at one provider; the model recorded in each run artifact is read from the
serving process, not from the client.

| | DeepSeek V4-Flash | MiniMax-M2.7 |
|---|---|---|
| requested as | `deepseek-chat` (alias) | `MiniMax-M2.7` |
| actually served | `deepseek-v4-flash` | `MiniMax-M2.7` |
| parameters | 284B total, ~13B active/token | 230B total, ~10B active/token |
| architecture | sparse MoE, top-6 of 256 routed experts + 1 shared | sparse MoE, 8 of 256 experts, 62 layers |
| context | 1M tokens | 200K tokens |
| endpoint | hosted API | self-hosted LiteLLM gateway |
| wall clock, 97 claims | **17.2 min** | 47.7 min |

| Layer | DeepSeek | MiniMax |
|---|---|---|
| Verdict accuracy (excl. live market, n=86) | **96.5%** | **96.5%** |
| Confidently-wrong verdicts | **0** / 94 | **0** / 94 |
| Decisive numbers traceable to a tool call | **100%** (42/42) | **100%** (42/42) |
| Tool-path correctness (DeepEval) | 98.4% | 94.8% |
| Calibration, decisive ECE | 0.039 | 0.046 |
| Evidence-path match | 97.9% | 96.8% |

Five properties held identically under both: zero confidently-wrong verdicts, full grounding,
zero out-of-lane tool calls, correct zero-tool discipline on all 26 claims that should spend
nothing, and perfect accuracy in the top confidence bin. Those are enforced by the
deterministic layer rather than the model, and their invariance across two very different
models is the evidence for that claim.

One caveat stated plainly: this is a single run per model. The raw accuracy including live
market data reads 96.8% vs 93.5%, but that gap is a market-data outage during the MiniMax run,
not model quality. Separating a real difference from run-to-run noise would take roughly three
runs each — the measured noise floor over four earlier runs was pass^4 = 0.975.

The evidence is in [`docs/eval/`](docs/eval/): redacted run artifacts (claim text withheld —
the 97-claim dataset is held out privately, and the
[dataset card](docs/eval/DATASET_CARD.md) records its composition, SHA-256, labelling rules,
and three fully published sample rows), the computed layer summaries, and the full write-up.
Every cell in the tables above recomputes from those files without an API call; the dataset
itself is available privately to reviewers against the published hash.

---

### Retrieval and rejection accuracy

Measured against SEC primary-source values, not self-reported. *Silently wrong* — a confident
number that disagrees with the filing — is treated as the failure that matters.

| Metric | Result |
|---|---|
| XBRL retrieval accuracy | **198/199 (99.5%)**, zero silently-wrong (the one miss returns NOT_ENOUGH_INFO) |
| Reject classification (refusing unverifiable claims) | **64.2% recall at 100% precision** — never rejects a verifiable claim *(measured under the pre-September parser prompt; re-measure before quoting)* |

Retrieval falls back from SEC's period-targeted `companyconcept` read to the `frames`
endpoint when the first is empty for a company/concept — that fallback took accuracy from
~91% to 99.5%.

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
  (one ReAct agent making real tool calls, plus an optional delegated run).
- **A numeric verdict requires a structured source.** Values are compared only when they come
  from an XBRL fact, a market quote field, or — for fines and settlements only — a
  deterministic extraction from the filing's Legal Proceedings text (Python, not the model;
  the amount must sit beside penalty language and be the only candidate, or it declines). A
  number the model read out of prose is never evidence, so the remaining 45 metrics the
  parser can accept — analyst price targets among them — are declined up front with a stated
  limitation rather than answered from an LLM's reading.
- **Range claims are declined.** "Revenue was between X and Y" parses — the operator is
  legal — but is not compared: the parser carries only the band's midpoint, and comparing a
  midpoint refutes true claims whose band exceeds the tolerance. Such claims return
  NOT_ENOUGH_INFO and typically route to review.
- **Q4 numeric derivation is unsupported.** Q4 is not filed separately, and deriving it needs
  a 12-month fact minus a nine-month one; retrieval is scoped to a single resolved period per
  request, so that pair cannot be requested. Such claims are declined explicitly rather than
  answered approximately.
- **Pending reviews do not survive an API restart.** Checkpoints are `MemorySaver`-backed; an
  unresumable review is refused rather than answered from the reviewer's own submission. A
  review whose graph ran but whose audit write failed is marked
  `REVIEW_FINALIZATION_FAILED` and can be retried via `POST /review/{id}/reconcile` — but that
  retry reads the in-memory checkpoint, so it is **same-process recovery only**, not
  cross-process or restart-durable.
- The consensus step is **heuristic**, not learned.
- **Historical prices need a paid Finnhub tier**; on a free key the market agent reports
  NOT_ENOUGH_INFO rather than scraping around it.
- **Not a compliance product** — it applies model-risk-management *principles*; it certifies
  nothing.
- **Python 3.11, 3.12 and 3.13** are tested in CI and are the supported range.

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
