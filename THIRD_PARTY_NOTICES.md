# Third-Party Notices

FinVet's own source code is licensed under [Apache-2.0](LICENSE). This file records the
externally owned components FinVet depends on. **FinVet does not vendor, copy, modify, or
redistribute any of them** — they are installed or run by the user.

This is an engineering record, not legal advice. If you intend to redistribute FinVet, bundle
it into a product, or offer it as a hosted network service, have the terms below reviewed by
someone qualified.

---

## External servers (run as separate processes)

### SEC EDGAR MCP server — **AGPL-3.0**

| | |
|---|---|
| Source | https://github.com/stefanoamorelli/sec-edgar-mcp |
| Version run | `sec-edgar-mcp==1.1.0` from PyPI, installed by `docker/Dockerfile.sec` |
| License | GNU Affero General Public License v3.0 (declared in its `pyproject.toml`) |
| Transport | MCP streamable-HTTP (JSON-RPC 2.0), default `http://localhost:9870` |
| Tools consumed | `get_company_info`, `get_recent_filings`, `get_xbrl_concepts` |
| Relationship | FinVet **calls** it over the network. No source is copied into this repository. |

FinVet's adapter is `src/finvet/mcp/sec_edgar.py`; the generic transport is
`src/finvet/mcp/mcp_client.py`. Both are original FinVet code.

You obtain and run this server yourself (`docker compose --profile sec up -d`). It is not
included in this repository and is not distributed with it.

**Worth understanding before you deploy:** AGPL-3.0 §13 concerns making a *modified* version
of the covered program available to users over a network. FinVet neither modifies nor hosts
this server. If you fork the MCP server, or expose it to third parties as part of a hosted
service, that analysis changes and is yours to make.

### Ollama + Llama Guard 3 (optional)

Only used when `ENABLE_LLAMA_GUARD=true`. Run as a separate container; model weights are
pulled by you and are subject to Meta's Llama license terms.

---

## Python dependencies

Installed from PyPI, not redistributed here. The direct dependencies declared in
`pyproject.toml`, at the versions pinned in `uv.lock`; transitive packages are listed there.
Licenses as declared in package metadata:

| Dependency | Version | Purpose | License |
|---|---|---|---|
| langchain / langchain-core | 1.4.0 / 1.6.2 | agent framework | MIT |
| langgraph | 1.2.11 | pipeline graph, HITL checkpointing | MIT |
| langchain-openai | 1.6.0 | OpenAI-compatible chat client, used for every provider | MIT |
| tiktoken | 0.14.0 | token counting for filing chunks | MIT |
| fastapi | 0.141.1 | API layer | MIT |
| uvicorn | 0.52.4 | ASGI server | BSD-3-Clause |
| pydantic / pydantic-settings | 2.13.5 / 2.15.0 | models, config | MIT |
| sqlalchemy | 2.0.52 | ORM | MIT |
| **psycopg2-binary** | **2.9.12** | Postgres driver | **LGPL with exceptions** |
| pgvector | 0.5.0 | vector column type | MIT |
| tavily-python | 0.8.1 | news search SDK | MIT |
| httpx | 0.28.1 | HTTP client | BSD-3-Clause |
| beautifulsoup4 | 4.15.0 | filing HTML parsing | MIT |
| streamlit | 1.63.0 | UI | Apache-2.0 |
| pandas | 3.0.5 | UI tables | BSD-3-Clause |
| requests | 2.34.2 | HTTP (UI) | Apache-2.0 |
| python-dotenv | 1.2.3 | env loading | BSD-3-Clause |

Optional extras, not installed by default:

| Dependency | Version | Extra | License |
|---|---|---|---|
| langgraph-checkpoint-postgres | 3.1.2 | `memory` (claim memory store) | MIT |
| **psycopg** | **3.3.2** | `memory` | **LGPL-3.0-only** |
| deepeval | 4.2.2 | `evaluation` (tool-trajectory scoring) | Apache-2.0 |

The two **psycopg** packages are the only copyleft Python dependencies, and `psycopg` is
present only with the `memory` extra. FinVet imports them as ordinary libraries; it does not
link them statically or copy their source.

---

## External data services

FinVet queries these at runtime using **credentials you supply**. No responses are committed
to this repository. Each is governed by its own terms, which you accept by configuring it:

| Service | Used for | Credential |
|---|---|---|
| SEC EDGAR (`data.sec.gov`) | XBRL company facts | none; requires `SEC_EDGAR_USER_AGENT` |
| Finnhub | quotes, company overview, earnings | `FINNHUB_API_KEY` |
| Tavily | news search | `TAVILY_API_KEY` |
| LLM provider (DeepSeek by default; any OpenAI-compatible endpoint via `LLM_<ROLE>__BASE_URL`) | claim parsing, agent reasoning, verdicts | `DEEPSEEK_API_KEY`, or the key the chosen endpoint needs |

SEC filings retrieved into `data/filings/` are US government works and are not committed to
this repository.

FinVet does not scrape any provider and does not disguise its client identity.
