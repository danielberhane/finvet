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
| Tested against | commit `e137595fcb68f8661d3b18f3de21eb7083888f39` (2026-01-25) |
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

Installed from PyPI, not redistributed here. Licenses as declared in package metadata:

| Dependency | Version | Purpose | License |
|---|---|---|---|
| langchain | 1.2.8 | agent framework | MIT |
| langgraph | 1.0.7 | pipeline graph | MIT |
| langchain-deepseek | 1.0.1 | DeepSeek LLM binding | MIT |
| langgraph-checkpoint-postgres | 3.0.4 | HITL checkpointing | MIT |
| fastapi | 0.128.0 | API layer | MIT |
| uvicorn | 0.40.0 | ASGI server | BSD-3-Clause |
| pydantic / pydantic-settings | 2.12.5 / 2.12.0 | models, config | MIT |
| sqlalchemy | 2.0.46 | ORM | MIT |
| **psycopg** | **3.3.2** | Postgres driver | **LGPL-3.0-only** |
| **psycopg2-binary** | **2.9.11** | Postgres driver | **LGPL with exceptions** |
| pgvector | 0.4.2 | vector column type | MIT |
| scikit-learn | 1.8.0 | similarity utilities | BSD-3-Clause |
| numpy | 2.4.2 | numerics | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pandas | 3.0.5 | UI tables | BSD-3-Clause |
| tavily-python | 0.7.21 | news search SDK | MIT |
| httpx / httpx-sse | 0.28.1 / 0.4.3 | HTTP + SSE | BSD-3-Clause / MIT |
| beautifulsoup4 | 4.14.3 | filing HTML parsing | MIT |
| streamlit | 1.62.0 | UI | Apache-2.0 |
| requests | 2.32.5 | HTTP (UI) | Apache-2.0 |
| python-dotenv | 1.2.1 | env loading | BSD-3-Clause |
| pyyaml | 6.0.3 | config parsing | MIT |

The two **psycopg** packages are the only copyleft Python dependencies. FinVet imports them as
ordinary libraries; it does not link them statically or copy their source.

---

## External data services

FinVet queries these at runtime using **credentials you supply**. No responses are committed
to this repository. Each is governed by its own terms, which you accept by configuring it:

| Service | Used for | Credential |
|---|---|---|
| SEC EDGAR (`data.sec.gov`) | XBRL company facts | none; requires `SEC_EDGAR_USER_AGENT` |
| Finnhub | quotes, company overview, earnings | `FINNHUB_API_KEY` |
| Tavily | news search | `TAVILY_API_KEY` |
| DeepSeek | claim parsing, agent reasoning, verdicts | `DEEPSEEK_API_KEY` |

SEC filings retrieved into `data/filings/` are US government works and are not committed to
this repository.

FinVet does not scrape any provider and does not disguise its client identity.
