# Third-Party Notices

FinVet's own code is licensed under [Apache-2.0](LICENSE). It depends on the components
below but does not vendor, copy, modify or redistribute any of them: each is installed or
run by the user under its own licence. This file is an engineering record, not legal
advice. Before redistributing FinVet, bundling it into a product, or offering it as a
hosted service, have the terms reviewed by someone qualified.

## SEC EDGAR MCP server (AGPL-3.0)

| | |
|---|---|
| Source | https://github.com/stefanoamorelli/sec-edgar-mcp |
| Version | `sec-edgar-mcp==1.1.0` from PyPI, installed by `docker/Dockerfile.sec` |
| Licence | GNU Affero General Public License v3.0 |
| Relationship | Runs as its own container. FinVet calls it over the network (MCP streamable-HTTP) through its own adapter, `src/finvet/mcp/sec_edgar.py`. No source from the server is in this repository. |

AGPL-3.0 §13 applies to anyone who modifies the server and offers it to users over a
network. FinVet does neither. If you fork the server or host it for third parties, that
analysis is yours to make.

## Ollama and Llama Guard 3 (optional)

Ollama serves the local embedding model and, when `ENABLE_LLAMA_GUARD=true`, Llama Guard 3.
Both run as a separate container. The model weights are pulled by you and are subject to
their own terms; Llama Guard 3 is governed by Meta's Llama licence.

## Python dependencies

Declared in `pyproject.toml` and pinned in `uv.lock`, installed from PyPI, not
redistributed. All are under permissive licences (MIT, BSD, Apache-2.0) except two
copyleft Postgres drivers, which FinVet imports as ordinary libraries and does not link
statically or copy: `psycopg2-binary` (LGPL with exceptions), always installed, and
`psycopg` (LGPL-3.0), installed only with the optional `memory` extra.

## External data services

FinVet queries these at runtime with credentials you supply. No responses are committed to
this repository. Each is governed by its own terms, which you accept by configuring it.

| Service | Used for | Credential |
|---|---|---|
| SEC EDGAR (`data.sec.gov`) | XBRL company facts and filings | none; `SEC_EDGAR_USER_AGENT` must identify you, per the [SEC Fair Access policy](https://www.sec.gov/os/webmaster-faq#developers) |
| Finnhub | market quotes and company data | `FINNHUB_API_KEY` |
| Tavily | news search | `TAVILY_API_KEY` |
| LLM provider (DeepSeek by default; any OpenAI-compatible endpoint) | parsing, agent reasoning, verdicts | `DEEPSEEK_API_KEY`, or the key your endpoint needs |

SEC filings are US government works. Those you download into `data/filings/` stay on your
machine and are not committed. FinVet does not scrape any provider and identifies itself
on every request.
