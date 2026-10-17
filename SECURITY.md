# Security Policy

## Supported versions

The current release line (`v2.0.x`) and `main`.

## Reporting a vulnerability

Use the private [security advisory form](https://github.com/danielberhane/finvet/security/advisories/new),
not a public issue. Describe what you did, what happened and what you expected. You will
hear back within seven days. There is no bug bounty.

## Before you deploy

FinVet is built for local and demonstration use and is not hardened for public hosting.

- The API has no authentication or rate limiting. Anyone who can reach port 8000 can spend
  your LLM and data-provider credits.
- Every port binds to `127.0.0.1`. Setting `FINVET_BIND_ADDR` exposes the stack; change the
  shipped `POSTGRES_PASSWORD` first.
- Pending human reviews are held in memory and lost on restart.
- The audit trail is a checksummed record of what happened, not tamper-proof.

## Scope

Third-party services FinVet calls (SEC EDGAR and its MCP server, Finnhub, Tavily, Ollama,
your LLM provider) are outside this policy.
