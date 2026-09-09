# Security

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/danielberhane/finvet/security/advisories/new) rather
than opening a public issue. Please include what you did, what happened, and
what you expected. This is a research project maintained by one person, so
expect an acknowledgement within a week rather than within hours.

## Before you deploy this

FinVet is packaged for local and demonstration use. It is **not hardened for
public hosting**, and the following are known and deliberate:

- **The API has no authentication, authorization, or rate limiting.** Anyone
  who can reach port 8000 can spend your LLM and data-provider credits.
- **Docker Compose ships a development database password** and binds every
  published port to `127.0.0.1`. Overriding `FINVET_BIND_ADDR` exposes the
  stack; change `POSTGRES_PASSWORD` first.
- **Human-review checkpoints live in process memory.** A restart loses pending
  reviews, which the API reports as a `409 checkpoint_unavailable` rather than
  answering from the reviewer's own submission.
- **The audit trail records what the pipeline did; it is not tamper-evident.**
  Its checksum detects accidental corruption, not a motivated editor with
  database access.

## What the design does defend against

These are properties of the code, and regressions in them are security bugs
worth reporting:

- **Prompt injection through claim text.** Input passes a regex guard and, when
  enabled, Llama Guard. A free-form context field was removed from the request
  model because it was an injection channel; the API accepts an identifier the
  server resolves itself, and rejects unknown fields outright.
- **Model-authored numbers reaching a verdict.** A numeric verdict is computed
  in Python from a structured observation carrying its own provenance. A claim
  with no such observation returns `NOT_ENOUGH_INFO` rather than the model's
  reading of prose.
- **Agent scope.** Each agent binds only its own tools, and the SEC agent holds
  no delegation tool, so agent-to-agent recursion cannot occur.
- **Credential crossing.** Each LLM role names the environment variable holding
  its own key. Pointing a role at a new provider without setting that
  provider's key raises rather than falling back to another provider's
  credential.

## Scope

Third-party services FinVet calls (SEC EDGAR, the MCP server, Finnhub, Tavily,
Ollama, and whichever LLM provider you configure) are outside this policy.
Report issues in those to their maintainers.
