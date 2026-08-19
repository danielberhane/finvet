# FinVet Validation Strategy

## Overview

This document defines the testing and validation approach for FinVet, targeting enterprise MRM (Model Risk Management) readiness aligned with SR 11-7, E-23, and SS1/23 regulatory guidance.

## Test Taxonomy

### 1. Unit Tests (`tests/unit/`)

Pure logic tests with no external dependencies. All LLM calls and APIs are mocked.

| Test File | Coverage |
|-----------|----------|
| `test_helpers.py` | Confidence labels, preliminary analysis builder |
| `test_base_agent.py` | Tolerance thresholds, verdict override, context assembly |
| `test_domain_agents.py` | Agent wrappers, SEC provenance extraction, error fallback |
| `test_workflow.py` | Routing logic, consensus adjustments, HITL routing |
| `test_response_generator.py` | Summary text, metadata assembly, source formatting |
| `test_guards.py` | Input guardrails (injection, PII, length) |
| `test_settings.py` | LLM config defaults |
| `test_llm_factory.py` | LLM creation |

Run: `.venv/bin/python -m pytest tests/unit/ -v`

### 2. Integration Tests (`tests/integration/`)

End-to-end tests that call the API with real or mock LLMs.

| Test | Purpose |
|------|---------|
| `test_api.py` | API endpoint contracts, error responses |
| `test_mcp_apple.py` | SEC MCP server connectivity |

Run: `.venv/bin/python -m pytest tests/integration/ -v`

### 3. Golden Tests (`tests/golden/`)

Regression tests against known-correct claim verifications. Each golden test specifies:
- Input claim text
- Expected verdict (SUPPORTS / REFUTES / NOT_ENOUGH_INFO)
- Expected data source type (XBRL / RAG / A2A)
- Tolerance for confidence range

### 4. LLM-as-Judge Evaluation

Use an LLM to evaluate the quality of verification reasoning:
- **Faithfulness**: Does the reasoning accurately reflect the tool results?
- **Completeness**: Were all relevant data points considered?
- **Correctness**: Is the final verdict consistent with the evidence?

Frameworks: RAGAS, DeepEval, or custom judge prompts.

## Evaluation Metrics

### Accuracy Metrics
- **Verdict Accuracy**: % of claims where FinVet verdict matches ground truth
- **False Positive Rate**: Claims incorrectly marked SUPPORTS
- **False Negative Rate**: Claims incorrectly marked REFUTES
- **NOT_ENOUGH_INFO Rate**: % of claims where system couldn't find evidence

### Confidence Calibration
- **Calibration Curve**: Plot confidence vs actual accuracy per bin
- **Expected Calibration Error (ECE)**: Weighted average of |accuracy - confidence|
- **Overconfidence Rate**: % of wrong verdicts with confidence > 0.80

### Latency
- **P50/P95 Verification Time**: End-to-end from request to response
- **Agent Execution Time**: Per-agent breakdown
- **Tool Call Latency**: External API response times

### Provenance Quality
- **Data Source Coverage**: % of verdicts backed by at least one source
- **RAG Relevance**: % of retrieved chunks relevant to the claim
- **A2A Corroboration Rate**: % of SEC findings confirmed by news

## Regulatory Alignment

### SR 11-7 (Fed) — Model Risk Management
- **Model Documentation**: Architecture docs, prompt text files (versionable)
- **Validation**: Unit tests + golden tests provide ongoing validation
- **Audit Trail**: Every verification logged with tools called, data sources, and verdict

### E-23 (OCC) — Model Governance
- **Input Guardrails**: Injection detection, PII scrubbing, length limits
- **Output Guardrails**: HITL for low-confidence claims
- **Override Tracking**: HITL decisions logged with reviewer notes

### SS1/23 (PRA) — Model Risk Management for AI
- **Explainability**: Full reasoning chain from agent → verdict
- **Data Lineage**: Provenance tracking (XBRL, RAG, A2A) on every response
- **Human Oversight**: HITL checkpoint with approve/override/reject

## Prompt Versioning

System prompts are stored as text files in `src/finvet/agents/prompts/`:
- `sec_system.txt`
- `market_system.txt`
- `news_system.txt`

This enables:
1. **Git-tracked prompt changes** — every edit is a commit
2. **A/B testing** — load different prompt versions per request
3. **Validation regression** — re-run golden tests after prompt changes
4. **Audit compliance** — prompt version tied to each verification

## Implementation Roadmap

### Phase 1: Foundation (current)
- Unit tests for all critical paths
- Constants module for reproducible thresholds
- Prompt text files for versioning

### Phase 2: Golden Test Suite
- Build 50+ golden claims across all agent types
- Automated regression on PR merge
- Confidence calibration baseline

### Phase 3: LLM-as-Judge
- RAGAS faithfulness/relevance metrics
- Custom judge prompt for verdict correctness
- Weekly evaluation reports

### Phase 4: Enterprise Readiness
- PostgresSaver for production HITL persistence
- Prompt version tracking in audit trail
- Model card generation from test results
