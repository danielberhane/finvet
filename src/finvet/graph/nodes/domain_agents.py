"""Domain agent nodes for running ReAct verification agents.

These nodes wrap the SEC, Market, and News ReAct agents and integrate them
into the LangGraph workflow. Each agent uses DeepSeek to reason about
which tools to call and returns structured evidence.
"""

from typing import Dict, Type
from ...config.constants import AGENT_MAX_ITERATIONS
from ...models.state import VerificationState
from ...agents import SECAgent, MarketAgent, NewsAgent
from ...agents.base import BaseVerificationAgent, compose_failure_reasoning
from ...tools.sec_tools import period_target_for, use_period_target
from ...utils.logging import get_logger

logger = get_logger(__name__)


def _error_evidence(agent_type: str, source_desc: str, error_msg: str) -> Dict:
    """Build fallback evidence dict on agent failure."""
    return {
        "agent": agent_type,
        "verdict": "NOT_ENOUGH_INFO",
        "confidence": 0.2,
        "retrieved_value": None,
        "source_description": source_desc,
        "source_url": None,
        "magnitude_difference_percent": None,
        "tools_called": [],
        "tool_calls_detail": [],
        "reasoning": compose_failure_reasoning(error_msg),
        "execution_time_ms": 0,
        "override_applied": False,
        "llm_original_verdict": None,
    }


def _run_agent(
    agent_cls: Type[BaseVerificationAgent],
    agent_type: str,
    source_desc: str,
    state: VerificationState,
) -> Dict:
    """Common agent execution wrapper.

    Creates the agent, runs it, logs the result, and returns
    the evidence dict. Falls back to error evidence on failure.
    """
    request_id = state.get("request_id", "unknown")
    logger.info(f"Running {agent_type.upper()} agent (request: {request_id})")

    try:
        agent = agent_cls(max_iterations=AGENT_MAX_ITERATIONS)
        evidence = agent.execute(state)

        logger.info(
            f"{agent_type.upper()} agent completed: {evidence['verdict']} "
            f"(confidence: {evidence['confidence']:.2f}, "
            f"tools: {len(evidence['tools_called'])})"
        )

        return {
            "agent_evidence": evidence,
            "agent_type": agent_type,
        }

    except Exception as e:
        logger.error(f"{agent_type.upper()} agent failed: {e}")
        return {
            "agent_evidence": _error_evidence(agent_type, source_desc, str(e)),
            "agent_type": agent_type,
        }


def run_market_agent(state: VerificationState) -> Dict:
    """Run the Market ReAct agent for market data claims."""
    return _run_agent(MarketAgent, "market", "Finnhub", state)


def run_news_agent(state: VerificationState) -> Dict:
    """Run the News ReAct agent for news/event claims."""
    return _run_agent(NewsAgent, "news", "Financial News", state)


def run_sec_agent(state: VerificationState) -> Dict:
    """Run the SEC ReAct agent with RAG/A2A provenance extraction.

    The period resolved upstream is applied to every SEC tool call the agent
    makes. It is injected rather than passed as a tool argument: period_resolver
    already determined it, so routing it through the model would only create a
    chance for it to arrive wrong.
    """
    target = period_target_for(state.get("canonical_period"))
    if target:
        logger.info(f"SEC retrieval targeting period {target[0]} ({target[1]})")
    with use_period_target(*(target or (None, None))):
        result = _run_agent(SECAgent, "sec", "SEC EDGAR", state)

    # SEC-specific: extract RAG and A2A provenance into dedicated state fields
    evidence = result.get("agent_evidence", {})
    rag_chunks = []
    corroboration = None
    for prov in evidence.get("provenance", []):
        prov_result = prov.get("result", {})
        if not isinstance(prov_result, dict) or not prov_result.get("success"):
            continue
        if prov["tool"] == "search_filing_text":
            for chunk in prov_result.get("chunks", []):
                chunk["search_query"] = prov.get("args", {}).get("query", "")
                rag_chunks.append(chunk)
        elif prov["tool"] == "corroborate_with_news":
            corroboration = prov_result
            corroboration["finding"] = prov.get("args", {}).get("finding", "")

    if rag_chunks:
        result["rag_chunks_retrieved"] = rag_chunks
    if corroboration:
        result["corroboration_result"] = corroboration

    return result
