"""Domain agent nodes for running ReAct verification agents.

These nodes wrap the SEC, Market, and News ReAct agents and integrate them
into the LangGraph workflow. Each agent uses DeepSeek to reason about
which tools to call and returns structured evidence.
"""

from typing import Dict, Optional, Type
from ...config.constants import AGENT_MAX_ITERATIONS, CORROBORATION_METRICS
from ...config.metrics import verification_strategy_for
from ...models.a2a import reclassify_corroboration
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
        # Why the run ended. A2A needs this: an agent that crashed and an
        # authoritative filing that says nothing both yield NOT_ENOUGH_INFO,
        # and only one of them is evidence.
        "execution_status": "failed",
        "error": error_msg,
    }


def _run_agent(
    agent_cls: Type[BaseVerificationAgent],
    agent_type: str,
    source_desc: str,
    state: VerificationState,
    **agent_kwargs,
) -> Dict:
    """Common agent execution wrapper.

    Creates the agent, runs it, logs the result, and returns
    the evidence dict. Falls back to error evidence on failure.

    agent_kwargs reach the agent constructor — used by the A2A path to build a
    SEC agent with a reduced iteration budget for A2A delegation.
    """
    request_id = state.get("request_id", "unknown")
    logger.info(f"Running {agent_type.upper()} agent (request: {request_id})")

    try:
        agent_kwargs.setdefault("max_iterations", AGENT_MAX_ITERATIONS)
        agent = agent_cls(**agent_kwargs)
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
    declined = _unsupported_claim(state)
    if declined is not None:
        logger.info(f"Market claim declined: {declined['limitation']}")
        return {"agent_evidence": declined, "agent_type": "market"}
    return _run_agent(MarketAgent, "market", "Finnhub", state)


def run_news_agent(state: VerificationState) -> Dict:
    """Run the News ReAct agent, corroborating material events against filings.

    Two ways the delegation happens, and they are not the same mechanism:

    - "agent": the model called corroborate_with_filing itself, so the result is
      already in the ReAct provenance and is lifted out below.
    - "policy": the model did not, but the claim is one an issuer's filing can
      settle. The tool is invoked here, after the loop — which means the result
      is NOT in the message history and must be attached explicitly. Provenance
      is built from AIMessage/ToolMessage pairs (base.py:198), and a call made
      out here never appears there.
    """
    result = _run_agent(NewsAgent, "news", "Financial News", state)
    evidence = result.get("agent_evidence", {})

    corroboration = None
    for prov in evidence.get("provenance", []):
        prov_result = prov.get("result", {})
        # A failed delegation is lifted too, not skipped. Dropping it left no
        # record that corroboration had been attempted at all, which is a worse
        # audit trail than mislabelling the failure -- and it silently handed
        # the claim to the policy path to run a second time.
        # reclassify_corroboration maps success=False to A2A_FAILED.
        if not isinstance(prov_result, dict):
            continue
        if prov["tool"] == "corroborate_with_filing":
            corroboration = prov_result
            corroboration.setdefault("finding", prov.get("args", {}).get("finding", ""))

    if corroboration is None and _policy_wants_corroboration(state, evidence):
        corroboration = _corroborate_by_policy(state, evidence)

    if corroboration:
        # The single classification point. The tool cannot do this -- it runs
        # inside the ReAct loop, before this agent has a verdict to compare
        # against -- and when it tried, it compared the SEC verdict with itself
        # and recorded contradictions as agreement.
        corroboration = reclassify_corroboration(
            evidence.get("verdict", ""), corroboration
        )
        logger.info(
            f"A2A status: news={evidence.get('verdict')} "
            f"sec={corroboration.get('verdict')} -> {corroboration.get('status')} "
            f"(trigger: {corroboration.get('trigger_mode')})"
        )
        result["corroboration_result"] = corroboration

    return result


def _policy_wants_corroboration(state: VerificationState, evidence: Dict) -> bool:
    """Should the node delegate even though the model did not ask to?

    Narrow on purpose: a metric an issuer must disclose, a company to look it up
    against, and an actual news verdict to check. Without all three the nested
    run costs latency and returns nothing meaningful.
    """
    parsed = state.get("parsed_claim")
    if parsed is None or getattr(parsed, "metric", None) not in CORROBORATION_METRICS:
        return False
    if not getattr(parsed, "ticker", None):
        return False
    return bool(evidence.get("verdict"))


def _corroborate_by_policy(state: VerificationState, evidence: Dict) -> Optional[Dict]:
    """Invoke the SEC delegation from the node and return its result dict."""
    from ...tools.corroborate_sec import _corroborate

    parsed = state.get("parsed_claim")
    logger.info(
        f"A2A policy trigger: metric={getattr(parsed, 'metric', None)} "
        f"ticker={getattr(parsed, 'ticker', None)}"
    )
    try:
        return _corroborate(
            finding=state.get("claim_raw", ""),
            ticker=getattr(parsed, "ticker", "") or "",
            metric=getattr(parsed, "metric", "") or "",
            claimed_value=getattr(parsed, "value", None),
            operator=getattr(parsed, "operator", None) or "eq",
            period=getattr(parsed, "period", "") or "",
            event_date=_event_date_for(state),
            trigger_mode="policy",
        ).model_dump()
    except Exception as e:
        logger.error(f"A2A policy corroboration failed: {e}")
        return None


def _event_date_for(state: VerificationState) -> str:
    """ISO date of the claimed event, when the resolved period names one.

    The temporal gate needs a date to decide whether any filing could yet cover
    the event. On the model-triggered path the model supplies it; the policy
    path has no model in the loop, so it reads the resolved period instead.
    Returning "" is honest -- the result records temporal_scope="unknown" and
    the attempt still runs, rather than silently claiming the gate was applied.
    """
    canonical = state.get("canonical_period")
    end_date = getattr(canonical, "end_date", None) if canonical else None
    return end_date or ""


def run_sec_agent_scoped(
    state: VerificationState,
    *,
    max_iterations: Optional[int] = None,
) -> Dict:
    """Run the SEC agent with the resolved period applied to every tool call.

    Shared by the normal SEC route and by the News -> SEC A2A delegation. The
    delegation must not reimplement this: calling SECAgent.execute directly
    skips use_period_target, and a nested agent reading a different fiscal
    period than the parent is the kind of inconsistency that surfaces later as
    an unexplainable disagreement between two of your own agents.

    No recursion guard is needed. The SEC agent holds no delegation tool, so
    News -> SEC terminates by construction.
    """
    target = period_target_for(state.get("canonical_period"))
    if target:
        logger.info(f"SEC retrieval targeting period {target[0]} ({target[1]})")
    kwargs = {}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    with use_period_target(*(target or (None, None))):
        return _run_agent(SECAgent, "sec", "SEC EDGAR", state, **kwargs)


def _limitation_evidence(agent_type: str, source_desc: str, limitation: str,
                         reasoning: str) -> Dict:
    """Evidence for a claim the system knowingly cannot verify.

    Distinct from _error_evidence: nothing failed. The system is declining a
    claim it has no way to answer, and says which limitation applies rather
    than letting an agent improvise and return a confident guess.
    """
    evidence = _error_evidence(agent_type, source_desc, reasoning)
    evidence.update({
        "execution_status": "completed",
        "error": None,
        "limitation": limitation,
        "reasoning": reasoning,
    })
    return evidence


def _unsupported_claim(state: VerificationState) -> Optional[Dict]:
    """Reasons to decline before an agent runs, or None to proceed."""
    parsed = state.get("parsed_claim")
    if parsed is None:
        return None

    # Q4 numeric claims. Deriving Q4 needs a 12-month fact minus a nine-month
    # fact, and retrieval is scoped to one resolved period per request, so the
    # pair cannot be requested. The duration guard in sec_edgar means this
    # already fails safe; declining up front makes the limitation legible
    # instead of surfacing as an unexplained NOT_ENOUGH_INFO.
    canonical = state.get("canonical_period")
    if (getattr(parsed, "claim_type", None) == "sec"
            and getattr(parsed, "value", None) is not None
            and getattr(canonical, "fiscal_quarter", None) == "Q4"):
        return _limitation_evidence(
            "sec", "SEC EDGAR", "unsupported_q4_derivation",
            "Q4 figures are not filed separately and deriving them requires "
            "two differently-scoped retrievals, which this pipeline does not "
            "support. No verdict was attempted.")

    # A metric no tool can serve. SERVABLE_METRICS knew about the gap and
    # nothing consulted it, so these reached an agent with no way to answer.
    if verification_strategy_for(parsed) == "unsupported":
        return _limitation_evidence(
            {"sec": "sec", "market": "market"}.get(
                getattr(parsed, "claim_type", ""), "news"),
            "unavailable", "unsupported_metric",
            f"No available tool serves the metric "
            f"{getattr(parsed, 'metric', None)!r}. No verdict was attempted.")

    return None


def run_sec_agent(state: VerificationState) -> Dict:
    """Run the SEC ReAct agent with RAG provenance extraction.

    The period resolved upstream is applied to every SEC tool call the agent
    makes. It is injected rather than passed as a tool argument: period_resolver
    already determined it, so routing it through the model would only create a
    chance for it to arrive wrong.
    """
    declined = _unsupported_claim(state)
    if declined is not None:
        logger.info(f"SEC claim declined: {declined['limitation']}")
        return {"agent_evidence": declined, "agent_type": "sec"}

    result = run_sec_agent_scoped(state)

    # SEC-specific: lift retrieved filing chunks into a dedicated state field
    evidence = result.get("agent_evidence", {})
    rag_chunks = []
    for prov in evidence.get("provenance", []):
        prov_result = prov.get("result", {})
        if not isinstance(prov_result, dict) or not prov_result.get("success"):
            continue
        if prov["tool"] == "search_filing_text":
            for chunk in prov_result.get("chunks", []):
                chunk["search_query"] = prov.get("args", {}).get("query", "")
                rag_chunks.append(chunk)

    if rag_chunks:
        result["rag_chunks_retrieved"] = rag_chunks

    return result
