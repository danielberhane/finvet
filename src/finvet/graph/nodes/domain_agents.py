"""Domain agent nodes for running ReAct verification agents.

These nodes wrap the SEC, Market, and News ReAct agents and integrate them
into the LangGraph workflow. Each agent reasons with the LLM configured for
the `agent` role about which tools to call, and returns structured evidence.
"""

import re
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
    # The same pre-flight the SEC and market routes run. It was wired into
    # those two and not this one, so a metric the vocabulary declined still
    # reached an agent here -- and news is the only route macro claims take.
    # A guard the relevant path never calls is not a guard.
    declined = _unsupported_claim(state)
    if declined is not None:
        logger.info(f"News claim declined: {declined['limitation']}")
        return {"agent_evidence": declined, "agent_type": "news"}

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
            # `claimed_value` on the model-invoked path is a tool argument the
            # model chose, not something the claim said. A claim naming no
            # amount was recorded carrying one -- "Apple was fined by the
            # European Commission" came back with claimed_value 500,000,000,
            # a figure the model supplied from its own reading. The claim is
            # the authority on what it claimed.
            if getattr(state.get("parsed_claim"), "value", None) is None:
                corroboration["claimed_value"] = None

    if corroboration is None and _policy_wants_corroboration(state, evidence):
        corroboration = _corroborate_by_policy(state, evidence)

    if corroboration:
        # SEC is authoritative for what an issuer disclosed. When the
        # delegation settled the number deterministically, the news verdict
        # follows it -- otherwise the answer reads NOT_ENOUGH_INFO beside a
        # filing that plainly answered the question, which is what sent a
        # $1 trillion fine claim to a human next to a filed EUR 500 million.
        #
        # `retrieved_value` is the licence, and it is a narrow one: since the
        # fail-closed fix it is non-None *only* when Python compared a trusted
        # observation. A model's reading of prose leaves it None, so this can
        # never adopt a number the model supplied.
        # Classification runs FIRST, against the news agent's *own* verdict.
        # Adopting before comparing would make the two sides agree by
        # construction and destroy the disagreement signal -- the parent would
        # be compared with a copy of the target, which is the same defect
        # reclassify_corroboration was written to fix, arriving from the other
        # direction.
        #
        # The single classification point. The tool cannot do this -- it runs
        # inside the ReAct loop, before this agent has a verdict to compare
        # against.
        # What the *press* said, not what this agent may certify. Those are
        # two different facts and the news agent holds both: `_apply_override`
        # discards the reading for verdict purposes -- fine_amount has no XBRL
        # concept, so prose may not settle a number -- and keeps it as
        # `llm_original_verdict`.
        #
        # Comparing the certification was the bug. It is a structural decline,
        # meaning "I have no way to certify a fine amount", and a decline can
        # neither agree nor disagree with a finding. Every delegation therefore
        # collapsed to NO_MATCHING_DISCLOSURE, and CORROBORATES / CONTRADICTS
        # were unreachable no matter how decisive the filing was.
        #
        # Only where the claim named a value, though. A valueless claim takes
        # the *qualitative* decline instead, and there the reading was thrown
        # out because refuting on absence is a fallacy (D13,
        # `non_corroboration_is_not_contradiction`) -- a judgment that the
        # reading was unsound, not merely uncertified. Reviving it as a
        # comparison side would let a verdict the system deliberately rejected
        # declare two sources in conflict.
        #
        # llm_original_verdict is also None on the deterministic-fallback and
        # error paths, where the final verdict is already deterministic and is
        # the right thing to compare.
        claim_named_a_value = getattr(
            state.get("parsed_claim"), "value", None) is not None
        parent_reading = (
            (evidence.get("llm_original_verdict") if claim_named_a_value else None)
            or evidence.get("verdict", "")
        )

        corroboration = reclassify_corroboration(parent_reading, corroboration)
        _adopt_filing_verdict(evidence, corroboration)
        logger.info(
            f"A2A status: news={evidence.get('verdict')} "
            f"sec={corroboration.get('verdict')} -> {corroboration.get('status')} "
            f"(trigger: {corroboration.get('trigger_mode')})"
        )
        result["corroboration_result"] = corroboration

    return result


_DECISIVE = frozenset({"SUPPORTS", "REFUTES"})


def _adopt_filing_verdict(evidence: Dict, corroboration: Dict) -> Dict:
    """Let a deterministically settled filing decide the news claim.

    Mutates and returns `evidence` so the caller's dict is the adopted one.

    Two conditions, both required. The delegation reached a decisive verdict,
    and it carries a `retrieved_value` -- which is non-None only when Python
    compared a trusted observation, never when a model read a number out of
    prose. Without both, nothing changes and the existing fail-closed answer
    stands.
    """
    verdict = corroboration.get("verdict")
    retrieved = corroboration.get("retrieved_value")
    if verdict not in _DECISIVE or retrieved is None:
        return evidence

    logger.info(
        f"Filing settled the claim: news={evidence.get('verdict')} -> {verdict} "
        f"(filed value {retrieved})"
    )
    evidence["verdict"] = verdict
    evidence["confidence"] = corroboration.get("confidence") or evidence.get("confidence")
    evidence["retrieved_value"] = retrieved
    # The source travels with the figure. Without it the response asserts a
    # number and names nothing that produced it -- Layer 3 scored exactly this
    # at 96.6% instead of 100%, on rows 49 and 50.
    evidence["trusted_observation"] = corroboration.get("trusted_observation")
    # The filing answered it, so the decline no longer applies. Left in place
    # it would suppress review reasoning for a verdict that now stands on a
    # trusted number.
    evidence["limitation"] = None
    evidence["verdict_source"] = "delegated_filing"
    return evidence


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
        # A failure is an outcome, not an absence. Returning None dropped the
        # delegation from state entirely, so the audit trail could not show
        # that a corroboration had been attempted and had broken -- which
        # reads exactly like a claim nobody thought to check.
        logger.error(f"A2A policy corroboration failed: {e}")
        from ...models.a2a import A2A_FAILED, A2AResult

        return A2AResult(
            success=False, status=A2A_FAILED, source_agent="news",
            target_agent="sec", trigger_mode="policy",
            finding=state.get("claim_raw", ""),
            metric=getattr(parsed, "metric", "") or "",
            claimed_value=getattr(parsed, "value", None),
            error=str(e),
        ).model_dump()


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
    scope_retrieval: bool = True,
) -> Dict:
    """Run the SEC agent, optionally with the resolved period applied.

    Shared by the normal SEC route and by the News -> SEC A2A delegation. The
    delegation must not reimplement this: calling SECAgent.execute directly
    skips use_period_target, and a nested agent reading a different fiscal
    period than the parent is the kind of inconsistency that surfaces later as
    an unexplainable disagreement between two of your own agents.

    `scope_retrieval=False` for that delegation, because it asks a different
    question. The SEC route asks what a named period reported, and must read
    that period. A corroboration asks whether the issuer has disclosed a matter
    *at all* -- and scoping it to one period answered "no" by excluding the
    filings that carry the disclosure. "Apple was fined by the European
    Commission over App Store practices" resolved to 2025-12-31; Apple's newest
    indexed filings close 2025-12-27 and 2025-09-27, both earlier, so every
    search returned zero and the nested agent died at its recursion limit. The
    same claim on the SEC route, where nothing is scoped, found evidence
    immediately.

    Temporal eligibility is not lost by this: `_filing_could_cover` already
    compares the event against the newest filing on record and returns
    NOT_APPLICABLE_YET when none could carry it. That check is the honest one
    and it sits above retrieval. Nothing numeric depends on the nested run's
    period either -- CORROBORATION_METRICS holds fine_amount and
    settlement_amount, never a GAAP figure.

    No recursion guard is needed. The SEC agent holds no delegation tool, so
    News -> SEC terminates by construction.
    """
    target = (period_target_for(state.get("canonical_period"))
              if scope_retrieval else None)
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


_FOURTH_QUARTER = re.compile(r"\bq\s*4\b|\bfourth\s+quarter\b", re.IGNORECASE)


def _names_fourth_quarter(period: Optional[str]) -> bool:
    """Whether the claim itself says Q4.

    Read from the claim rather than from the resolver's opinion of it.
    `Microsoft's Q4 fiscal 2025 revenue was $76 billion` returned three
    different verdicts in three consecutive runs, because the decline tested
    `canonical_period.fiscal_quarter` and period resolution is not
    deterministic for that phrasing. When it produced an annual period instead,
    the claim went to an agent, which retrieved the *annual* $281.7B and
    refuted a *quarterly* $76B with it -- a correct number from the wrong
    period scope, and the guard could not object because the annual fact
    matched the annual window it was given.

    Whether a claim names a quarter is a property of the claim.
    """
    return bool(period) and bool(_FOURTH_QUARTER.search(str(period)))


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
            and (getattr(canonical, "fiscal_quarter", None) == "Q4"
                 or _names_fourth_quarter(getattr(parsed, "period", None)))):
        return _limitation_evidence(
            "sec", "SEC EDGAR", "unsupported_q4_derivation",
            "Q4 figures are not filed separately and deriving them requires "
            "two differently-scoped retrievals, which this pipeline does not "
            "support. No verdict was attempted.")

    # A SEC lookup needs a CIK, which comes from a ticker. Without one there is
    # no filing to fetch, and the claim is unverifiable in principle rather than
    # merely unverified this time -- no larger tool budget or stronger model
    # changes that.
    #
    # "A large US bank posted $30 billion in net income last year" reached the
    # agent, which spent 14 tool calls and 150 seconds hunting a company that
    # was never named, then escalated to a reviewer who would read the same
    # claim and reach the same conclusion.
    #
    # `_policy_wants_corroboration` already declines on a missing ticker, and
    # the parser's vocabulary already contains `ambiguous_entity`. Enforced
    # here as well, deterministically, so the guarantee does not depend on the
    # model noticing.
    if (getattr(parsed, "claim_type", None) == "sec"
            and not getattr(parsed, "ticker", None)):
        return _limitation_evidence(
            "sec", "SEC EDGAR", "no_company_identified",
            "The claim does not name a company, and a filing lookup needs one. "
            "No verdict was attempted.")

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
