"""Agent-to-Agent corroboration in the News -> SEC direction.

The News agent verifies a reported event; this asks the issuer's own filing
whether that event is disclosed there. A 10-K's Legal Proceedings section is the
primary source and press coverage is secondary, so this direction checks the
weaker source against the stronger one.

Delegation is structured, not a sentence. The nested SEC agent receives a real
ParsedClaim carrying the claimed value, because _apply_override compares
parsed_claim.value against what the agent retrieved — hand it only prose and the
deterministic comparison silently does nothing, which is the one part of FinVet
that must never be skipped.
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from ..config.constants import A2A_MAX_ITERATIONS
from ..models.a2a import (
    A2A_FAILED,
    A2A_NO_MATCHING_DISCLOSURE,
    A2A_NOT_APPLICABLE_YET,
    A2AResult,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _filing_could_cover(event_date: str, latest_period_end: Optional[str]) -> bool:
    """Could a filing covering `latest_period_end` mention an event on `event_date`?

    A periodic filing is a point-in-time document. If the newest one closed
    before the event happened, its silence is not evidence of absence, and
    running the agent would only produce a misleading NOT_ENOUGH_INFO.
    """
    if not event_date or not latest_period_end:
        return True  # unknown dates: attempt it rather than refuse
    return event_date[:10] <= latest_period_end[:10]


@tool
def corroborate_with_filing(
    finding: str,
    ticker: str,
    metric: str = "",
    claimed_value: Optional[float] = None,
    operator: str = "eq",
    period: str = "",
    event_date: str = "",
) -> Dict[str, Any]:
    """Ask the SEC agent whether a company's own filing corroborates a reported event.

    Use this for a news claim about a fine, settlement, or other material event
    at an identifiable public company. Press coverage is secondary evidence; the
    company's own 10-K or 10-Q is the primary source, and material legal matters
    appear in Legal Proceedings and the contingency notes.

    Do NOT use for macroeconomic claims, analyst estimates, price targets, or
    anything with no issuer to file it.

    Args:
        finding: The event to check, stated plainly.
                 (e.g., "Apple was fined EUR 500 million by the European Commission")
        ticker: Company ticker symbol (e.g., "AAPL").
        metric: The claim's metric, e.g. "fine_amount" or "settlement_amount".
        claimed_value: The asserted amount in base units (e.g. 500000000).
                       Pass this whenever the claim names a number — it is what
                       lets the deterministic comparison run.
        operator: Comparison operator; "eq" unless the claim says otherwise.
        period: Period as stated (e.g. "2025"), if any.
        event_date: ISO date of the event (e.g. "2025-04-23"), if known.

    Returns:
        A2AResult fields as a dict: success, direction, status, verdict,
        confidence, reasoning, retrieved_value, claimed_value, sources,
        provenance, tools_used, trigger_mode, error.
    """
    # Dict, not the model: LangChain stringifies a tool's return value and a
    # BaseModel repr survives neither json.loads nor ast.literal_eval, so
    # _parse_provenance would drop it (see filing_search.py and commit 5110e91).
    return _corroborate(
        finding=finding, ticker=ticker, metric=metric,
        claimed_value=claimed_value, operator=operator,
        period=period, event_date=event_date, trigger_mode="agent",
    ).model_dump()


def _corroborate(
    *,
    finding: str,
    ticker: str,
    metric: str = "",
    claimed_value: Optional[float] = None,
    operator: str = "eq",
    period: str = "",
    event_date: str = "",
    trigger_mode: str = "agent",
) -> A2AResult:
    """Shared body, callable directly by the policy path in domain_agents.

    Kept separate from the @tool wrapper so run_news_agent can invoke it without
    going through LangChain's tool plumbing, and so it returns the model rather
    than a dict to its Python caller.

    This returns the target's findings and never classifies them. Status
    describes the relationship between the parent claim's verdict and the
    target's, and inside the News agent's loop the parent verdict does not
    exist yet. `reclassify_corroboration` decides it afterwards.
    """
    base = dict(
        direction="news_to_sec", source_agent="news", target_agent="sec",
        claimed_value=claimed_value, finding=finding, trigger_mode=trigger_mode,
        metric=metric or "",
        temporal_scope="checked" if event_date else "unknown",
    )
    try:
        # Imported here, not at module scope: agents import tools, so a top-level
        # import would close the cycle.
        from ..graph.nodes.domain_agents import run_sec_agent_scoped
        from ..graph.nodes.period_resolver import period_resolver
        from ..models.claim import ParsedClaim

        logger.info(f"A2A corroboration: News -> SEC for '{finding[:80]}...'")

        # metric is deliberately dropped from the nested claim. ParsedClaim
        # validates metric against claim_type, and fine_amount/settlement_amount
        # are *news* metrics — they have no GAAP concept, which is precisely why
        # the filing must be searched as narrative rather than looked up in XBRL.
        # metric only feeds context display and the XBRL-concept fallback; the
        # claimed *value* is what _apply_override needs, and that survives.
        # The original metric is kept on the result for the audit trail.
        # operator and value are a validated pair: a comparator with nothing to
        # compare is rejected, so a claim that names no amount carries neither.
        parsed = ParsedClaim(
            claim_type="sec",
            ticker=ticker or None,
            metric=None,
            operator=(operator or "eq") if claimed_value is not None else None,
            value=claimed_value,
            period=period or None,
            reject_reason=None,
        )
        state: Dict[str, Any] = {
            "claim_raw": finding,
            "request_id": "a2a-filing-corroboration",
            "user_id": "a2a",
            "parsed_claim": parsed,
        }
        # Resolve the period the same way the normal SEC route does, so the
        # nested agent targets the same filing the parent would have.
        try:
            state.update(period_resolver(state))
        except Exception as e:  # a claim with no period still deserves an attempt
            logger.info(f"A2A period resolution skipped: {e}")

        if not _filing_could_cover(event_date, _latest_period_end(state)):
            return A2AResult(
                success=True, status=A2A_NOT_APPLICABLE_YET,
                verdict="NOT_ENOUGH_INFO",
                reasoning=(
                    f"No filing on record covers {event_date}; the most recent "
                    f"period closed earlier, so its silence is not evidence."
                ),
                **base,
            )

        out = run_sec_agent_scoped(
            state, max_iterations=A2A_MAX_ITERATIONS
        )
        ev = out.get("agent_evidence", {})

        # run_sec_agent_scoped catches its own failures and returns
        # NOT_ENOUGH_INFO evidence, which is indistinguishable from an
        # authoritative filing that genuinely says nothing. Only one of those
        # is evidence, so read the execution status rather than the verdict.
        if ev.get("execution_status") == "failed":
            logger.error(
                f"A2A delegation failed inside the SEC agent: {ev.get('error')}"
            )
            return A2AResult(
                success=False, status=A2A_FAILED,
                error=ev.get("error") or "nested SEC agent failed",
                **base,
            )

        target_verdict = ev.get("verdict", "NOT_ENOUGH_INFO")

        # Pull filing references straight out of the nested agent's provenance.
        # run_sec_agent_scoped deliberately does not do the rag_chunks_retrieved
        # extraction that run_sec_agent does — that writes into pipeline state,
        # which a delegated run has no business touching. Reading provenance
        # here keeps the evidence without that side effect.
        sources = []
        for prov in ev.get("provenance", []) or []:
            if prov.get("tool") != "search_filing_text":
                continue
            result = prov.get("result") or {}
            if not isinstance(result, dict):
                continue
            for chunk in result.get("chunks", []) or []:
                sources.append({
                    "filing_type": chunk.get("filing_type"),
                    "period_end": chunk.get("period_end"),
                    "section": chunk.get("section"),
                    "query": prov.get("args", {}).get("query"),
                })

        logger.info(
            f"A2A corroboration result: {target_verdict} "
            f"(confidence: {ev.get('confidence', 0):.2f}, mode: {trigger_mode})"
        )

        return A2AResult(
            success=True,
            # Neutral placeholder: reclassify_corroboration sets the real
            # status once the parent verdict exists. Never classify here.
            status=A2A_NO_MATCHING_DISCLOSURE,
            verdict=target_verdict,
            confidence=ev.get("confidence", 0.0),
            reasoning=ev.get("reasoning", ""),
            retrieved_value=ev.get("retrieved_value"),
            sources=sources,
            provenance=ev.get("provenance", []) or [],
            tools_used=ev.get("tools_called", []) or [],
            **base,
        )

    except Exception as e:
        logger.error(f"corroborate_with_filing failed: {e}")
        return A2AResult(
            success=False, status=A2A_FAILED, error=str(e), **base
        )


def _latest_period_end(state: Dict[str, Any]) -> Optional[str]:
    """End date of the resolved period, when one exists."""
    cp = state.get("canonical_period")
    return getattr(cp, "end_date", None) if cp else None
