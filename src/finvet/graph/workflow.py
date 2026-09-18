"""LangGraph workflow for the FinVet verification pipeline.

This module registers 12 nodes. At most 9 run for any single claim: the router
selects one domain agent, and the two HITL nodes only execute when confidence
falls below the threshold.

1.  input_guardrails    - Validate and sanitize input
2.  claim_parser        - Parse claim into structured fields
3.  period_resolver     - Resolve time periods to dates (SEC claims only)
4.  sec_agent           - ReAct verification against SEC EDGAR
5.  market_agent        - ReAct verification against Finnhub
6.  news_agent          - ReAct verification against Tavily
7.  reject_handler      - Terminal path for unsafe / non-financial claims
8.  confidence_adjuster - Confidence adjustment on the agent verdict
9.  output_guardrails   - Confidence threshold + output safety -> HITL routing
10. hitl_gate           - the graph pauses before it for human review
11. apply_hitl_decision - Apply the reviewer's decision
12. response_generator  - Format final response

LangGraph checkpointing enables HITL:
- Graph pauses at hitl_gate when hitl_required=True
- State persists via checkpointer (MemorySaver or PostgresSaver)
- Human reviews via /review endpoint, graph resumes with decision
"""

from typing import Dict

from langgraph.graph import StateGraph, END

from ..models.state import VerificationState
from .nodes import (
    input_guardrails,
    claim_parser,
    period_resolver,
    run_sec_agent,
    run_market_agent,
    run_news_agent,
    output_guardrails,
    response_generator,
)
from ..audit import get_audit_logger
from ..config.constants import (
    CONFIDENCE_CLOSE_MATCH_BONUS,
    CONFIDENCE_CLOSE_MATCH_PCT,
    CONFIDENCE_LARGE_DIFF_PENALTY,
    CONFIDENCE_LARGE_DIFF_PCT,
    CONFIDENCE_AUTOMATED_CAP,
    CONFIDENCE_THOROUGH_BONUS,
    CONFIDENCE_THOROUGH_TOOL_COUNT,
)
from ..utils.logging import get_logger
from ..utils.helpers import get_confidence_label

logger = get_logger(__name__)


def create_verification_graph(checkpointer=None):
    """
    Create the LangGraph workflow for claim verification.

    Args:
        checkpointer: Optional LangGraph checkpointer (MemorySaver, PostgresSaver, etc.)
                      for HITL persistence. If provided, compiles with interrupt_before
                      so the graph pauses at hitl_gate for human review.

    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = StateGraph(VerificationState)

    # Add nodes
    graph.add_node("input_guardrails", input_guardrails)
    graph.add_node("claim_parser", claim_parser)
    graph.add_node("period_resolver", period_resolver)
    graph.add_node("sec_agent", run_sec_agent)
    graph.add_node("market_agent", run_market_agent)
    graph.add_node("news_agent", run_news_agent)
    graph.add_node("reject_handler", _handle_rejection)
    graph.add_node("confidence_adjuster", _adjust_confidence)
    graph.add_node("output_guardrails", output_guardrails)
    graph.add_node("hitl_gate", _hitl_gate)
    graph.add_node("apply_hitl_decision", _apply_hitl_decision)
    graph.add_node("response_generator", response_generator)

    # Define edges
    graph.set_entry_point("input_guardrails")
    graph.add_edge("input_guardrails", "claim_parser")

    # After parsing, route based on claim type
    graph.add_conditional_edges(
        "claim_parser",
        _route_after_parsing,
        {
            "sec": "period_resolver",
            "market": "market_agent",
            "news": "news_agent",
            "reject": "reject_handler",
        }
    )

    # After period resolution, run SEC agent
    graph.add_edge("period_resolver", "sec_agent")

    # All agents go to the confidence adjuster
    graph.add_edge("sec_agent", "confidence_adjuster")
    graph.add_edge("market_agent", "confidence_adjuster")
    graph.add_edge("news_agent", "confidence_adjuster")

    # Rejection goes directly to response
    graph.add_edge("reject_handler", "response_generator")

    # Confidence adjuster to output guardrails
    graph.add_edge("confidence_adjuster", "output_guardrails")

    # After output guardrails, route based on whether HITL is needed
    # Non-HITL claims skip hitl_gate entirely (no interrupt)
    graph.add_conditional_edges(
        "output_guardrails",
        _route_after_guardrails,
        {
            "needs_hitl": "hitl_gate",
            "no_hitl": "response_generator",
        }
    )

    # HITL gate → apply decision → generate response
    # With a checkpointer, interrupt_before pauses BEFORE hitl_gate.
    # When resumed (after human review), hitl_gate runs, then the
    # decision is applied, and response_generator produces the final output.
    graph.add_edge("hitl_gate", "apply_hitl_decision")
    graph.add_edge("apply_hitl_decision", "response_generator")

    # Response generator is the end
    graph.add_edge("response_generator", END)

    # Compile with or without checkpointer
    if checkpointer:
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["hitl_gate"],
        )
        logger.info("Graph compiled with checkpointer and HITL interrupt support")
    else:
        compiled = graph.compile()
        logger.warning("Graph compiled WITHOUT checkpointer - HITL interrupts disabled")

    return compiled


def _route_after_parsing(state: VerificationState) -> str:
    """Route to appropriate agent based on claim_type."""
    parsed_claim = state.get("parsed_claim")

    if not parsed_claim:
        logger.error("No parsed claim in state")
        return "reject"

    claim_type = parsed_claim.claim_type
    logger.info(f"Routing claim type: {claim_type}")
    return claim_type


def _handle_rejection(state: VerificationState) -> Dict:
    """Handle rejected claims."""
    parsed_claim = state.get("parsed_claim")
    return {
        "verdict": "REJECTED",
        "confidence": 1.0,
        "confidence_label": "HIGH",
        "disposition": "rejected_parser",
        "disposition_detail": getattr(parsed_claim, "reject_reason", None),
    }


def _adjust_confidence(state: VerificationState) -> Dict:
    """Adjusts confidence. Never overturns a verdict.

    One agent runs per claim; its verdict is copied out unchanged and every
    branch below touches only the confidence. The single exception is the
    first guard -- if the agent produced no evidence at all there is no
    verdict to carry, and NOT_ENOUGH_INFO is the honest answer rather than a
    judgement about the claim. A verdict genuinely changes in exactly two
    places: `_apply_override` in the agent, and `_apply_hitl_decision` below.

    Three signals nudge the agent's own confidence, then a cap: a very close
    numeric match, a very large mismatch, and whether the agent called
    several tools. The thresholds live in `config/constants.py` so they can
    be read without reading this function; the cap is below 1.0 because
    full confidence is reserved for a human decision.
    """
    agent_evidence = state.get("agent_evidence", {})

    if not agent_evidence:
        return {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.2,
            "confidence_label": "LOW",
        }

    verdict = agent_evidence.get("verdict", "NOT_ENOUGH_INFO")
    confidence = agent_evidence.get("confidence", 0.5)

    adjustments = []
    parsed_claim = state.get("parsed_claim")
    operator = (
        (getattr(parsed_claim, "operator", None) or "eq")
        if parsed_claim else "eq"
    )
    magnitude_diff = agent_evidence.get("magnitude_difference_percent")
    # Magnitude adjustments apply to strict equality only. Directional claims
    # (gt/gte/lt/lte) expect large differences, and approx/range stated their
    # own imprecision — no close-match bonus for a claim that never promised
    # precision, no penalty for one the override already judged at the
    # widened tolerance.
    if magnitude_diff is not None and operator == "eq":
        if magnitude_diff > CONFIDENCE_LARGE_DIFF_PCT:
            confidence += CONFIDENCE_LARGE_DIFF_PENALTY
            adjustments.append({"reason": "large_magnitude_difference", "amount": CONFIDENCE_LARGE_DIFF_PENALTY})
        elif magnitude_diff < CONFIDENCE_CLOSE_MATCH_PCT:
            confidence += CONFIDENCE_CLOSE_MATCH_BONUS
            adjustments.append({"reason": "close_match", "amount": CONFIDENCE_CLOSE_MATCH_BONUS})

    tools_called = agent_evidence.get("tools_called", [])
    if len(tools_called) >= CONFIDENCE_THOROUGH_TOOL_COUNT:
        confidence += CONFIDENCE_THOROUGH_BONUS
        adjustments.append({"reason": "thorough_investigation", "amount": CONFIDENCE_THOROUGH_BONUS})

    confidence = max(0.0, min(CONFIDENCE_AUTOMATED_CAP, confidence))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_label": get_confidence_label(confidence),
        "confidence_adjustments": adjustments,
    }


def _route_after_guardrails(state: VerificationState) -> str:
    """Route after output guardrails based on whether HITL review is needed.

    Non-HITL claims skip hitl_gate entirely so they aren't
    paused by interrupt_before.
    """
    if state.get("hitl_required", False):
        return "needs_hitl"
    return "no_hitl"


def _hitl_gate(state: VerificationState) -> Dict:
    """The gate a flagged claim waits at for a human.

    The pause is not in this function. The graph is compiled with
    interrupt_before=["hitl_gate"], so LangGraph stops and the checkpointer
    saves state *before* this node runs; the /review route writes the
    decision into that saved state and resumes. Only then does this node
    execute -- it records the gate in the audit trail and marks it passed.
    """
    request_id = state.get("request_id", "unknown")
    hitl_required = state.get("hitl_required", False)

    if hitl_required:
        # Record the gate in the audit trail
        audit = get_audit_logger()
        audit.log_event(
            event_type="hitl_gate_reached",
            request_id=request_id,
            data={
                "hitl_triggers": state.get("hitl_triggers", []),
                "verdict_before_hitl": state.get("verdict"),
                "confidence_before_hitl": state.get("confidence"),
            }
        )
        logger.info(f"HITL gate reached after review (request: {request_id})")

    return {
        "hitl_gate_passed": True,
    }


def _apply_hitl_decision(state: VerificationState) -> Dict:
    """Apply human reviewer's decision if any."""
    request_id = state.get("request_id", "unknown")
    hitl_decision = state.get("hitl_decision")
    hitl_override_verdict = state.get("hitl_override_verdict")
    hitl_reviewer_notes = state.get("hitl_reviewer_notes")

    if hitl_decision is None:
        # No HITL decision (running without checkpointer, or non-HITL claim).
        # response_generator already returns a pending_review response for this
        # case (hitl_required and hitl_decision is None), so a flagged verdict is
        # never released — nothing to suppress here.
        return {}

    audit = get_audit_logger()

    if hitl_decision == "approve":
        audit.log_event(
            event_type="hitl_approved",
            request_id=request_id,
            data={"notes": hitl_reviewer_notes}
        )
        logger.info(f"HITL approved automated verdict (request: {request_id})")
        return {
            "hitl_applied": True,
            "disposition": "approved_human",
            "disposition_detail": hitl_reviewer_notes,
        }

    elif hitl_decision == "override":
        audit.log_event(
            event_type="hitl_overridden",
            request_id=request_id,
            data={
                "original_verdict": state.get("verdict"),
                "override_verdict": hitl_override_verdict,
                "notes": hitl_reviewer_notes,
            }
        )
        logger.info(
            f"HITL overrode verdict to {hitl_override_verdict} (request: {request_id})"
        )
        return {
            "verdict": hitl_override_verdict,
            "confidence": 0.95,
            "confidence_label": "HIGH",
            "hitl_applied": True,
            "disposition": "overridden_human",
            "disposition_detail": hitl_reviewer_notes,
        }

    elif hitl_decision == "reject":
        audit.log_event(
            event_type="hitl_rejected",
            request_id=request_id,
            data={"notes": hitl_reviewer_notes}
        )
        logger.info(f"HITL rejected claim (request: {request_id})")
        return {
            "verdict": "REJECTED",
            "confidence": 1.0,
            "confidence_label": "HIGH",
            "hitl_applied": True,
            "disposition": "rejected_human",
            "disposition_detail": hitl_reviewer_notes,
        }

    return {}


