"""Response generator node for creating final user-facing responses."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ...models.state import VerificationState
from ...utils.logging import get_logger
from ...utils.helpers import get_confidence_label, build_preliminary_analysis

logger = get_logger(__name__)

# Dispositions that mean the run ended without a verdict on the claim's merits.
_REJECT_DISPOSITIONS = {"rejected_parser", "rejected_human", "rejected_input_guard"}

_REJECT_REASON_MESSAGES = {
    "non_financial": "This is not a financial claim that can be verified.",
    "question": "This is a question, not a claim. Please rephrase as a statement.",
    "incomplete": "The claim is missing required information (ticker or value).",
    "advice_seeking": "FinVet verifies factual claims; it does not give investment advice.",
}


def response_generator(state: VerificationState) -> Dict:
    """
    Generate final formatted response for the user.

    Creates a complete response with:
    - Verdict and confidence
    - Summary and explanation
    - Source citations
    - Disclosures
    - Metadata

    Args:
        state: Current verification state with agent evidence

    Returns:
        Dictionary with final_response
    """
    request_id = state.get("request_id", "unknown")
    claim_raw = state.get("claim_raw", "")
    hitl_required = state.get("hitl_required", False)
    parsed_claim = state.get("parsed_claim")
    agent_evidence = state.get("agent_evidence")

    logger.info(f"Response generator starting (request: {request_id})")

    # If HITL is required and no decision has been made, return pending response
    # After HITL resume, hitl_decision is set so we generate a final response
    hitl_decision = state.get("hitl_decision")
    if hitl_required and hitl_decision is None:
        return _generate_hitl_response(state)

    # If the run was rejected, return a rejection response. Keyed on disposition
    # so this covers a human reviewer's reject (where claim_type is still
    # "sec"/"market"/"news") and not just the parser reject path. The claim_type
    # check remains as a fallback for states that never passed through
    # reject_handler — e.g. a checkpoint written before disposition existed.
    if (
        state.get("disposition") in _REJECT_DISPOSITIONS
        or (parsed_claim and parsed_claim.claim_type == "reject")
    ):
        return _generate_rejection_response(state)

    # If no agent evidence, return error
    if not agent_evidence:
        return _generate_error_response(
            request_id,
            "NO_EVIDENCE",
            "Verification failed to produce evidence"
        )

    # Generate success response
    verdict = state.get("verdict", agent_evidence.get("verdict", "NOT_ENOUGH_INFO"))
    confidence = state.get("confidence", agent_evidence.get("confidence", 0.5))
    confidence_label = state.get("confidence_label", get_confidence_label(confidence))

    # Build summary
    summary = _build_summary(verdict, confidence, agent_evidence, parsed_claim)

    # Build explanation
    explanation = _build_explanation(verdict, agent_evidence, state)

    # Format sources from agent evidence + RAG/A2A provenance
    sources = _format_sources(agent_evidence, state)

    # Collect disclosures
    disclosures = []
    if state.get("canonical_period") and hasattr(state["canonical_period"], "assumptions"):
        disclosures.extend(state["canonical_period"].assumptions)

    final_response = {
        "status": "success",
        "request_id": request_id,
        "claim": claim_raw,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "summary": summary,
        "explanation": explanation,
        "sources": sources,
        "disclosures": disclosures,
        "metadata": _format_metadata(state, agent_evidence),
    }

    logger.info(
        f"Response generated: {verdict} ({confidence:.2f}) (request: {request_id})"
    )

    return {
        "final_response": final_response,
        "execution_end_time": datetime.utcnow().isoformat(),
    }


def _generate_hitl_response(state: VerificationState) -> Dict:
    """Generate response for HITL pending case with full agent evidence."""
    request_id = state.get("request_id", "unknown")
    hitl_triggers = state.get("hitl_triggers", [])
    claim_raw = state.get("claim_raw", "")
    agent_evidence = state.get("agent_evidence", {})

    preliminary_analysis = build_preliminary_analysis(state, agent_evidence)
    agent_type = preliminary_analysis["agent"]
    explanation = preliminary_analysis["reasoning"] or f"Review triggers: {', '.join(hitl_triggers)}"

    final_response = {
        "status": "pending_review",
        "request_id": request_id,
        "claim": claim_raw,
        "verdict": "PENDING",
        "confidence": 0.0,
        "confidence_label": "PENDING",
        "summary": "This claim requires human review before a verdict can be provided.",
        "explanation": explanation,
        "sources": _format_sources(agent_evidence, state) if agent_evidence else [],
        "disclosures": ["Human review required for this claim"],
        "metadata": {
            "hitl_required": True,
            "hitl_triggers": hitl_triggers,
            "agent": agent_type,
            "tools_called": preliminary_analysis["tools_called"],
            "disposition": "pending_review",
            "disposition_detail": ", ".join(hitl_triggers) or None,
        },
        "preliminary_analysis": preliminary_analysis,
    }

    return {
        "final_response": final_response,
        "execution_end_time": datetime.utcnow().isoformat(),
    }


def _generate_rejection_response(state: VerificationState) -> Dict:
    """Generate response for rejected claims (parser reject or human reject)."""
    request_id = state.get("request_id", "unknown")
    claim_raw = state.get("claim_raw", "")
    parsed_claim = state.get("parsed_claim")

    # Fall back to the parsed claim for states that predate disposition.
    disposition = state.get("disposition") or "rejected_parser"
    detail = state.get("disposition_detail") or getattr(
        parsed_claim, "reject_reason", None
    )

    if disposition == "rejected_human":
        summary = "Claim rejected by human reviewer."
        explanation = detail or "A human reviewer rejected this claim during review."
    else:
        reject_reason = detail or "unknown"
        summary = f"Claim rejected: {reject_reason}"
        explanation = _REJECT_REASON_MESSAGES.get(
            reject_reason, "Claim could not be processed."
        )

    final_response = {
        "status": "rejected",
        "request_id": request_id,
        "claim": claim_raw,
        "verdict": "REJECTED",
        "confidence": 1.0,
        "confidence_label": "HIGH",
        "summary": summary,
        "explanation": explanation,
        "sources": [],
        "disclosures": [],
        "metadata": {
            # Kept for backward compatibility; only meaningful for a parser
            # reject, where detail is a ParsedClaim.reject_reason value.
            "reject_reason": detail if disposition == "rejected_parser" else None,
            "disposition": disposition,
            "disposition_detail": detail,
        },
    }

    return {
        "final_response": final_response,
        "execution_end_time": datetime.utcnow().isoformat(),
    }


def _generate_error_response(
    request_id: str,
    error_code: str,
    error_message: str
) -> Dict:
    """Generate error response."""
    final_response = {
        "status": "error",
        "request_id": request_id,
        "claim": "",
        "verdict": "ERROR",
        "confidence": 0.0,
        "confidence_label": "ERROR",
        "summary": "An error occurred during verification",
        "explanation": error_message,
        "sources": [],
        "disclosures": [],
        "metadata": {
            "error_code": error_code,
            "error_message": error_message,
        },
    }

    return {
        "final_response": final_response,
        "execution_end_time": datetime.utcnow().isoformat(),
    }


def _build_summary(
    verdict: str,
    confidence: float,
    agent_evidence: Dict,
    parsed_claim
) -> str:
    """Build a summary of the verification result."""
    agent_type = agent_evidence.get("agent", "unknown")

    if verdict == "SUPPORTS":
        return f"Claim is SUPPORTED by {agent_type.upper()} data with {confidence:.0%} confidence."
    elif verdict == "REFUTES":
        return f"Claim is REFUTED by {agent_type.upper()} data with {confidence:.0%} confidence."
    else:
        return f"Insufficient evidence from {agent_type.upper()} to verify this claim."


def _build_explanation(
    verdict: str,
    agent_evidence: Dict,
    state: VerificationState
) -> str:
    """Build a detailed explanation."""
    reasoning = agent_evidence.get("reasoning", "")
    retrieved_value = agent_evidence.get("retrieved_value")
    magnitude_diff = agent_evidence.get("magnitude_difference_percent")
    source_desc = agent_evidence.get("source_description", "")
    parts = []

    if reasoning:
        parts.append(reasoning)

    if retrieved_value is not None or magnitude_diff is not None or source_desc:
        summary_lines = []
        if retrieved_value is not None:
            summary_lines.append(f"- Retrieved value: {_format_number(retrieved_value)}")
        if magnitude_diff is not None:
            summary_lines.append(f"- Difference from claimed value: {magnitude_diff:.1f}%")
        if source_desc:
            summary_lines.append(f"- Data source: {source_desc}")
        parts.append("\n".join(summary_lines))

    return "\n\n".join(parts) if parts else "No detailed explanation available."


def _format_sources(
    agent_evidence: Dict,
    state: Optional[VerificationState] = None,
) -> List[Dict[str, Any]]:
    """Format source citations from agent evidence and RAG/A2A provenance."""
    sources = []

    source_url = agent_evidence.get("source_url")
    source_desc = agent_evidence.get("source_description", "")
    tools_called = agent_evidence.get("tools_called", [])

    if source_url:
        sources.append({
            "type": agent_evidence.get("agent", "unknown"),
            "description": source_desc,
            "url": source_url,
        })

    # Add info about tools used
    if tools_called:
        sources.append({
            "type": "tools",
            "description": f"Tools used: {', '.join(tools_called)}",
        })

    # Add RAG filing text source
    rag_chunks = state.get("rag_chunks_retrieved", []) if state else []
    if rag_chunks:
        sections = sorted(set(c.get("section", "") for c in rag_chunks))
        filings = sorted(set(
            f"{c.get('filing_type', '')} {c.get('period_end', '')}"
            for c in rag_chunks
        ))
        sources.append({
            "type": "rag",
            "description": (
                f"SEC filing narrative text ({len(rag_chunks)} chunks "
                f"from sections: {', '.join(sections)})"
            ),
            "filings": filings,
        })

    # Add A2A corroboration source
    corroboration = state.get("corroboration_result") if state else None
    if corroboration:
        sources.append({
            "type": "a2a",
            "description": (
                f"Cross-source verification via News agent: "
                f"{corroboration.get('news_verdict', 'N/A')} "
                f"(confidence: {corroboration.get('news_confidence', 0):.0%})"
            ),
        })

    return sources


def _format_metadata(state: VerificationState, agent_evidence: Dict) -> Dict[str, Any]:
    """Format metadata for the response."""
    execution_end = datetime.utcnow().isoformat()
    parsed_claim = state.get("parsed_claim")

    operator = getattr(parsed_claim, "operator", None) if parsed_claim else None
    metric = getattr(parsed_claim, "metric", None) if parsed_claim else None

    metadata = {
        "agent": agent_evidence.get("agent"),
        "disposition": state.get("disposition") or "released",
        "disposition_detail": state.get("disposition_detail"),
        "tools_called": agent_evidence.get("tools_called", []),
        "tool_calls_detail": agent_evidence.get("tool_calls_detail", []),
        "execution_time_ms": agent_evidence.get("execution_time_ms", 0),
        "timestamp": execution_end,
        "total_tokens_used": state.get("total_tokens_used", 0),
        "retrieved_value": agent_evidence.get("retrieved_value"),
        "claimed_value": parsed_claim.value if parsed_claim else None,
        "operator": operator,
        "metric": metric,
        "magnitude_difference_percent": agent_evidence.get("magnitude_difference_percent"),
        "source_description": agent_evidence.get("source_description", ""),
    }

    # Always build data source provenance breakdown (XBRL vs RAG vs A2A)
    xbrl_tools = {"get_income_statement", "get_balance_sheet", "get_cash_flow"}
    tools_used = set(agent_evidence.get("tools_called", []))
    rag_chunks = state.get("rag_chunks_retrieved", [])
    corroboration = state.get("corroboration_result")
    data_sources = {}

    if tools_used & xbrl_tools:
        data_sources["xbrl"] = {
            "used": True,
            "tools": sorted(tools_used & xbrl_tools),
        }

    if rag_chunks:
        data_sources["rag"] = {
            "used": True,
            "chunks_retrieved": len(rag_chunks),
            "sections": sorted(set(c.get("section", "") for c in rag_chunks)),
        }

    if corroboration:
        data_sources["a2a"] = {
            "used": True,
            "news_verdict": corroboration.get("news_verdict", ""),
            "news_confidence": corroboration.get("news_confidence", 0),
        }

    metadata["data_sources"] = data_sources

    return metadata


def _format_number(value: float) -> str:
    """Format a number for display."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"
