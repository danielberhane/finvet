"""Shared utility functions used across the FinVet pipeline."""

from typing import Any, Dict, Optional

from ..config.constants import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MODERATE_THRESHOLD


def get_confidence_label(confidence: float) -> str:
    """Convert confidence score to human-readable label.

    Thresholds:
        >= 0.85 → HIGH
        >= 0.70 → MODERATE
        <  0.70 → LOW
    """
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    elif confidence >= CONFIDENCE_MODERATE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def build_preliminary_analysis(
    state: dict,
    agent_evidence: dict,
) -> Dict[str, Any]:
    """Build preliminary analysis dict for HITL pending responses.

    Used by both main.py (interrupt path) and response_generator.py (graph path)
    to produce a consistent preliminary_analysis structure.
    """
    parsed_claim = state.get("parsed_claim")
    preliminary_confidence = state.get(
        "confidence", agent_evidence.get("confidence", 0.0)
    )

    return {
        "verdict": state.get(
            "verdict", agent_evidence.get("verdict", "NOT_ENOUGH_INFO")
        ),
        "confidence": preliminary_confidence,
        "confidence_label": get_confidence_label(preliminary_confidence),
        "agent": agent_evidence.get("agent", "unknown"),
        "reasoning": agent_evidence.get("reasoning", ""),
        "retrieved_value": agent_evidence.get("retrieved_value"),
        "claimed_value": parsed_claim.value if parsed_claim else None,
        "magnitude_difference_percent": agent_evidence.get("magnitude_difference_percent"),
        "tools_called": agent_evidence.get("tools_called", []),
        "tool_calls_detail": agent_evidence.get("tool_calls_detail", []),
        "source_description": agent_evidence.get("source_description", ""),
    }
