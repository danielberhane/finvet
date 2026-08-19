"""Data models for verdicts."""

from enum import Enum


class Verdict(str, Enum):
    """Verification verdict."""

    SUPPORTS = "SUPPORTS"  # Claim is supported by evidence
    REFUTES = "REFUTES"  # Claim is refuted by evidence
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"  # Insufficient evidence to determine


class ConfidenceLabel(str, Enum):
    """Human-readable confidence labels."""

    HIGH = "High"  # >= 0.85
    MODERATE = "Moderate"  # 0.70 - 0.84
    LOW = "Low"  # < 0.70
