"""Regex-based guard provider for fast input safety checks.

Handles: prompt injection, PII scrubbing, length validation,
text normalization (NFKC, whitespace, curly quotes), and language detection.
"""

import re
import time
import unicodedata
from typing import List, Optional, Tuple

from .provider import GuardResult

# --- Injection patterns (case-insensitive) ---
_INJECTION_PATTERNS = [
    # Tolerate stacked qualifiers between the verb and "instructions"
    # ("ignore all previous instructions", "ignore the above instructions").
    r"ignore\s+(?:\w+\s+){0,4}instructions",
    r"disregard\s+(?:\w+\s+){0,4}(?:rules|instructions)",
    r"you\s+are\s+now",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if",
    r"<\s*script",
    r"SELECT\s+.*FROM",
    r"DROP\s+TABLE",
    r"INSERT\s+INTO",
    r"DELETE\s+FROM",
]

# --- PII patterns ---
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
_CC_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# --- Language detection ---
_ENGLISH_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "revenue", "earnings", "profit", "loss", "stock", "share", "billion", "million",
})

_MIN_LENGTH = 10
_MAX_LENGTH = 2000


class RegexGuardProvider:
    """Fast regex-based guard for input sanitization (<1 ms)."""

    def classify_input(self, text: str) -> GuardResult:
        t0 = time.perf_counter()
        categories: List[str] = []
        flags: List[str] = []

        # 1. Prompt injection
        injection_pattern = _check_injection(text)
        if injection_pattern:
            return GuardResult(
                safe=False,
                categories=["injection"],
                violation_type="INJECTION_DETECTED",
                provider="regex",
                latency_ms=_ms(t0),
            )

        # 2. PII scrubbing
        scrubbed, pii_types = _scrub_pii(text)
        if "sensitive" in pii_types:
            return GuardResult(
                safe=False,
                categories=["pii_sensitive"],
                violation_type="PII_DETECTED",
                provider="regex",
                latency_ms=_ms(t0),
            )
        if pii_types:
            flags.append("pii_redacted")

        # 3. Length check
        length = len(scrubbed)
        if length < _MIN_LENGTH:
            return GuardResult(
                safe=False,
                categories=["too_short"],
                violation_type="CLAIM_TOO_SHORT",
                provider="regex",
                latency_ms=_ms(t0),
            )
        if length > _MAX_LENGTH:
            return GuardResult(
                safe=False,
                categories=["too_long"],
                violation_type="CLAIM_TOO_LONG",
                provider="regex",
                latency_ms=_ms(t0),
            )

        # 4. Normalize
        normalized = _normalize(scrubbed)

        # 5. Language detection
        is_english, lang_conf = _detect_language(normalized)
        if not is_english and lang_conf > 0.8:
            return GuardResult(
                safe=False,
                categories=["unsupported_language"],
                violation_type="UNSUPPORTED_LANGUAGE",
                provider="regex",
                latency_ms=_ms(t0),
            )
        if lang_conf < 0.8:
            flags.append("language_uncertain")

        return GuardResult(
            safe=True,
            categories=categories,
            scrubbed_text=normalized,
            flags=flags,
            provider="regex",
            latency_ms=_ms(t0),
        )

    def classify_output(self, response_text: str, original_claim: str) -> GuardResult:
        """Regex guard is input-only; always returns safe for output."""
        return GuardResult(safe=True, provider="regex")


# --- Helpers ---

def _check_injection(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return pattern
    return None


def _scrub_pii(text: str) -> Tuple[str, List[str]]:
    pii_found: List[str] = []
    scrubbed = text

    if _SSN_PATTERN.search(text):
        pii_found.append("sensitive")
        return "[REDACTED]", pii_found

    if _CC_PATTERN.search(text):
        pii_found.append("sensitive")
        return "[REDACTED]", pii_found

    if _EMAIL_PATTERN.search(text):
        pii_found.append("email")
        scrubbed = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", scrubbed)

    if _PHONE_PATTERN.search(text):
        pii_found.append("phone")
        scrubbed = _PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)

    return scrubbed, pii_found


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char)[0] != "C" or char in "\n\t"
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def _detect_language(text: str) -> Tuple[bool, float]:
    words = text.lower().split()
    if not words:
        return False, 0.0
    english_count = sum(1 for w in words if w in _ENGLISH_WORDS)
    confidence = english_count / len(words)
    is_english = confidence >= 0.3
    return is_english, min(confidence * 2, 1.0)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
