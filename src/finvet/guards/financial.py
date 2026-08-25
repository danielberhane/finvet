"""Financial output guard — regex patterns for investment advice / regulatory red flags.

Output-only guard. Catches responses that cross the line from factual
verification into personalized financial advice (S6 / SR 26-2 concern).
"""

import re
import time
from typing import List

from .provider import GuardResult

_ADVICE_PATTERNS = [
    r"\byou\s+should\s+(buy|sell|hold)\b",
    r"\bstrong\s+buy\b",
    r"\boutperform\b",
    r"\bprice\s+target\b",
    r"\bguaranteed\s+return\b",
    r"\brisk[- ]free\b",
    r"\binvestment\s+advice\b",
    r"\bfinancial\s+advice\b",
    r"\brecommend\s+(buying|selling|holding)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ADVICE_PATTERNS]


class FinancialGuardProvider:
    """Regex guard for catching investment advice in output text."""

    def classify_input(self, text: str) -> GuardResult:
        """Financial guard is output-only; always returns safe for input."""
        return GuardResult(safe=True, provider="financial")

    def classify_output(self, response_text: str, original_claim: str) -> GuardResult:
        t0 = time.perf_counter()
        matched: List[str] = []
        for pattern in _COMPILED:
            if pattern.search(response_text):
                matched.append(pattern.pattern)

        if matched:
            return GuardResult(
                safe=False,
                categories=["S6_financial_advice"],
                violation_type="FINANCIAL_ADVICE_DETECTED",
                provider="financial",
                latency_ms=_ms(t0),
            )
        return GuardResult(safe=True, provider="financial", latency_ms=_ms(t0))


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
