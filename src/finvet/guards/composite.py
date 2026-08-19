"""Composite guard — chains multiple providers with short-circuit logic.

Input classification: runs providers in order, passes scrubbed_text forward,
short-circuits on first failure.

Output classification: same short-circuit on first failure.
"""

import time
from typing import List

from .provider import GuardProvider, GuardResult


class CompositeGuardProvider:
    """Chains guard providers with short-circuit on first failure."""

    def __init__(self, providers: List[GuardProvider]):
        self._providers = providers

    def classify_input(self, text: str) -> GuardResult:
        t0 = time.perf_counter()
        current_text = text
        all_flags: List[str] = []
        all_categories: List[str] = []

        for provider in self._providers:
            result = provider.classify_input(current_text)
            all_flags.extend(result.flags)
            all_categories.extend(result.categories)

            if not result.safe:
                return GuardResult(
                    safe=False,
                    categories=all_categories,
                    scrubbed_text=result.scrubbed_text,
                    flags=all_flags,
                    violation_type=result.violation_type,
                    provider="composite",
                    latency_ms=_ms(t0),
                )
            # Pass scrubbed text to next provider
            if result.scrubbed_text is not None:
                current_text = result.scrubbed_text

        return GuardResult(
            safe=True,
            categories=all_categories,
            scrubbed_text=current_text,
            flags=all_flags,
            provider="composite",
            latency_ms=_ms(t0),
        )

    def classify_output(self, response_text: str, original_claim: str) -> GuardResult:
        t0 = time.perf_counter()
        all_flags: List[str] = []
        all_categories: List[str] = []

        for provider in self._providers:
            result = provider.classify_output(response_text, original_claim)
            all_flags.extend(result.flags)
            all_categories.extend(result.categories)

            if not result.safe:
                return GuardResult(
                    safe=False,
                    categories=all_categories,
                    flags=all_flags,
                    violation_type=result.violation_type,
                    provider="composite",
                    latency_ms=_ms(t0),
                )

        return GuardResult(
            safe=True,
            categories=all_categories,
            flags=all_flags,
            provider="composite",
            latency_ms=_ms(t0),
        )


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
