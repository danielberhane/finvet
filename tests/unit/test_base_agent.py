"""Tests for BaseVerificationAgent verdict override and tolerance logic."""

from unittest.mock import MagicMock
from finvet.agents.base import BaseVerificationAgent
from finvet.config.constants import (
    TOLERANCE_DEFAULT,
    TOLERANCE_LARGE_VALUE_THRESHOLD,
    TOLERANCE_MARKET,
    TOLERANCE_NEWS,
    TOLERANCE_SEC_LARGE,
    TOLERANCE_SEC_SMALL,
)


class ConcreteAgent(BaseVerificationAgent):
    """Minimal concrete agent for testing base class methods."""

    def __init__(self, agent_type="sec"):
        # Skip __init__ (avoids LLM creation) and set fields directly
        self.agent_type = agent_type
        self.tools = []
        self.system_prompt = "test"
        self.max_iterations = 5
        self.tool_map = {}

    def _get_source_description(self) -> str:
        return "Test Source"


class TestGetTolerance:

    def test_market_tolerance(self):
        agent = ConcreteAgent(agent_type="market")
        assert agent._get_tolerance(100.0) == TOLERANCE_MARKET

    def test_sec_large_value_tolerance(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(2_000_000_000) == TOLERANCE_SEC_LARGE

    def test_sec_small_value_tolerance(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(500_000_000) == TOLERANCE_SEC_SMALL

    def test_sec_boundary_value(self):
        agent = ConcreteAgent(agent_type="sec")
        # Exactly $1B should use small tolerance (not >=)
        assert agent._get_tolerance(TOLERANCE_LARGE_VALUE_THRESHOLD) == TOLERANCE_SEC_LARGE

    def test_news_tolerance(self):
        agent = ConcreteAgent(agent_type="news")
        assert agent._get_tolerance(50.0) == TOLERANCE_NEWS

    def test_unknown_agent_default(self):
        agent = ConcreteAgent(agent_type="unknown")
        assert agent._get_tolerance(100.0) == TOLERANCE_DEFAULT

    def test_none_value_sec(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(None) == TOLERANCE_SEC_SMALL


class TestBuildContext:

    def test_minimal_context(self):
        agent = ConcreteAgent()
        state = {"claim_raw": "Apple revenue was $94B"}
        context = agent._build_context(state)
        assert "Apple revenue was $94B" in context
        assert "# Your Task" in context

    def test_context_with_parsed_claim(self):
        from finvet.models.claim import ParsedClaim
        agent = ConcreteAgent()
        parsed = ParsedClaim(
            claim_type="sec",
            ticker="AAPL",
            value=94_000_000_000,
            period="FY2024",
        )
        state = {"claim_raw": "test", "parsed_claim": parsed}
        context = agent._build_context(state)
        assert "AAPL" in context
        assert "94,000,000,000" in context
        assert "FY2024" in context

    def test_context_with_memory(self):
        agent = ConcreteAgent()
        state = {
            "claim_raw": "test",
            "memory_context": {
                "claim": "prior claim",
                "verdict": "SUPPORTS",
                "confidence": 0.90,
                "similarity": 0.96,
                "summary": "Previously verified",
            },
        }
        context = agent._build_context(state)
        assert "Prior Verification" in context
        assert "prior claim" in context
        assert "MUST verify independently" in context

    def test_context_with_canonical_period(self):
        from finvet.models.claim import CanonicalPeriod
        agent = ConcreteAgent()
        period = CanonicalPeriod(
            period_type="quarterly",
            start_date="2024-07-01",
            end_date="2024-09-28",
            fiscal_year=2024,
            fiscal_quarter="Q4",
        )
        state = {"claim_raw": "test", "canonical_period": period}
        context = agent._build_context(state)
        assert "quarterly" in context
        assert "2024-07-01" in context
        assert "Q4" in context


class TestEqToleranceCoversRealRoundingSpread:
    """The SEC-large equality tolerance must cover how people actually round.

    Measured across 255 real-sourced eq rows (claimed vs SEC-filed value):
    median 0.004%, p95 1.034%, max 3.067%. The original 1.0% threshold sat
    *below* p95 — true claims like "$185 billion" against a filed
    $182.8B-per-rounding were being REFUTED on rounding alone. 1.5% covers the
    observed p95 with margin while staying far under the 3.067% outlier, so
    genuinely wrong claims still fail.
    """

    def _override(self, claimed, retrieved):
        from finvet.agents.base import VerdictOutput
        agent = ConcreteAgent(agent_type="sec")
        parsed = MagicMock()
        parsed.value = claimed
        parsed.comparison = "eq"
        verdict_output = VerdictOutput(
            verdict="NOT_ENOUGH_INFO", confidence=0.5,
            reasoning="test", retrieved_value=retrieved,
        )
        verdict, confidence, diff = agent._apply_override(
            verdict_output, {"parsed_claim": parsed}, []
        )
        return verdict, diff

    def test_p95_rounding_spread_is_supported(self):
        """1.2% difference on a >$1B claim — inside the measured p95 band."""
        verdict, diff = self._override(185_000_000_000.0, 182_780_000_000.0)
        assert 1.0 < diff < 1.5  # the band the old tolerance wrongly refuted
        assert verdict == "SUPPORTS"

    def test_exact_p95_case_is_supported(self):
        """The measured p95 itself: 1.034% must pass."""
        verdict, diff = self._override(100_000_000_000.0, 98_966_000_000.0)
        assert verdict == "SUPPORTS"

    def test_genuinely_wrong_claim_still_refuted(self):
        """3% off is the outlier region, not rounding — must stay REFUTES."""
        verdict, diff = self._override(100_000_000_000.0, 97_000_000_000.0)
        assert verdict == "REFUTES"

    def test_small_value_tolerance_unchanged(self):
        """Sub-$1B claims keep the 2.0% threshold; only SEC-large moved."""
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(500_000_000) == 2.0
