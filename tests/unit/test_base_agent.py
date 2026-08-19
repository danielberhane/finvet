"""Tests for BaseVerificationAgent verdict override and tolerance logic."""

from unittest.mock import MagicMock, patch
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
