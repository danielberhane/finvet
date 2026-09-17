"""A market claim with no metric is declined, not run.

The parser prompt teaches "Amazon's stock has doubled since 2020" as
`market / metric null / value null`, and `verification_strategy_for` sent
every null-metric claim that was not `sec` down the "news_search" branch --
a branch written for the two agents that read prose. The market agent has no
prose to read and no quote to fetch for a claim that names no metric, so it
ran, spent its budget, and ended in NOT_ENOUGH_INFO. Now the strategy says
"unsupported" and the pre-flight declines it at no cost.
"""

import pytest

from finvet.config.metrics import verification_strategy_for
from finvet.graph.nodes import domain_agents
from finvet.graph.nodes.domain_agents import run_market_agent
from finvet.models.claim import ParsedClaim


class TestTheStrategy:

    def test_market_with_no_metric_is_unsupported(self):
        claim = ParsedClaim(claim_type="market", ticker="AMZN", metric=None,
                            period="2020")
        assert verification_strategy_for(claim) == "unsupported"

    def test_sec_with_no_metric_still_reads_filing_text(self):
        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric=None)
        assert verification_strategy_for(claim) == "filing_rag"

    def test_news_with_no_metric_still_searches(self):
        claim = ParsedClaim(claim_type="news", ticker="MSFT", metric=None)
        assert verification_strategy_for(claim) == "news_search"

    def test_market_with_a_servable_metric_is_unchanged(self):
        claim = ParsedClaim(claim_type="market", ticker="AAPL",
                            metric="closing_price", operator="eq", value=150.0)
        assert verification_strategy_for(claim) == "market"


class TestTheNodeDeclines:

    def test_no_agent_runs(self, monkeypatch):
        def _refuse(*a, **k):
            raise AssertionError("the market agent ran for a claim with no metric")
        monkeypatch.setattr(domain_agents, "_run_agent", _refuse)

        out = run_market_agent({
            "request_id": "r",
            "parsed_claim": ParsedClaim(claim_type="market", ticker="AMZN",
                                        metric=None, period="2020"),
            "claim_raw": "Amazon's stock has doubled since 2020"})

        evidence = out["agent_evidence"]
        assert evidence["limitation"] == "unsupported_metric"
        assert evidence["tools_called"] == []
        assert evidence["agent"] == "market"

    def test_a_stated_reason_keeps_it_out_of_the_review_queue(self, monkeypatch):
        from finvet.graph.nodes.output_guardrails import output_guardrails
        monkeypatch.setattr(domain_agents, "_run_agent",
                            lambda *a, **k: pytest.fail("agent ran"))

        declined = run_market_agent({
            "request_id": "r",
            "parsed_claim": ParsedClaim(claim_type="market", ticker="AMZN",
                                        metric=None, period="2020"),
            "claim_raw": "Amazon's stock has doubled since 2020"})["agent_evidence"]
        out = output_guardrails({
            "request_id": "r", "claim_raw": "Amazon's stock has doubled since 2020",
            "verdict": declined["verdict"], "confidence": declined["confidence"],
            "agent_evidence": declined})

        assert "low_confidence" not in (out.get("hitl_triggers") or [])
