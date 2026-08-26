"""Macro numeric claims are declined, and the decline actually fires.

`RELEASE_A_DECISIONS.md` D10 says all macro numeric claims fail closed until
their sources have deterministic period-selection and freshness contracts. FRED
series carry an observation date but no contract for *which* observation a
claim means: "inflation was 3.1%" names no vintage, and CPI is revised, so the
same claim is true or false depending on which release you read. Nothing in the
pipeline resolves that, so a decisive verdict on it rests on an unstated choice.

Two things have to hold for the decline to be real, and the second is the one
that was missing:

1. `verification_strategy_for` must classify the metric as unsupported.
2. The route the claim actually travels must *consult* that classification.

`_unsupported_claim` was wired into `run_sec_agent` and `run_market_agent` and
not into `run_news_agent` — the only route macro claims take. A guard that the
relevant path never calls is not a guard, and the whole vocabulary would have
been declared unsupported while every macro claim still reached an agent.
"""

import pytest

from finvet.models.claim import ParsedClaim


def _macro_claim(metric="cpi_inflation", value=3.1):
    return ParsedClaim(claim_type="news", ticker="US", metric=metric,
                       operator="eq", value=value)


class TestTheVocabularyDeclaresMacroUnsupported:

    @pytest.mark.parametrize("metric", [
        "cpi_inflation", "core_pce", "consumer_confidence",
        "earnings_growth_forecast",
    ])
    def test_macro_metrics_are_unsupported(self, metric):
        from finvet.config.metrics import verification_strategy_for

        assert verification_strategy_for(_macro_claim(metric)) == "unsupported"

    def test_the_news_servable_set_is_empty(self):
        from finvet.config.metrics import SERVABLE_METRICS

        assert SERVABLE_METRICS["news"] == frozenset()

    def test_narrative_metrics_still_route_to_news_search(self):
        """`fine_amount` and `settlement_amount` are checked before the
        servable set, so emptying it must not silence them."""
        from finvet.config.metrics import verification_strategy_for

        for metric in ("fine_amount", "settlement_amount"):
            assert verification_strategy_for(_macro_claim(metric)) == "news_search"

    def test_a_narrative_claim_with_no_metric_still_routes(self):
        claim = ParsedClaim(claim_type="news", ticker="MSFT", metric=None,
                            operator=None, value=None)
        from finvet.config.metrics import verification_strategy_for

        assert verification_strategy_for(claim) == "news_search"

    def test_sec_and_market_vocabularies_are_untouched(self):
        """Emptying one claim type's set must not empty the others."""
        from finvet.config.metrics import SERVABLE_METRICS, verification_strategy_for

        assert SERVABLE_METRICS["sec"] and SERVABLE_METRICS["market"]
        sec = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                          operator="eq", value=1.0)
        market = ParsedClaim(claim_type="market", ticker="AAPL",
                             metric="closing_price", operator="eq", value=1.0)
        assert verification_strategy_for(sec) == "xbrl"
        assert verification_strategy_for(market) == "market"


class TestTheNewsRouteConsultsTheClassification:
    """The gap that made the classification inert."""

    def _run_news(self, claim):
        from unittest.mock import patch

        from finvet.graph.nodes import domain_agents

        ran = {"agent": False}

        def _fake_run_agent(cls, kind, label, state, **kwargs):
            ran["agent"] = True
            return {"agent_evidence": {"agent": "news", "verdict": "SUPPORTS",
                                       "confidence": 0.9, "provenance": [],
                                       "tools_called": []},
                    "agent_type": "news"}

        with patch.object(domain_agents, "_run_agent", _fake_run_agent):
            out = domain_agents.run_news_agent(
                {"request_id": "t", "claim_raw": "inflation was 3.1%",
                 "parsed_claim": claim})
        return out, ran["agent"]

    def test_a_macro_claim_is_declined_before_an_agent_runs(self):
        out, agent_ran = self._run_news(_macro_claim())

        assert agent_ran is False, (
            "the news route ran an agent for a metric no tool may serve")
        assert out["agent_evidence"]["verdict"] == "NOT_ENOUGH_INFO"

    def test_the_decline_says_why(self):
        out, _ = self._run_news(_macro_claim())
        evidence = out["agent_evidence"]

        assert evidence.get("limitation"), (
            "an unexplained NOT_ENOUGH_INFO is the outcome declining exists "
            "to replace")
        # "completed", not "failed": nothing broke. The system is declining a
        # claim it has no way to answer, which is a different fact from a tool
        # erroring, and the audit trail keeps them apart.
        assert evidence["execution_status"] == "completed"
        assert evidence["error"] is None

    def test_a_narrative_claim_still_reaches_the_agent(self):
        """The control. Without it, this suite passes on a route that declines
        everything."""
        _, agent_ran = self._run_news(_macro_claim("fine_amount", 5e8))
        assert agent_ran is True

    def test_a_claim_with_no_metric_still_reaches_the_agent(self):
        claim = ParsedClaim(claim_type="news", ticker="MSFT", metric=None,
                            operator=None, value=None)
        _, agent_ran = self._run_news(claim)
        assert agent_ran is True


class TestTheMacroToolIsOffTheDefaultToolbox:

    def test_the_news_agent_holds_no_macro_tool(self):
        from finvet.agents.news_agent.react_agent import NewsAgent

        agent = NewsAgent.__new__(NewsAgent)
        names = {getattr(t, "name", "") for t in NewsAgent.build_tools(agent)} \
            if hasattr(NewsAgent, "build_tools") else None
        if names is None:
            pytest.skip("NewsAgent exposes no tool-list accessor")
        assert "get_macro_indicator" not in names

    def test_the_module_is_kept_for_future_work(self):
        """Removed from the default toolbox, not deleted."""
        import importlib

        module = importlib.import_module("finvet.tools.macro_tools")
        assert hasattr(module, "get_macro_indicator")
