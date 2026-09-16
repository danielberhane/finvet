"""Tests for domain agent wrapper functions."""

from unittest.mock import patch, MagicMock
from finvet.graph.nodes import domain_agents
from finvet.models.claim import ParsedClaim
from finvet.graph.nodes.domain_agents import (
    _error_evidence,
    run_market_agent,
    run_news_agent,
    run_sec_agent,
)


def _parsed(claim_type):
    """The minimum the pre-flight accepts: a company, and a servable metric.
    A node never sees a state without a parse, so these tests should not
    either (a missing parse is declined; see
    test_a_missing_parse_is_declined_not_run.py)."""
    return {
        "sec": ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                           operator="eq", value=391e9, period="fiscal 2024"),
        "market": ParsedClaim(claim_type="market", ticker="AAPL",
                              metric="closing_price", operator="eq", value=150.0),
        "news": ParsedClaim(claim_type="news", ticker="AAPL", metric=None),
    }[claim_type]


class TestErrorEvidence:

    def test_structure(self):
        result = _error_evidence("market", "Finnhub", "Connection failed")
        assert result["agent"] == "market"
        assert result["verdict"] == "NOT_ENOUGH_INFO"
        assert result["confidence"] == 0.2
        assert result["source_description"] == "Finnhub"
        assert "Connection failed" in result["reasoning"]
        assert result["tools_called"] == []

    def test_different_agents(self):
        for agent_type, source in [("sec", "SEC EDGAR"), ("news", "Financial News")]:
            result = _error_evidence(agent_type, source, "error")
            assert result["agent"] == agent_type
            assert result["source_description"] == source


class TestRunAgent:

    @patch("finvet.graph.nodes.domain_agents.MarketAgent")
    def test_run_market_agent_success(self, MockAgent):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.90,
            "tools_called": ["get_stock_quote"],
            "agent": "market",
        }
        MockAgent.return_value = mock_instance

        state = {"request_id": "test_123", "parsed_claim": _parsed("market")}
        result = run_market_agent(state)

        assert result["agent_type"] == "market"
        assert result["agent_evidence"]["verdict"] == "SUPPORTS"

    @patch("finvet.graph.nodes.domain_agents.MarketAgent")
    def test_run_market_agent_failure(self, MockAgent):
        MockAgent.side_effect = RuntimeError("API down")

        state = {"request_id": "test_123", "parsed_claim": _parsed("market")}
        result = run_market_agent(state)

        assert result["agent_type"] == "market"
        assert result["agent_evidence"]["verdict"] == "NOT_ENOUGH_INFO"
        assert "API down" in result["agent_evidence"]["reasoning"]

    @patch("finvet.graph.nodes.domain_agents.NewsAgent")
    def test_run_news_agent_success(self, MockAgent):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "REFUTES",
            "confidence": 0.80,
            "tools_called": ["search_financial_news"],
            "agent": "news",
        }
        MockAgent.return_value = mock_instance

        state = {"request_id": "test_456", "parsed_claim": _parsed("news")}
        result = run_news_agent(state)

        assert result["agent_type"] == "news"
        assert result["agent_evidence"]["verdict"] == "REFUTES"


class TestSECProvenance:

    @patch("finvet.graph.nodes.domain_agents.SECAgent")
    def test_rag_provenance_extraction(self, MockAgent):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.92,
            "tools_called": ["get_income_statement", "search_filing_text"],
            "agent": "sec",
            "provenance": [
                {
                    "tool": "search_filing_text",
                    "args": {"query": "revenue Q4"},
                    "result": {
                        "success": True,
                        "chunks": [
                            {"section": "mda", "chunk_text": "Revenue grew..."},
                        ],
                    },
                },
            ],
        }
        MockAgent.return_value = mock_instance

        state = {"request_id": "test_sec", "parsed_claim": _parsed("sec")}
        result = run_sec_agent(state)

        assert "rag_chunks_retrieved" in result
        assert len(result["rag_chunks_retrieved"]) == 1
        assert result["rag_chunks_retrieved"][0]["search_query"] == "revenue Q4"

    @patch("finvet.graph.nodes.domain_agents.SECAgent")
    def test_sec_route_produces_no_corroboration(self, MockAgent):
        """The SEC -> News direction was removed; this node no longer sets it.

        Corroboration now originates from run_news_agent, so a SEC-routed claim
        must not populate corroboration_result even if provenance carried
        something unexpected.
        """
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.88,
            "tools_called": ["search_filing_text"],
            "agent": "sec",
            "provenance": [],
        }
        MockAgent.return_value = mock_instance

        result = run_sec_agent({"request_id": "test_no_a2a"})

        assert "corroboration_result" not in result

    @patch("finvet.graph.nodes.domain_agents.SECAgent")
    def test_failed_provenance_skipped(self, MockAgent):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.90,
            "tools_called": ["search_filing_text"],
            "agent": "sec",
            "provenance": [
                {
                    "tool": "search_filing_text",
                    "args": {"query": "test"},
                    "result": {"success": False, "error": "No chunks found"},
                },
            ],
        }
        MockAgent.return_value = mock_instance

        state = {"request_id": "test_skip"}
        result = run_sec_agent(state)

        assert "rag_chunks_retrieved" not in result


class TestClaimsTheSystemDeclines:
    """Two limitations, named before an agent runs rather than improvised.

    A claim the pipeline cannot serve used to reach an agent anyway, which then
    produced whatever it could from tools that do not carry the answer. Naming
    the limitation up front turns an unexplained NOT_ENOUGH_INFO into a stated
    one, and saves the round trip.
    """

    class _Q4Period:
        fiscal_quarter = "Q4"
        end_date = "2024-09-28"
        start_date = "2024-06-30"

    def _sec_state(self, **claim_kwargs):
        from finvet.models.claim import ParsedClaim
        kwargs = dict(claim_type="sec", ticker="AAPL", metric="revenue",
                      operator="eq", value=100e9)
        kwargs.update(claim_kwargs)
        return {"request_id": "t", "claim_raw": "c",
                "parsed_claim": ParsedClaim(**kwargs)}

    def test_q4_numeric_claim_is_declined_before_the_agent_runs(self):
        state = self._sec_state()
        state["canonical_period"] = self._Q4Period()

        with patch.object(domain_agents, "run_sec_agent_scoped") as scoped:
            out = run_sec_agent(state)

        scoped.assert_not_called()
        evidence = out["agent_evidence"]
        assert evidence["limitation"] == "unsupported_q4_derivation"
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["confidence"] == 0.2
        assert evidence["execution_status"] == "completed"

    def test_q4_claim_without_a_number_still_runs(self):
        """Only numeric Q4 claims need the derivation. A narrative question
        about the quarter is answerable from filing text."""
        state = self._sec_state(metric=None, operator=None, value=None)
        state["canonical_period"] = self._Q4Period()

        with patch.object(domain_agents, "run_sec_agent_scoped",
                          return_value={"agent_evidence": {"verdict": "SUPPORTS"}}) as scoped:
            run_sec_agent(state)

        scoped.assert_called_once()

    def test_non_q4_numeric_claim_still_runs(self):
        class _Annual:
            fiscal_quarter = None
            end_date = "2024-09-28"

        state = self._sec_state()
        state["canonical_period"] = _Annual()

        with patch.object(domain_agents, "run_sec_agent_scoped",
                          return_value={"agent_evidence": {"verdict": "SUPPORTS"}}) as scoped:
            run_sec_agent(state)

        scoped.assert_called_once()

    def test_metric_no_tool_serves_is_declined(self):
        """price_to_book is an official parser example that no Market result
        model exposes. SERVABLE_METRICS knew; nothing consulted it."""
        from finvet.models.claim import ParsedClaim

        state = {"request_id": "t", "claim_raw": "c",
                 "parsed_claim": ParsedClaim(
                     claim_type="market", ticker="GS", metric="price_to_book",
                     operator="eq", value=1.34)}

        with patch.object(domain_agents, "_run_agent") as run:
            out = run_market_agent(state)

        run.assert_not_called()
        assert out["agent_evidence"]["limitation"] == "unsupported_metric"

    def test_servable_market_metric_still_runs(self):
        from finvet.models.claim import ParsedClaim

        state = {"request_id": "t", "claim_raw": "c",
                 "parsed_claim": ParsedClaim(
                     claim_type="market", ticker="TSLA", metric="market_cap",
                     operator="gt", value=8e11)}

        with patch.object(domain_agents, "_run_agent",
                          return_value={"agent_evidence": {}, "agent_type": "market"}) as run:
            run_market_agent(state)

        run.assert_called_once()
