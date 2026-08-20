"""Tests for domain agent wrapper functions."""

from unittest.mock import patch, MagicMock
from finvet.graph.nodes.domain_agents import (
    _error_evidence,
    run_market_agent,
    run_news_agent,
    run_sec_agent,
)


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

        state = {"request_id": "test_123"}
        result = run_market_agent(state)

        assert result["agent_type"] == "market"
        assert result["agent_evidence"]["verdict"] == "SUPPORTS"

    @patch("finvet.graph.nodes.domain_agents.MarketAgent")
    def test_run_market_agent_failure(self, MockAgent):
        MockAgent.side_effect = RuntimeError("API down")

        state = {"request_id": "test_123"}
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

        state = {"request_id": "test_456"}
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

        state = {"request_id": "test_sec"}
        result = run_sec_agent(state)

        assert "rag_chunks_retrieved" in result
        assert len(result["rag_chunks_retrieved"]) == 1
        assert result["rag_chunks_retrieved"][0]["search_query"] == "revenue Q4"

    @patch("finvet.graph.nodes.domain_agents.SECAgent")
    def test_a2a_provenance_extraction(self, MockAgent):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "verdict": "SUPPORTS",
            "confidence": 0.88,
            "tools_called": ["corroborate_with_news"],
            "agent": "sec",
            "provenance": [
                {
                    "tool": "corroborate_with_news",
                    "args": {"finding": "Revenue increased 5%"},
                    "result": {
                        "success": True,
                        "news_verdict": "CONFIRMED",
                        "news_confidence": 0.85,
                    },
                },
            ],
        }
        MockAgent.return_value = mock_instance

        state = {"request_id": "test_a2a"}
        result = run_sec_agent(state)

        assert "corroboration_result" in result
        assert result["corroboration_result"]["finding"] == "Revenue increased 5%"
        assert result["corroboration_result"]["news_verdict"] == "CONFIRMED"

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
