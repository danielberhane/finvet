"""Tests for the response generator node."""

from finvet.graph.nodes.response_generator import (
    response_generator,
    _build_summary,
    _build_explanation,
    _format_sources,
    _format_metadata,
    _format_number,
)
from finvet.models.claim import ParsedClaim


def _make_parsed(claim_type="sec", value=None, comparison=None):
    if claim_type == "reject":
        return ParsedClaim(claim_type="reject", reject_reason="non_financial")
    return ParsedClaim(
        claim_type=claim_type, ticker="AAPL", value=value,
        comparison=comparison, period="FY2024",
    )


class TestBuildSummary:

    def test_supports_summary(self):
        result = _build_summary("SUPPORTS", 0.90, {"agent": "sec"}, None)
        assert "SUPPORTED" in result
        assert "SEC" in result
        assert "90%" in result

    def test_refutes_summary(self):
        result = _build_summary("REFUTES", 0.85, {"agent": "market"}, None)
        assert "REFUTED" in result
        assert "MARKET" in result

    def test_nei_summary(self):
        result = _build_summary("NOT_ENOUGH_INFO", 0.40, {"agent": "news"}, None)
        assert "Insufficient" in result


class TestBuildExplanation:

    def test_with_reasoning(self):
        evidence = {"reasoning": "Revenue matched within tolerance"}
        result = _build_explanation("SUPPORTS", evidence, {})
        assert "Revenue matched" in result

    def test_with_comparison_details(self):
        parsed = _make_parsed(value=94e9, comparison="eq")
        state = {"parsed_claim": parsed}
        evidence = {
            "reasoning": "test",
            "retrieved_value": 94.2e9,
            "magnitude_difference_percent": 0.21,
            "source_description": "SEC EDGAR",
        }
        result = _build_explanation("SUPPORTS", evidence, state)
        assert "0.21%" in result or "0.2%" in result
        assert "SEC EDGAR" in result

    def test_directional_shows_comparison(self):
        parsed = _make_parsed(value=100.0, comparison="gt")
        state = {"parsed_claim": parsed}
        evidence = {
            "reasoning": "test",
            "retrieved_value": 120.0,
            "magnitude_difference_percent": 20.0,
            "source_description": "Finnhub",
        }
        result = _build_explanation("SUPPORTS", evidence, state)
        assert "20.0%" in result  # magnitude_diff shown for all claims
        assert "Finnhub" in result


class TestFormatSources:

    def test_tool_sources(self):
        evidence = {
            "tools_called": ["get_income_statement", "get_company_info"],
            "source_description": "SEC EDGAR",
        }
        sources = _format_sources(evidence)
        assert any(s["type"] == "tools" for s in sources)

    def test_rag_sources(self):
        evidence = {"tools_called": []}
        state = {
            "rag_chunks_retrieved": [
                {"section": "mda", "filing_type": "10-K", "period_end": "2024-09-28"},
            ],
        }
        sources = _format_sources(evidence, state)
        rag = [s for s in sources if s["type"] == "rag"]
        assert len(rag) == 1
        assert "1 chunks" in rag[0]["description"]

    def test_a2a_sources(self):
        evidence = {"tools_called": []}
        state = {
            "corroboration_result": {
                "news_verdict": "CONFIRMED",
                "news_confidence": 0.85,
            },
        }
        sources = _format_sources(evidence, state)
        a2a = [s for s in sources if s["type"] == "a2a"]
        assert len(a2a) == 1
        assert "CONFIRMED" in a2a[0]["description"]


class TestFormatMetadata:

    def test_data_sources_xbrl(self):
        evidence = {
            "tools_called": ["get_income_statement", "get_company_info"],
            "agent": "sec",
        }
        state = {"parsed_claim": _make_parsed(value=94e9, comparison="eq")}
        metadata = _format_metadata(state, evidence)
        assert "xbrl" in metadata["data_sources"]

    def test_data_sources_rag(self):
        evidence = {"tools_called": [], "agent": "sec"}
        state = {
            "parsed_claim": _make_parsed(),
            "rag_chunks_retrieved": [{"section": "mda"}],
        }
        metadata = _format_metadata(state, evidence)
        assert "rag" in metadata["data_sources"]


class TestFormatNumber:

    def test_billions(self):
        assert _format_number(94_000_000_000) == "$94.00B"

    def test_millions(self):
        assert _format_number(5_500_000) == "$5.50M"

    def test_thousands(self):
        assert _format_number(12_500) == "$12.50K"

    def test_small(self):
        assert _format_number(42.50) == "$42.50"


class TestResponseGenerator:

    def test_rejection_response(self):
        state = {
            "request_id": "test_rej",
            "claim_raw": "What is the price?",
            "parsed_claim": _make_parsed("reject"),
        }
        result = response_generator(state)
        final = result["final_response"]
        assert final["verdict"] == "REJECTED"
        assert final["status"] == "rejected"

    def test_no_evidence_error(self):
        state = {
            "request_id": "test_err",
            "claim_raw": "Apple revenue",
        }
        result = response_generator(state)
        final = result["final_response"]
        assert final["status"] == "error"
        assert final["verdict"] == "ERROR"

    def test_success_response(self):
        state = {
            "request_id": "test_ok",
            "claim_raw": "Apple revenue was $94B",
            "parsed_claim": _make_parsed(value=94e9, comparison="eq"),
            "agent_evidence": {
                "agent": "sec",
                "verdict": "SUPPORTS",
                "confidence": 0.92,
                "retrieved_value": 94.2e9,
                "source_description": "SEC EDGAR",
                "magnitude_difference_percent": 0.21,
                "tools_called": ["get_income_statement"],
                "tool_calls_detail": [],
                "reasoning": "Revenue matches within tolerance",
            },
            "verdict": "SUPPORTS",
            "confidence": 0.92,
            "confidence_label": "HIGH",
        }
        result = response_generator(state)
        final = result["final_response"]
        assert final["status"] == "success"
        assert final["verdict"] == "SUPPORTS"
        assert final["confidence"] == 0.92
