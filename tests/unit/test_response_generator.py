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


def _make_parsed(claim_type="sec", value=None, operator=None):
    if claim_type == "reject":
        return ParsedClaim(claim_type="reject", reject_reason="non_financial")
    return ParsedClaim(
        claim_type=claim_type, ticker="AAPL", value=value,
        operator=operator, period="FY2024",
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
        parsed = _make_parsed(value=94e9, operator="eq")
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
        parsed = _make_parsed(value=100.0, operator="gt")
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
        state = {"parsed_claim": _make_parsed(value=94e9, operator="eq")}
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

    def test_override_reaches_the_response(self):
        """The README's headline claim is only true if both verdicts ship."""
        evidence = {
            "tools_called": [],
            "agent": "sec",
            "verdict": "REFUTES",
            "override_applied": True,
            "llm_original_verdict": "SUPPORTS",
        }
        state = {"parsed_claim": _make_parsed()}
        metadata = _format_metadata(state, evidence)
        assert metadata["override_applied"] is True
        assert metadata["llm_original_verdict"] == "SUPPORTS"

    def test_no_override_still_reports_the_fields(self):
        evidence = {"tools_called": [], "agent": "sec", "verdict": "SUPPORTS"}
        state = {"parsed_claim": _make_parsed()}
        metadata = _format_metadata(state, evidence)
        assert metadata["override_applied"] is False
        assert metadata["llm_original_verdict"] is None


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
            "parsed_claim": _make_parsed(value=94e9, operator="eq"),
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


class TestRejectDisposition:
    """A human reject must produce a rejection response, not a success one.

    Regression: a human reject has hitl_decision="reject" (so the pending branch
    is skipped) and a claim_type of "sec"/"market"/"news" (so the old
    claim_type == "reject" check was skipped). It fell into the success branch
    and stored status="success" with the summary "Insufficient evidence from SEC
    to verify this claim." while the audit verdict column read REJECTED.
    """

    def _human_reject_state(self):
        return {
            "request_id": "test_hr",
            "claim_raw": "Apple Q4 2024 revenue was $94B",
            "parsed_claim": _make_parsed("sec", value=94e9, operator="eq"),
            "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS", "reasoning": "..."},
            "verdict": "REJECTED",
            "confidence": 1.0,
            "hitl_required": True,
            "hitl_decision": "reject",
            "disposition": "rejected_human",
            "disposition_detail": "Agent misread the restatement",
        }

    def test_human_reject_is_not_a_success_response(self):
        final = response_generator(self._human_reject_state())["final_response"]
        assert final["status"] == "rejected"
        assert final["verdict"] == "REJECTED"
        assert "Insufficient evidence" not in final["summary"]

    def test_human_reject_surfaces_reviewer_reasoning(self):
        final = response_generator(self._human_reject_state())["final_response"]
        assert "human reviewer" in final["summary"].lower()
        assert final["explanation"] == "Agent misread the restatement"
        assert final["metadata"]["disposition"] == "rejected_human"

    def test_parser_reject_keeps_its_reason(self):
        final = response_generator({
            "request_id": "test_pr",
            "claim_raw": "What is the price?",
            "parsed_claim": _make_parsed("reject"),
            "disposition": "rejected_parser",
            "disposition_detail": "non_financial",
        })["final_response"]
        assert final["status"] == "rejected"
        assert final["metadata"]["disposition"] == "rejected_parser"
        assert "not a financial claim" in final["explanation"].lower()

    def test_parser_reject_without_disposition_still_works(self):
        """States predating disposition (older checkpoints) must not regress."""
        final = response_generator({
            "request_id": "test_legacy",
            "claim_raw": "What is the price?",
            "parsed_claim": _make_parsed("reject"),
        })["final_response"]
        assert final["status"] == "rejected"
        assert final["verdict"] == "REJECTED"
        assert final["metadata"]["disposition"] == "rejected_parser"

    def test_advice_seeking_reject_explains_no_advice(self):
        """Advice-seeking now reaches the parser (not an HTTP 400) and returns an
        auditable rejection that says FinVet does not give advice."""
        final = response_generator({
            "request_id": "test_advice",
            "claim_raw": "Should I buy Apple stock right now?",
            "parsed_claim": _make_parsed("reject"),
            "disposition": "rejected_parser",
            "disposition_detail": "advice_seeking",
        })["final_response"]
        assert final["status"] == "rejected"
        assert "advice" in final["explanation"].lower()

    def test_released_response_is_labelled_released(self):
        final = response_generator({
            "request_id": "test_ok",
            "claim_raw": "Apple revenue was $94B",
            "parsed_claim": _make_parsed("sec", value=94e9, operator="eq"),
            "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS", "reasoning": "ok"},
            "verdict": "SUPPORTS",
            "confidence": 0.9,
        })["final_response"]
        assert final["status"] == "success"
        assert final["metadata"]["disposition"] == "released"

    def test_pending_review_is_labelled_pending(self):
        final = response_generator({
            "request_id": "test_pend",
            "claim_raw": "Apple revenue was $94B",
            "parsed_claim": _make_parsed("sec", value=94e9, operator="eq"),
            "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS", "reasoning": "?"},
            "verdict": "SUPPORTS",
            "confidence": 0.4,
            "hitl_required": True,
            "hitl_triggers": ["low_confidence"],
        })["final_response"]
        assert final["status"] == "pending_review"
        assert final["metadata"]["disposition"] == "pending_review"


class TestMetadataEmitsTheContract:
    """CONTRACT (2026-08-20): the payload carries operator and metric only.
    The comparison key is gone — the dual-key window served the 115 historical
    audit rows through the migration; rows from today onward key on operator,
    and the cutover date lives in the commit that removed it."""

    def _metadata(self, **claim_kwargs):
        from finvet.models.claim import ParsedClaim
        state = {
            "request_id": "t", "claim_raw": "c", "verdict": "SUPPORTS",
            "confidence": 0.9,
            "parsed_claim": ParsedClaim(claim_type="sec", ticker="AAPL",
                                        **claim_kwargs),
            "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS",
                               "confidence": 0.9, "reasoning": "r",
                               "tools_called": [], "tool_calls_detail": [],
                               "execution_time_ms": 1},
        }
        return response_generator(state)["final_response"]["metadata"]

    def test_operator_present_and_comparison_gone(self):
        meta = self._metadata(value=1e9, operator="approx")
        assert meta["operator"] == "approx"
        assert "comparison" not in meta

    def test_metric_is_in_the_payload(self):
        meta = self._metadata(metric="revenue")
        assert meta["metric"] == "revenue"

    def test_absent_metric_is_an_explicit_null(self):
        meta = self._metadata()
        assert "metric" in meta and meta["metric"] is None
