"""Tests for workflow routing, consensus, and HITL logic."""

from finvet.graph.workflow import (
    _route_after_parsing,
    _simple_consensus,
    _route_after_guardrails,
    _handle_rejection,
    _apply_hitl_decision,
)
from finvet.config.constants import (
    CONSENSUS_CLOSE_MATCH_THRESHOLD,
    CONSENSUS_LARGE_DIFF_THRESHOLD,
    CONSENSUS_MAX_CONFIDENCE,
)
from finvet.models.claim import ParsedClaim


def _make_parsed(claim_type="sec", **kwargs):
    """Helper to create a ParsedClaim for routing tests.

    The contract pairs operator and value (non-null iff non-null), so a test
    that asks for a comparator gets a value alongside it unless it brought
    its own."""
    if claim_type == "reject":
        return ParsedClaim(claim_type="reject", reject_reason="non_financial", **kwargs)
    if kwargs.get("comparison") is not None and "value" not in kwargs:
        kwargs["value"] = 1_000_000_000.0
    return ParsedClaim(claim_type=claim_type, ticker="AAPL", **kwargs)


class TestRouteAfterParsing:

    def test_routes_sec(self):
        state = {"parsed_claim": _make_parsed("sec")}
        assert _route_after_parsing(state) == "sec"

    def test_routes_market(self):
        state = {"parsed_claim": _make_parsed("market")}
        assert _route_after_parsing(state) == "market"

    def test_routes_news(self):
        state = {"parsed_claim": _make_parsed("news")}
        assert _route_after_parsing(state) == "news"

    def test_routes_reject(self):
        state = {"parsed_claim": _make_parsed("reject")}
        assert _route_after_parsing(state) == "reject"

    def test_no_parsed_claim(self):
        assert _route_after_parsing({}) == "reject"


class TestSimpleConsensus:

    def test_no_evidence(self):
        result = _simple_consensus({})
        assert result["verdict"] == "NOT_ENOUGH_INFO"
        assert result["confidence"] == 0.2
        assert result["confidence_label"] == "LOW"

    def test_passes_through_verdict(self):
        state = {
            "agent_evidence": {
                "verdict": "SUPPORTS",
                "confidence": 0.85,
                "tools_called": ["get_income_statement"],
            },
            "parsed_claim": _make_parsed("sec", comparison="eq"),
        }
        result = _simple_consensus(state)
        assert result["verdict"] == "SUPPORTS"

    def test_close_match_bonus(self):
        """Magnitude diff < 2% should get a bonus."""
        state = {
            "agent_evidence": {
                "verdict": "SUPPORTS",
                "confidence": 0.80,
                "magnitude_difference_percent": 1.0,
                "tools_called": [],
            },
            "parsed_claim": _make_parsed("sec", comparison="eq"),
        }
        result = _simple_consensus(state)
        assert result["confidence"] > 0.80

    def test_large_diff_penalty(self):
        """Magnitude diff > 20% should get a penalty."""
        state = {
            "agent_evidence": {
                "verdict": "REFUTES",
                "confidence": 0.80,
                "magnitude_difference_percent": 25.0,
                "tools_called": [],
            },
            "parsed_claim": _make_parsed("sec", comparison="eq"),
        }
        result = _simple_consensus(state)
        assert result["confidence"] < 0.80

    def test_thorough_investigation_bonus(self):
        """Using >= 3 tools should get a bonus."""
        state = {
            "agent_evidence": {
                "verdict": "SUPPORTS",
                "confidence": 0.80,
                "tools_called": ["tool1", "tool2", "tool3"],
            },
            "parsed_claim": _make_parsed("sec"),
        }
        result = _simple_consensus(state)
        assert result["confidence"] > 0.80

    def test_confidence_capped(self):
        """Confidence should never exceed CONSENSUS_MAX_CONFIDENCE."""
        state = {
            "agent_evidence": {
                "verdict": "SUPPORTS",
                "confidence": 0.94,
                "magnitude_difference_percent": 0.5,
                "tools_called": ["t1", "t2", "t3"],
            },
            "parsed_claim": _make_parsed("sec", comparison="eq"),
        }
        result = _simple_consensus(state)
        assert result["confidence"] <= CONSENSUS_MAX_CONFIDENCE

    def test_directional_claim_no_magnitude_adjustment(self):
        """Directional claims (gt) should not get magnitude-based adjustments."""
        state = {
            "agent_evidence": {
                "verdict": "SUPPORTS",
                "confidence": 0.85,
                "magnitude_difference_percent": 50.0,
                "tools_called": [],
            },
            "parsed_claim": _make_parsed("sec", comparison="gt"),
        }
        result = _simple_consensus(state)
        # No penalty applied, confidence unchanged
        assert result["confidence"] == 0.85


class TestRouteAfterGuardrails:

    def test_hitl_required(self):
        assert _route_after_guardrails({"hitl_required": True}) == "needs_hitl"

    def test_no_hitl(self):
        assert _route_after_guardrails({"hitl_required": False}) == "no_hitl"
        assert _route_after_guardrails({}) == "no_hitl"


class TestHandleRejection:

    def test_rejection_response(self):
        result = _handle_rejection({})
        assert result["verdict"] == "REJECTED"
        assert result["confidence"] == 1.0


class TestApplyHITLDecision:

    def test_approve(self):
        state = {
            "request_id": "test",
            "hitl_decision": "approve",
            "verdict": "SUPPORTS",
        }
        result = _apply_hitl_decision(state)
        assert result.get("hitl_applied") is True

    def test_override(self):
        state = {
            "request_id": "test",
            "hitl_decision": "override",
            "hitl_override_verdict": "REFUTES",
            "verdict": "SUPPORTS",
        }
        result = _apply_hitl_decision(state)
        assert result["verdict"] == "REFUTES"
        assert result["confidence"] == 0.95

    def test_reject(self):
        state = {
            "request_id": "test",
            "hitl_decision": "reject",
        }
        result = _apply_hitl_decision(state)
        assert result["verdict"] == "REJECTED"
        assert result["confidence"] == 1.0

    def test_no_decision(self):
        result = _apply_hitl_decision({"request_id": "test"})
        assert result == {}


class TestDisposition:
    """Every terminal path must record why the run ended.

    verdict="REJECTED" has two producers — the parser reject path and a human
    reviewer's reject — so the verdict alone cannot tell an auditor which one
    refused the claim.
    """

    def test_parser_reject_records_disposition(self):
        state = {"parsed_claim": _make_parsed("reject")}
        result = _handle_rejection(state)
        assert result["verdict"] == "REJECTED"
        assert result["disposition"] == "rejected_parser"
        assert result["disposition_detail"] == "non_financial"

    def test_human_reject_records_disposition(self):
        state = {
            "request_id": "test",
            "hitl_decision": "reject",
            "hitl_reviewer_notes": "Agent misread the restatement",
            "verdict": "SUPPORTS",
        }
        result = _apply_hitl_decision(state)
        assert result["verdict"] == "REJECTED"
        assert result["disposition"] == "rejected_human"
        assert result["disposition_detail"] == "Agent misread the restatement"

    def test_reject_dispositions_are_distinguishable(self):
        parser = _handle_rejection({"parsed_claim": _make_parsed("reject")})
        human = _apply_hitl_decision({
            "request_id": "test", "hitl_decision": "reject", "verdict": "SUPPORTS",
        })
        assert parser["verdict"] == human["verdict"] == "REJECTED"
        assert parser["disposition"] != human["disposition"]

    def test_approve_records_disposition(self):
        result = _apply_hitl_decision({
            "request_id": "test", "hitl_decision": "approve", "verdict": "SUPPORTS",
        })
        assert result["disposition"] == "approved_human"

    def test_override_records_disposition(self):
        result = _apply_hitl_decision({
            "request_id": "test",
            "hitl_decision": "override",
            "hitl_override_verdict": "REFUTES",
            "verdict": "SUPPORTS",
        })
        assert result["disposition"] == "overridden_human"
        assert result["verdict"] == "REFUTES"

    def test_no_decision_leaves_state_untouched(self):
        assert _apply_hitl_decision({"request_id": "test", "verdict": "SUPPORTS"}) == {}
