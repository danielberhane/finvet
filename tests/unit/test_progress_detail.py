"""A progress event carries what the node produced, and nothing else.

The streaming route iterated `for node_name, updates in event.items()`, merged
`updates` into the final result, and yielded only the node's name. So the
pipeline computed the parsed claim, the resolved period, the retrieved value
and the deterministic comparison, and told the client none of it — the UI was
left inferring the claim type from keywords in the raw text and captioning a
spinner with a guess.

The delta is already in hand; the only question is what may leave the server.
Not `updates` itself: `agent_evidence` carries `provenance` and
`tool_calls_detail`, which hold whole filing excerpts, and shipping those to a
browser on every step would be both wasteful and a quiet way to leak retrieved
text into places it was never reviewed for. So the payload is an allow-list per
node — names and counts, never bodies.
"""

import json

import pytest

from finvet.api.routes.verify import _progress_detail


class _Claim:
    claim_type = "sec"
    ticker = "AAPL"
    metric = "revenue"
    operator = "eq"
    value = 391_000_000_000.0
    range_min = None
    range_max = None
    period = "fiscal year 2024"
    reject_reason = None


class _Period:
    start_date = "2023-10-01"
    end_date = "2024-09-28"
    assumptions = ["Interpreted 'fiscal year 2024' as the issuer's fiscal year"]


def _agent_updates(**overrides):
    evidence = {
        "agent": "sec",
        "verdict": "SUPPORTS",
        "confidence": 0.95,
        "retrieved_value": 391_035_000_000.0,
        "magnitude_difference_percent": 0.009,
        "llm_original_verdict": "NOT_ENOUGH_INFO",
        "override_applied": True,
        "limitation": None,
        "temporal_status": "resolved",
        "tools_called": ["get_company_info", "get_income_statement"],
        "trusted_observation": {
            "tool": "get_income_statement", "metric": "revenue",
            "concept": "Revenues", "value": 391_035_000_000.0,
            "units": "USD", "period_end": "2024-09-28", "observed_at": None},
        # The two that must never leave the server.
        "provenance": [{"tool": "search_filing_text", "result": {
            "chunks": [{"chunk_text": "SECRET FILING TEXT " * 40}]}}],
        "tool_calls_detail": [{"tool": "get_income_statement",
                               "result_preview": "PREVIEW BODY " * 40}],
        "reasoning": "a long model explanation " * 30,
    }
    evidence.update(overrides)
    return {"agent_evidence": evidence, "agent_type": "sec"}


class TestEachNodeReportsWhatItProduced:

    def test_the_parser_reports_how_it_read_the_claim(self):
        detail = _progress_detail("claim_parser", {"parsed_claim": _Claim()})

        assert detail["claim_type"] == "sec"
        assert detail["ticker"] == "AAPL"
        assert detail["metric"] == "revenue"
        assert detail["value"] == 391_000_000_000.0
        assert detail["period"] == "fiscal year 2024"

    def test_the_period_resolver_reports_the_window_and_its_assumption(self):
        detail = _progress_detail("period_resolver",
                                  {"canonical_period": _Period()})

        assert detail["start"] == "2023-10-01"
        assert detail["end"] == "2024-09-28"
        assert "fiscal year" in detail["assumption"]

    def test_an_agent_reports_the_comparison_it_made(self):
        """The payoff. These four fields are what make the override visible."""
        detail = _progress_detail("sec_agent", _agent_updates())

        assert detail["retrieved_value"] == 391_035_000_000.0
        assert detail["magnitude_difference_percent"] == 0.009
        assert detail["llm_original_verdict"] == "NOT_ENOUGH_INFO"
        assert detail["override_applied"] is True

    def test_an_agent_reports_the_observation_identity(self):
        detail = _progress_detail("sec_agent", _agent_updates())

        assert detail["concept"] == "Revenues"
        assert detail["period_end"] == "2024-09-28"

    def test_an_agent_reports_tool_names_only(self):
        detail = _progress_detail("sec_agent", _agent_updates())

        assert detail["tools"] == ["get_company_info", "get_income_statement"]

    def test_retrieval_is_reported_as_a_count(self):
        """A count is the useful part; the passages are already going to the
        final response, reviewed, once."""
        detail = _progress_detail(
            "sec_agent",
            dict(_agent_updates(), rag_chunks_retrieved=[{}, {}, {}]))

        assert detail["rag_chunks"] == 3

    def test_a_delegation_reports_its_status(self):
        detail = _progress_detail(
            "news_agent",
            dict(_agent_updates(),
                 corroboration_result={"status": "NO_MATCHING_DISCLOSURE"}))

        assert detail["a2a_status"] == "NO_MATCHING_DISCLOSURE"

    def test_consensus_reports_the_settled_verdict(self):
        detail = _progress_detail("consensus", {
            "verdict": "SUPPORTS", "confidence": 0.95,
            "confidence_label": "HIGH", "consensus_reasons": ["…"]})

        assert detail["verdict"] == "SUPPORTS"
        assert detail["confidence"] == 0.95

    def test_the_guardrails_report_whether_a_person_is_needed(self):
        detail = _progress_detail("output_guardrails", {
            "hitl_required": True, "hitl_triggers": ["low_confidence"]})

        assert detail["hitl_required"] is True
        assert detail["hitl_triggers"] == ["low_confidence"]

    def test_an_unknown_node_reports_nothing(self):
        """A node added later should be silent, not leak its whole delta."""
        assert _progress_detail("some_new_node", _agent_updates()) == {}

    def test_a_node_with_nothing_to_say_reports_nothing(self):
        assert _progress_detail("claim_parser", {}) == {}


class TestNothingUnreviewedLeavesTheServer:
    """The constraint that decides the allow-list."""

    def _serialised(self, node, updates):
        return json.dumps(_progress_detail(node, updates), default=str)

    @pytest.mark.parametrize("node", ["sec_agent", "market_agent", "news_agent"])
    def test_filing_text_never_appears(self, node):
        assert "SECRET FILING TEXT" not in self._serialised(node, _agent_updates())

    @pytest.mark.parametrize("node", ["sec_agent", "market_agent", "news_agent"])
    def test_tool_result_bodies_never_appear(self, node):
        assert "PREVIEW BODY" not in self._serialised(node, _agent_updates())

    def test_the_structural_keys_are_absent_entirely(self):
        detail = _progress_detail("sec_agent", _agent_updates())

        for forbidden in ("provenance", "tool_calls_detail",
                          "rag_chunks_retrieved", "messages", "reasoning"):
            assert forbidden not in detail

    def test_a_long_string_is_truncated(self):
        """Belt and braces: an allow-listed field that turns out to be prose
        must not become a kilobyte on every step."""
        detail = _progress_detail("period_resolver", {
            "canonical_period": type("P", (), {
                "start_date": "2024-01-01", "end_date": "2024-12-31",
                "assumptions": ["x" * 500]})()})

        assert len(detail["assumption"]) <= 210

    def test_the_whole_payload_stays_small(self):
        """It is sent on every node of every run."""
        assert len(self._serialised("sec_agent", _agent_updates())) < 800
