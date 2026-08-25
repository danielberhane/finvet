"""News -> SEC corroboration: recursion safety, delegation fidelity, escalation.

The A2A path was unreachable before this: corroborate_with_news fired 0 times in
496 benchmark runs, because every metric that warrants cross-source checking
(fine_amount, settlement_amount) parses as `news`, and only the SEC agent owned a
corroboration tool. Inverting the direction is the fix; these tests pin the parts
that are easy to get quietly wrong.
"""

from unittest.mock import patch

import pytest

from finvet.agents.base import BaseVerificationAgent
from finvet.agents.news_agent.react_agent import NewsAgent
from finvet.agents.sec_agent.react_agent import SECAgent
from finvet.config.constants import CORROBORATION_METRICS
from finvet.graph.nodes import domain_agents
from finvet.models.a2a import (
    A2A_CONTRADICTS,
    A2A_CORROBORATES,
    A2A_NOT_APPLICABLE_YET,
    A2A_NO_MATCHING_DISCLOSURE,
    A2AResult,
    classify_status,
)
from finvet.models.claim import ParsedClaim
from finvet.tools import corroborate_sec


def _parsed(metric="fine_amount", ticker="AAPL", value=500_000_000.0):
    return ParsedClaim(claim_type="news", ticker=ticker, metric=metric,
                       operator="eq", value=value, period="2025",
                       reject_reason=None)


class TestRecursionIsStructurallyImpossible:
    """News -> SEC -> News must not be constructible, not merely guarded."""

    def test_delegated_sec_agent_has_no_a2a_tool(self):
        agent = SECAgent(allow_a2a=False)
        assert "corroborate_with_news" not in agent.tool_map

    def test_top_level_sec_agent_keeps_it(self):
        assert "corroborate_with_news" in SECAgent().tool_map

    def test_news_agent_can_delegate_to_sec(self):
        assert "corroborate_with_filing" in NewsAgent().tool_map


class TestDelegationCarriesTheClaimedValue:
    """The correctness gap that a text-only delegation would open.

    _apply_override compares parsed_claim.value against the retrieved figure.
    Hand the nested agent only prose and claimed_val is None, magnitude_diff
    stays None, and the deterministic comparison silently does nothing — the one
    part of FinVet that must never be skipped.
    """

    def test_nested_state_has_a_parsed_claim_with_the_value(self):
        captured = {}

        def fake_scoped(state, **kwargs):
            captured["state"] = state
            captured["kwargs"] = kwargs
            return {"agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.9,
                                       "retrieved_value": 500_000_000.0}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            out = corroborate_sec._corroborate(
                finding="Apple was fined EUR 500 million",
                ticker="AAPL", metric="fine_amount",
                claimed_value=500_000_000.0, operator="eq", period="2025",
            )

        pc = captured["state"]["parsed_claim"]
        assert pc is not None, "nested agent received no ParsedClaim"
        assert pc.value == 500_000_000.0, "claimed value did not survive delegation"
        assert pc.ticker == "AAPL"
        assert pc.operator == "eq", "operator must accompany the value"
        # metric cannot ride along: fine_amount is a *news* metric and
        # ParsedClaim rejects it on a sec-typed claim. It is preserved on the
        # result instead, so the audit trail still records what was checked.
        assert pc.metric is None
        assert out.metric == "fine_amount"

    def test_no_claimed_value_means_no_operator(self):
        """ParsedClaim validates the pair; a comparator with nothing to compare
        is rejected, so a claim naming no amount must carry neither."""
        captured = {}

        def fake_scoped(state, **kwargs):
            captured["state"] = state
            return {"agent_evidence": {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.2}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            out = corroborate_sec._corroborate(
                finding="Apple disclosed a regulatory proceeding",
                ticker="AAPL", claimed_value=None)

        assert out.success is True, f"delegation failed: {out.error}"
        pc = captured["state"]["parsed_claim"]
        assert pc.value is None and pc.operator is None

    def test_delegation_disables_a2a_and_shortens_the_budget(self):
        captured = {}

        def fake_scoped(state, **kwargs):
            captured.update(kwargs)
            return {"agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.9}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            corroborate_sec._corroborate(finding="x", ticker="AAPL")

        assert captured["allow_a2a"] is False
        assert captured["max_iterations"] == 3

    def test_result_carries_both_numbers_for_audit(self):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {"verdict": "REFUTES", "confidence": 0.95,
                                       "retrieved_value": 400_000_000.0}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            out = corroborate_sec._corroborate(
                finding="x", ticker="AAPL", claimed_value=500_000_000.0,
                claim_verdict="SUPPORTS")

        assert out.claimed_value == 500_000_000.0
        assert out.retrieved_value == 400_000_000.0


class TestPeriodPropagation:
    """The nested agent must target the same filing the parent would have."""

    def test_delegation_goes_through_the_scoped_runner(self):
        """Not SECAgent.execute directly — that bypasses use_period_target."""
        called = {}

        def fake_scoped(state, **kwargs):
            called["yes"] = True
            return {"agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.9}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            corroborate_sec._corroborate(finding="x", ticker="AAPL", period="2025")

        assert called.get("yes"), "delegation bypassed the period-scoped runner"


class TestTemporalEligibility:
    """Silence from a document written before the event is not evidence."""

    def test_filing_predating_the_event_is_not_applicable(self):
        assert corroborate_sec._filing_could_cover("2025-04-23", "2024-12-31") is False

    def test_filing_covering_the_event_is_applicable(self):
        assert corroborate_sec._filing_could_cover("2024-06-01", "2024-12-31") is True

    def test_unknown_dates_do_not_block_the_attempt(self):
        assert corroborate_sec._filing_could_cover("", "2024-12-31") is True
        assert corroborate_sec._filing_could_cover("2025-04-23", None) is True

    def test_status_is_not_applicable_when_ineligible(self):
        assert classify_status("SUPPORTS", "SUPPORTS", applicable=False) == A2A_NOT_APPLICABLE_YET


class TestStatusTaxonomy:
    """Only genuine contradiction is a conflict."""

    def test_agreement(self):
        assert classify_status("SUPPORTS", "SUPPORTS") == A2A_CORROBORATES
        assert classify_status("REFUTES", "REFUTES") == A2A_CORROBORATES

    def test_contradiction(self):
        assert classify_status("SUPPORTS", "REFUTES") == A2A_CONTRADICTS
        assert classify_status("REFUTES", "SUPPORTS") == A2A_CONTRADICTS

    def test_silence_is_not_contradiction(self):
        assert classify_status("SUPPORTS", "NOT_ENOUGH_INFO") == A2A_NO_MATCHING_DISCLOSURE
        assert classify_status("NOT_ENOUGH_INFO", "SUPPORTS") == A2A_NO_MATCHING_DISCLOSURE


class TestPolicyTrigger:
    """Narrow on purpose: a disclosable metric, a company, and a news verdict."""

    def test_fires_for_a_fine_with_a_ticker(self):
        assert domain_agents._policy_wants_corroboration(
            {"parsed_claim": _parsed()}, {"verdict": "SUPPORTS"}) is True

    def test_scope_is_only_fines_and_settlements(self):
        assert CORROBORATION_METRICS == frozenset({"fine_amount", "settlement_amount"})

    @pytest.mark.parametrize("metric", ["layoffs", "acquisition_value", "cpi_inflation"])
    def test_does_not_fire_for_out_of_scope_metrics(self, metric):
        assert domain_agents._policy_wants_corroboration(
            {"parsed_claim": _parsed(metric=metric)}, {"verdict": "SUPPORTS"}) is False

    def test_does_not_fire_without_a_ticker(self):
        assert domain_agents._policy_wants_corroboration(
            {"parsed_claim": _parsed(ticker=None)}, {"verdict": "SUPPORTS"}) is False

    def test_does_not_fire_without_a_news_verdict(self):
        assert domain_agents._policy_wants_corroboration(
            {"parsed_claim": _parsed()}, {}) is False


class TestPolicyTriggeredProvenancePersists:
    """A tool called after the ReAct loop is not in the message history.

    _extract_tool_info builds provenance from AIMessage/ToolMessage pairs, so the
    policy path must attach its own result explicitly or the evidence vanishes.
    """

    def test_policy_result_reaches_state_though_absent_from_provenance(self):
        agent_evidence = {"verdict": "SUPPORTS", "confidence": 0.9,
                          "provenance": [], "tools_called": ["search_financial_news"]}

        def fake_run_agent(cls, atype, desc, state, **kw):
            return {"agent_evidence": agent_evidence, "agent_type": "news"}

        stub = A2AResult(success=True, direction="news_to_sec", source_agent="news",
                         target_agent="sec", status=A2A_CORROBORATES, verdict="SUPPORTS",
                         confidence=0.9, trigger_mode="policy").model_dump()

        with patch.object(domain_agents, "_run_agent", fake_run_agent), \
             patch.object(corroborate_sec, "_corroborate",
                          lambda **kw: A2AResult(**{**stub, "success": True})):
            out = domain_agents.run_news_agent(
                {"request_id": "t", "claim_raw": "Apple was fined",
                 "parsed_claim": _parsed()})

        assert out.get("corroboration_result") is not None
        assert out["corroboration_result"]["trigger_mode"] == "policy"

    def test_model_triggered_result_is_lifted_from_provenance_not_recalled(self):
        prov_result = A2AResult(
            success=True, direction="news_to_sec", source_agent="news",
            target_agent="sec", status=A2A_CORROBORATES, verdict="SUPPORTS",
            confidence=0.9, trigger_mode="agent").model_dump()
        agent_evidence = {
            "verdict": "SUPPORTS", "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": prov_result}],
        }
        calls = []

        def fake_run_agent(cls, atype, desc, state, **kw):
            return {"agent_evidence": agent_evidence, "agent_type": "news"}

        with patch.object(domain_agents, "_run_agent", fake_run_agent), \
             patch.object(corroborate_sec, "_corroborate",
                          lambda **kw: calls.append(kw)):
            out = domain_agents.run_news_agent(
                {"request_id": "t", "claim_raw": "c", "parsed_claim": _parsed()})

        assert out["corroboration_result"]["trigger_mode"] == "agent"
        assert calls == [], "policy path ran even though the model already delegated"


class TestProvenanceRoundTrip:
    """The evidence must survive LangChain's stringification (see 5110e91)."""

    def test_dict_survives(self):
        payload = A2AResult(success=True, direction="news_to_sec", source_agent="news",
                            target_agent="sec", status=A2A_CORROBORATES,
                            verdict="SUPPORTS", confidence=0.9).model_dump()
        parsed = BaseVerificationAgent._parse_provenance(str(payload))
        assert parsed.get("success") is True
        assert parsed["status"] == A2A_CORROBORATES

    def test_bare_model_does_not(self):
        payload = A2AResult(success=True, direction="news_to_sec", source_agent="news",
                            target_agent="sec", status=A2A_CORROBORATES,
                            verdict="SUPPORTS", confidence=0.9)
        parsed = BaseVerificationAgent._parse_provenance(str(payload))
        assert parsed.get("success") is None
        assert list(parsed) == ["raw"]
