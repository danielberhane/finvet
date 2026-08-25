"""News -> SEC corroboration: recursion safety, delegation fidelity, escalation.

The A2A path was unreachable before this: corroborate_with_news fired 0 times in
496 benchmark runs, because every metric that warrants cross-source checking
(fine_amount, settlement_amount) parses as `news`, and only the SEC agent owned a
corroboration tool. Inverting the direction is the fix, and the old direction was
then removed; these tests pin the parts that are easy to get quietly wrong.
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
    A2A_FAILED,
    A2A_NOT_APPLICABLE_YET,
    A2A_NO_MATCHING_DISCLOSURE,
    A2AResult,
    classify_status,
    reclassify_corroboration,
)
from finvet.models.claim import ParsedClaim
from finvet.tools import corroborate_sec


def _parsed(metric="fine_amount", ticker="AAPL", value=500_000_000.0):
    return ParsedClaim(claim_type="news", ticker=ticker, metric=metric,
                       operator="eq", value=value, period="2025",
                       reject_reason=None)


class TestRecursionIsStructurallyImpossible:
    """News -> SEC terminates because SEC holds no delegation tool at all.

    The SEC -> News direction was removed after measuring 0 invocations in 496
    runs: an audited filing is the strongest source available, so press
    agreement adds nothing to it, and the one case a filing cannot settle -- an
    outcome or subsequent event -- parses as a news claim and never reaches the
    SEC agent. With only one direction there is no cycle to guard.
    """

    def test_sec_agent_holds_no_delegation_tool(self):
        tools = SECAgent().tool_map
        assert not [t for t in tools if t.startswith("corroborate")]

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

    def test_delegation_shortens_the_budget(self):
        captured = {}

        def fake_scoped(state, **kwargs):
            captured.update(kwargs)
            return {"agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.9}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            corroborate_sec._corroborate(finding="x", ticker="AAPL")

        assert captured["max_iterations"] == 3

    def test_result_carries_both_numbers_for_audit(self):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {"verdict": "REFUTES", "confidence": 0.95,
                                       "retrieved_value": 400_000_000.0}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped", fake_scoped):
            out = corroborate_sec._corroborate(
                finding="x", ticker="AAPL", claimed_value=500_000_000.0)

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


class TestParentChildClassification:
    """Status must be computed against the parent verdict, not the child's own.

    corroborate_with_filing has no claim_verdict parameter, so on the
    model-triggered path _corroborate falls back to `claim_verdict or
    target_verdict` and compares the SEC verdict with itself -- which is
    CORROBORATES for any decisive verdict. run_news_agent then lifts that
    status verbatim. A filing that contradicts the news is recorded as
    agreeing with it, and source_disagreement never fires.

    The policy path passes a real claim_verdict and is unaffected; that
    divergence is why one path was correct and the other was not.
    """

    def _news_state(self):
        return {"request_id": "t", "claim_raw": "Issuer was fined",
                "parsed_claim": _parsed()}

    def test_model_triggered_disagreement_is_a_contradiction(self):
        nested = A2AResult(
            success=True, direction="news_to_sec", source_agent="news",
            target_agent="sec", status=A2A_CORROBORATES, verdict="REFUTES",
            confidence=0.95, trigger_mode="agent",
        ).model_dump()
        evidence = {
            "verdict": "SUPPORTS", "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "Issuer was fined"},
                            "result": nested}],
        }
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}):
            out = domain_agents.run_news_agent(self._news_state())

        assert out["corroboration_result"]["status"] == A2A_CONTRADICTS

    def test_model_triggered_agreement_stays_agreement(self):
        nested = A2AResult(
            success=True, direction="news_to_sec", source_agent="news",
            target_agent="sec", status=A2A_CORROBORATES, verdict="SUPPORTS",
            confidence=0.9, trigger_mode="agent",
        ).model_dump()
        evidence = {
            "verdict": "SUPPORTS", "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": nested}],
        }
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}):
            out = domain_agents.run_news_agent(self._news_state())

        assert out["corroboration_result"]["status"] == A2A_CORROBORATES

    def test_filing_silence_is_not_a_contradiction(self):
        nested = A2AResult(
            success=True, direction="news_to_sec", source_agent="news",
            target_agent="sec", status=A2A_CORROBORATES,
            verdict="NOT_ENOUGH_INFO", confidence=0.2, trigger_mode="agent",
        ).model_dump()
        evidence = {
            "verdict": "SUPPORTS", "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": nested}],
        }
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}):
            out = domain_agents.run_news_agent(self._news_state())

        assert out["corroboration_result"]["status"] == A2A_NO_MATCHING_DISCLOSURE


class TestNestedFailureIsNotSilence:
    """A crashed SEC agent and a filing that says nothing are different facts.

    run_sec_agent_scoped catches its own failures and returns NOT_ENOUGH_INFO
    evidence, so the delegation used to record an outage as
    NO_MATCHING_DISCLOSURE -- an authoritative "the filing does not mention
    this" -- when nothing had actually been checked.
    """

    def test_failed_nested_agent_reports_failure(self):
        def failing_scoped(state, **kwargs):
            return {"agent_evidence": {
                "verdict": "NOT_ENOUGH_INFO", "confidence": 0.2,
                "execution_status": "failed", "error": "MCP server unreachable"}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   failing_scoped):
            out = corroborate_sec._corroborate(finding="x", ticker="AAPL")

        assert out.success is False
        assert out.status == A2A_FAILED
        assert "MCP server unreachable" in (out.error or "")

    def test_completed_nested_agent_is_not_a_failure(self):
        def ok_scoped(state, **kwargs):
            return {"agent_evidence": {
                "verdict": "NOT_ENOUGH_INFO", "confidence": 0.3,
                "execution_status": "completed", "error": None}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   ok_scoped):
            out = corroborate_sec._corroborate(finding="x", ticker="AAPL")

        assert out.success is True
        assert out.status != A2A_FAILED

    def test_reclassify_leaves_a_failure_as_failed(self):
        failed = A2AResult(
            success=False, source_agent="news", target_agent="sec",
            status=A2A_NO_MATCHING_DISCLOSURE, verdict="NOT_ENOUGH_INFO",
            error="boom",
        ).model_dump()
        assert reclassify_corroboration("SUPPORTS", failed)["status"] == A2A_FAILED


class TestTemporalScopeIsRecorded:
    """Silence from a filing that closed before the event is not evidence."""

    def test_event_after_the_filing_is_not_applicable(self):
        def fake_scoped(state, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("nested agent ran despite an ineligible filing")

        class _Period:
            end_date = "2024-12-31"

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped), \
             patch("finvet.graph.nodes.period_resolver.period_resolver",
                   lambda s: {"canonical_period": _Period()}):
            out = corroborate_sec._corroborate(
                finding="fined in April 2025", ticker="AAPL",
                event_date="2025-04-23")

        assert out.status == A2A_NOT_APPLICABLE_YET
        assert out.temporal_scope == "checked"

    def test_not_applicable_survives_reclassification(self):
        """A calendar fact does not become a contradiction because the parent
        agent happened to reach a decisive verdict."""
        pending = A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_NOT_APPLICABLE_YET, verdict="NOT_ENOUGH_INFO",
        ).model_dump()
        out = reclassify_corroboration("REFUTES", pending)
        assert out["status"] == A2A_NOT_APPLICABLE_YET

    def test_missing_event_date_is_recorded_as_unknown(self):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.9,
                                       "execution_status": "completed"}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped):
            out = corroborate_sec._corroborate(finding="x", ticker="AAPL")

        assert out.temporal_scope == "unknown"


class TestTriggerParity:
    """Both triggers must reach the same status for the same verdict pair.

    The divergence is what let one path be correct while the other was not:
    the policy path passed a real parent verdict, the model path passed none.
    """

    @pytest.mark.parametrize("parent,target,expected", [
        ("SUPPORTS", "REFUTES", A2A_CONTRADICTS),
        ("REFUTES", "SUPPORTS", A2A_CONTRADICTS),
        ("SUPPORTS", "SUPPORTS", A2A_CORROBORATES),
        ("REFUTES", "REFUTES", A2A_CORROBORATES),
        # target NEI: the filing is silent, which is not disagreement
        ("SUPPORTS", "NOT_ENOUGH_INFO", A2A_NO_MATCHING_DISCLOSURE),
        # parent NEI: the news agent reached no verdict, so the filing has
        # nothing to agree or disagree with -- also not a contradiction
        ("NOT_ENOUGH_INFO", "SUPPORTS", A2A_NO_MATCHING_DISCLOSURE),
        ("NOT_ENOUGH_INFO", "REFUTES", A2A_NO_MATCHING_DISCLOSURE),
        ("NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", A2A_NO_MATCHING_DISCLOSURE),
    ])
    def test_status_matches_across_triggers(self, parent, target, expected):
        results = []
        for mode in ("agent", "policy"):
            payload = A2AResult(
                success=True, source_agent="news", target_agent="sec",
                status=A2A_NO_MATCHING_DISCLOSURE, verdict=target,
                trigger_mode=mode,
            ).model_dump()
            results.append(reclassify_corroboration(parent, payload)["status"])

        assert results == [expected, expected]


class TestOnlyContradictionEscalates:

    @pytest.mark.parametrize("status,should_escalate", [
        (A2A_CONTRADICTS, True),
        (A2A_CORROBORATES, False),
        (A2A_NO_MATCHING_DISCLOSURE, False),
        (A2A_NOT_APPLICABLE_YET, False),
        (A2A_FAILED, False),
    ])
    def test_hitl_trigger(self, status, should_escalate):
        from finvet.graph.nodes.output_guardrails import output_guardrails

        state = {
            "request_id": "t",
            "agent_evidence": {"verdict": "SUPPORTS", "confidence": 0.95,
                               "reasoning": "ok"},
            "verdict": "SUPPORTS",
            "confidence": 0.95,
            "corroboration_result": {"status": status, "verdict": "REFUTES",
                                     "source_agent": "news",
                                     "target_agent": "sec"},
        }
        out = output_guardrails(state)
        triggers = out.get("hitl_triggers", [])
        assert ("source_disagreement" in triggers) is should_escalate


class TestFailedDelegationIsRecorded:
    """A delegation that failed must leave a trace.

    The provenance loop used to skip any result whose success flag was falsy,
    so a failed model-triggered delegation vanished: no corroboration_result,
    no audit record that SEC had even been asked, and the policy path was then
    free to run the whole delegation a second time.
    """

    def _state(self):
        return {"request_id": "t", "claim_raw": "Issuer was fined",
                "parsed_claim": _parsed()}

    def test_failed_model_triggered_delegation_reaches_state(self):
        failed = A2AResult(
            success=False, source_agent="news", target_agent="sec",
            status=A2A_NO_MATCHING_DISCLOSURE, verdict="NOT_ENOUGH_INFO",
            trigger_mode="agent", error="MCP server unreachable",
        ).model_dump()
        evidence = {
            "verdict": "SUPPORTS", "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": failed}],
        }
        calls = []
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}), \
             patch.object(corroborate_sec, "_corroborate",
                          lambda **kw: calls.append(kw)):
            out = domain_agents.run_news_agent(self._state())

        assert out["corroboration_result"]["status"] == A2A_FAILED
        assert "MCP server unreachable" in out["corroboration_result"]["error"]
        assert calls == [], "policy path re-ran a delegation that already failed"
