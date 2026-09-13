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
from finvet.config.constants import (A2A_MAX_ITERATIONS, AGENT_MAX_ITERATIONS,
                                     CORROBORATION_METRICS)
from finvet.graph.nodes import domain_agents
from finvet.models.a2a import (
    A2A_CONTRADICTS,
    A2A_CORROBORATES,
    A2A_FAILED,
    A2A_NOT_APPLICABLE_YET,
    A2A_NO_CORPUS,
    A2A_NO_MATCHING_DISCLOSURE,
    A2A_PENDING_CLASSIFICATION,
    A2A_SOURCE_UNAVAILABLE,
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


def _searched_and_found_nothing():
    """Provenance for a filing that was actually read and did not mention it.

    Silence is only reportable when a search succeeded, so a fixture standing
    for "the filing is silent" has to carry the search that established it.
    """
    return [{"tool": "search_filing_text",
             "args": {"query": "fine", "ticker": "AAPL"},
             "result": {"success": True, "chunks": [], "total_found": 0,
                        "reason": "no_relevant_evidence"}}]


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

        # The property is the relationship, not the literal. This asserted 3
        # and broke when both budgets rose together, which is the one change
        # it should have tolerated.
        assert captured["max_iterations"] == A2A_MAX_ITERATIONS
        assert A2A_MAX_ITERATIONS < AGENT_MAX_ITERATIONS, (
            "a delegated run spends the caller's budget, so it must be shorter")

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
            provenance=_searched_and_found_nothing(),
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
        assert out.temporal_scope == "event_date"

    def test_not_applicable_survives_reclassification(self):
        """A calendar fact does not become a contradiction because the parent
        agent happened to reach a decisive verdict."""
        pending = A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_NOT_APPLICABLE_YET, verdict="NOT_ENOUGH_INFO",
        ).model_dump()
        out = reclassify_corroboration("REFUTES", pending)
        assert out["status"] == A2A_NOT_APPLICABLE_YET

    def test_undatable_period_is_recorded_as_unknown(self):
        """No explicit date and nothing datable to infer from.

        period_resolver defaults an undated claim to period_type="current",
        whose start is today -- a placeholder, not the event's date. Inferring
        from it would date every undated claim to now.
        """
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
                status=A2A_PENDING_CLASSIFICATION, verdict=target,
                trigger_mode=mode,
                provenance=_searched_and_found_nothing(),
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


class TestSilenceAboutAMaterialAmountIsStillOnlySilence:
    """The materiality promotion is gone, and this is what replaced it.

    A previous release promoted filing silence about a fine or settlement to
    UNDISCLOSED_MATERIAL_CLAIM and escalated it. Reaching that status meant
    deciding the issuer *should* have disclosed the amount -- a materiality
    judgment, made by testing a metric name against a set, with nothing
    calibrating it and no threshold anyone had measured. Release A declines to
    make that judgment, so the status does not exist rather than sitting unused.

    What remains is honest and narrower: when an applicable filing was actually
    searched and said nothing, that is NO_MATCHING_DISCLOSURE, and it does not
    escalate. A periodic report omits most things.
    """

    def _result(self, *, metric="fine_amount", claimed_value=5e8,
                temporal_scope="claim_period", verdict="NOT_ENOUGH_INFO",
                searched=True):
        """A delegation result as _corroborate produces it.

        `searched` controls the filing-search provenance, which is what now
        licenses the silence reading.
        """
        provenance = []
        if searched:
            provenance = [{"tool": "search_filing_text",
                           "args": {"query": "fine", "ticker": "AAPL"},
                           "result": {"success": True, "chunks": [],
                                      "total_found": 0,
                                      "reason": "no_relevant_evidence"}}]
        return A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_PENDING_CLASSIFICATION, verdict=verdict,
            metric=metric, claimed_value=claimed_value,
            temporal_scope=temporal_scope, provenance=provenance,
        ).model_dump()

    def test_silence_on_a_material_amount_stays_silence(self):
        out = reclassify_corroboration("NOT_ENOUGH_INFO", self._result())
        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    @pytest.mark.parametrize("parent", ["NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES"])
    def test_no_parent_verdict_promotes_silence(self, parent):
        out = reclassify_corroboration(parent, self._result())
        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    @pytest.mark.parametrize("scope", ["event_date", "claim_period", "unknown"])
    def test_dating_the_event_does_not_change_the_status(self, scope):
        """The date used to be the gate on promotion. With no promotion, it no
        longer decides anything about the status."""
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result(temporal_scope=scope))
        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    @pytest.mark.parametrize("metric", ["fine_amount", "settlement_amount",
                                        "layoffs", "acquisition_value", "revenue"])
    def test_the_metric_no_longer_selects_a_status(self, metric):
        """Membership of CORROBORATION_METRICS decided materiality before.
        Nothing about the metric name should change what the filing said."""
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result(metric=metric))
        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    def test_a_claim_naming_no_amount_is_unchanged(self):
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result(claimed_value=None))
        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    def test_a_filing_that_confirms_still_corroborates(self):
        out = reclassify_corroboration("SUPPORTS",
                                       self._result(verdict="SUPPORTS"))
        assert out["status"] == A2A_CORROBORATES

    def test_not_applicable_yet_survives_reclassification(self):
        pending = A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_NOT_APPLICABLE_YET, verdict="NOT_ENOUGH_INFO",
            metric="fine_amount", claimed_value=5e8,
            temporal_scope="claim_period",
        ).model_dump()
        assert reclassify_corroboration("NOT_ENOUGH_INFO", pending)["status"] \
            == A2A_NOT_APPLICABLE_YET

    def test_silence_without_a_successful_search_is_not_silence_at_all(self):
        """The distinction that replaced the materiality judgment: a claim
        about what a filing says requires having read one."""
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result(searched=False))
        assert out["status"] == A2A_SOURCE_UNAVAILABLE

    def test_silence_does_not_reach_hitl(self):
        """End to end through the real guardrail node. This assertion is the
        inverse of the one it replaces, and deliberately so."""
        from finvet.graph.nodes.output_guardrails import output_guardrails

        state = {
            "request_id": "t",
            "agent_evidence": {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.9,
                               "reasoning": "could not verify the amount"},
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.9,
            "corroboration_result": reclassify_corroboration(
                "NOT_ENOUGH_INFO", self._result()),
        }
        out = output_guardrails(state)
        triggers = out.get("hitl_triggers", []) or []
        assert "unsupported_material_claim" not in triggers
        assert "source_disagreement" not in triggers



class TestThePolicyPathRecordsWhatActuallyHappened:
    """Driven through run_news_agent, not a hand-built result.

    Only the SEC route runs period_resolver (workflow.py:94), so a news claim
    arrives with no canonical_period and the policy path has no event date to
    pass. An earlier feature depended on that date and could therefore never
    fire on the one path that produces these results, while unit tests holding
    hand-built temporal_scope passed happily. The lesson outlived the feature:
    these drive the real node.
    """

    def _news_state(self, period="2024"):
        return {
            "request_id": "t",
            "claim_raw": "Apple was fined EUR 500 million in 2024",
            "parsed_claim": ParsedClaim(
                claim_type="news", ticker="AAPL", metric="fine_amount",
                operator="eq", value=5e8, period=period, reject_reason=None),
        }

    def _run(self, state, sec_verdict="NOT_ENOUGH_INFO"):
        news_ev = {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.4,
                   "tools_called": ["search_financial_news"], "provenance": [],
                   "execution_status": "completed"}

        def fake_scoped(s, **kw):
            return {"agent_evidence": {
                "verdict": sec_verdict, "confidence": 0.3, "provenance": [],
                "tools_called": ["search_filing_text"],
                "execution_status": "completed"}}

        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": news_ev,
                                                   "agent_type": "news"}), \
             patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped):
            return domain_agents.run_news_agent(state)

    def test_news_route_has_no_canonical_period(self):
        """The precondition that broke it, pinned so it stays visible."""
        assert "canonical_period" not in self._news_state()
        assert domain_agents._event_date_for(self._news_state()) == ""

    def test_the_date_is_still_derived_from_the_resolved_period(self):
        """The temporal scope remains recorded even though no status now
        depends on it: an escalation sends work to a person, and they should
        be able to see whether the date was stated or inferred."""
        out = self._run(self._news_state())
        assert out["corroboration_result"]["temporal_scope"] == "claim_period"

    def test_naming_the_tool_is_not_proof_the_search_succeeded(self):
        """The nested agent reports `tools_called: ["search_filing_text"]` and
        empty provenance. A tool name is an intention; provenance is a result,
        and only a result can license a claim about what the filing says."""
        out = self._run(self._news_state())
        result = out["corroboration_result"]
        assert result["tools_used"] == [] or "search_filing_text" not in (
            result.get("provenance") or [])
        assert result["status"] == A2A_SOURCE_UNAVAILABLE

    def test_that_result_does_not_reach_hitl(self):
        from finvet.graph.nodes.output_guardrails import output_guardrails

        out = self._run(self._news_state())
        guard = output_guardrails({
            "request_id": "t",
            "agent_evidence": {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.9,
                               "reasoning": "could not verify the amount"},
            "verdict": "NOT_ENOUGH_INFO", "confidence": 0.9,
            "corroboration_result": out["corroboration_result"],
        })
        triggers = guard.get("hitl_triggers", []) or []
        assert "unsupported_material_claim" not in triggers
        assert "source_disagreement" not in triggers

    def test_claim_with_no_period_stays_unknown(self):
        """Nothing to infer a date from; the scope says so rather than
        implying the gate ran."""
        out = self._run(self._news_state(period=None))
        assert out["corroboration_result"]["temporal_scope"] == "unknown"


class TestClassificationThroughTheRealTool:
    """Drives the decorated tool, not a hand-built A2AResult.

    The classification tests above construct the nested result themselves,
    which asserts the shape the author remembered rather than the shape
    corroborate_with_filing emits. This invokes the real tool through
    LangChain, puts its actual return value into ReAct provenance, and runs
    run_news_agent over it.
    """

    def _news_state(self):
        return {"request_id": "t", "claim_raw": "Apple was fined EUR 500 million",
                "parsed_claim": _parsed()}

    def _tool_result(self, sec_verdict, execution_status="completed"):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {
                "verdict": sec_verdict, "confidence": 0.95,
                "retrieved_value": 4e8, "provenance": [], "tools_called": [],
                "execution_status": execution_status,
                "error": None if execution_status == "completed" else "sec down",
            }}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped):
            return corroborate_sec.corroborate_with_filing.invoke({
                "finding": "Apple was fined EUR 500 million",
                "ticker": "AAPL", "metric": "fine_amount",
                "claimed_value": 5e8, "operator": "eq", "period": "2025",
            })

    def _run_news(self, tool_result, parent_verdict):
        evidence = {
            "verdict": parent_verdict, "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "Apple was fined EUR 500 million"},
                            "result": tool_result}],
        }
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}):
            return domain_agents.run_news_agent(self._news_state())

    def test_real_tool_output_is_a_dict_that_survives_provenance(self):
        """If it were a bare model, _parse_provenance would drop it."""
        result = self._tool_result("REFUTES")
        assert isinstance(result, dict)
        assert result["verdict"] == "REFUTES"

    def test_parent_supports_nested_refutes_is_a_contradiction(self):
        out = self._run_news(self._tool_result("REFUTES"), "SUPPORTS")
        assert out["corroboration_result"]["status"] == A2A_CONTRADICTS

    def test_parent_supports_nested_supports_corroborates(self):
        out = self._run_news(self._tool_result("SUPPORTS"), "SUPPORTS")
        assert out["corroboration_result"]["status"] == A2A_CORROBORATES

    def test_nested_failure_is_failed_not_silence(self):
        """Spec invariant 3: a nested-agent failure is FAILED, never
        NO_MATCHING_DISCLOSURE."""
        out = self._run_news(
            self._tool_result("NOT_ENOUGH_INFO", execution_status="failed"),
            "SUPPORTS")
        assert out["corroboration_result"]["status"] == A2A_FAILED

    def test_a_completed_nei_delegation_is_not_labelled_undisclosed(self):
        """Spec invariant 5: filing silence may not be claimed unless an
        identified, applicable filing was successfully searched. A delegation
        that merely completed with NEI does not establish that."""
        out = self._run_news(self._tool_result("NOT_ENOUGH_INFO"), "SUPPORTS")
        status = out["corroboration_result"]["status"]
        assert status != "UNDISCLOSED_MATERIAL_CLAIM"
        assert status == A2A_SOURCE_UNAVAILABLE


class TestSourceAvailabilityIsEstablishedBeforeSilenceIsClaimed:
    """Filing silence may only be reported when a filing was actually read.

    `classify_status` maps any non-decisive pair to NO_MATCHING_DISCLOSURE,
    which reads as "the issuer's filing does not mention this". That sentence
    is only true if an applicable filing was identified and successfully
    searched. A delegation that completed with NOT_ENOUGH_INFO because the RAG
    service was down, or because nothing was ever searched, establishes nothing
    about what the filing says -- and reporting it as silence turns an absence
    of evidence into evidence of absence.

    Drives the real decorated tool, so what is classified is what
    `corroborate_with_filing` actually emits.
    """

    def _news_state(self):
        return {"request_id": "t", "claim_raw": "Apple was fined EUR 500 million",
                "parsed_claim": _parsed()}

    def _search(self, **overrides):
        """One `search_filing_text` provenance entry, as the tool records it."""
        result = {"success": True, "chunks": [], "total_found": 0,
                  "reason": "no_relevant_evidence", "error": None}
        result.update(overrides)
        return {"tool": "search_filing_text",
                "args": {"query": "fine", "ticker": "AAPL"},
                "result": result}

    def _tool_result(self, sec_verdict, *, provenance=None,
                     execution_status="completed"):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {
                "verdict": sec_verdict, "confidence": 0.95,
                "retrieved_value": 4e8,
                "provenance": provenance if provenance is not None else [],
                "tools_called": [], "execution_status": execution_status,
                "error": None if execution_status == "completed" else "sec down",
            }}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped):
            return corroborate_sec.corroborate_with_filing.invoke({
                "finding": "Apple was fined EUR 500 million",
                "ticker": "AAPL", "metric": "fine_amount",
                "claimed_value": 5e8, "operator": "eq", "period": "2025",
            })

    def _status(self, tool_result, parent_verdict="SUPPORTS"):
        evidence = {
            "verdict": parent_verdict, "confidence": 0.9,
            "tools_called": ["corroborate_with_filing"],
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": tool_result}],
        }
        with patch.object(domain_agents, "_run_agent",
                          lambda c, a, d, s, **k: {"agent_evidence": evidence,
                                                   "agent_type": "news"}):
            out = domain_agents.run_news_agent(self._news_state())
        return out["corroboration_result"]["status"]

    # -- the case the standing failure named --------------------------------

    def test_a_completed_nei_with_no_search_is_not_silence(self):
        """Nothing was searched, so nothing is known about the filing."""
        status = self._status(self._tool_result("NOT_ENOUGH_INFO"))
        assert status != A2A_NO_MATCHING_DISCLOSURE
        assert status == A2A_SOURCE_UNAVAILABLE

    def test_a_successful_search_with_no_hits_is_silence(self):
        """The contrast case: an applicable filing was read and said nothing."""
        status = self._status(
            self._tool_result("NOT_ENOUGH_INFO", provenance=[self._search()]))
        assert status == A2A_NO_MATCHING_DISCLOSURE

    # -- source availability -------------------------------------------------

    def test_an_unavailable_source_is_reported_as_such(self):
        status = self._status(self._tool_result(
            "NOT_ENOUGH_INFO",
            provenance=[self._search(
                success=False, reason=None,
                error="RAG service not available (check Postgres and the "
                      "Ollama embedder)")]))
        assert status == A2A_SOURCE_UNAVAILABLE

    def test_a_failed_search_is_failed_not_silence(self):
        status = self._status(self._tool_result(
            "NOT_ENOUGH_INFO",
            provenance=[self._search(success=False, reason=None,
                                     error="malformed query")]))
        assert status == A2A_FAILED

    def test_an_empty_corpus_is_reported_as_such(self):
        """Distinct from silence: there was no filing to be silent."""
        status = self._status(self._tool_result(
            "NOT_ENOUGH_INFO",
            provenance=[self._search(reason="no_corpus")]))
        assert status == A2A_NO_CORPUS

    def test_one_successful_search_is_enough_to_establish_silence(self):
        """A failed attempt followed by a successful one still read a filing."""
        status = self._status(self._tool_result(
            "NOT_ENOUGH_INFO",
            provenance=[self._search(success=False, reason=None, error="blip"),
                        self._search()]))
        assert status == A2A_NO_MATCHING_DISCLOSURE

    # -- comparison still works ---------------------------------------------

    def test_two_decisive_verdicts_that_disagree_contradict(self):
        status = self._status(
            self._tool_result("REFUTES", provenance=[self._search()]),
            parent_verdict="SUPPORTS")
        assert status == A2A_CONTRADICTS

    def test_two_decisive_verdicts_that_agree_corroborate(self):
        status = self._status(
            self._tool_result("SUPPORTS", provenance=[self._search()]),
            parent_verdict="SUPPORTS")
        assert status == A2A_CORROBORATES

    def test_a_decisive_pair_is_compared_even_without_a_filing_search(self):
        """The gate gowerns the *silence* reading, not the comparison: two
        decisive verdicts disagree regardless of how each was reached."""
        status = self._status(self._tool_result("REFUTES"),
                              parent_verdict="SUPPORTS")
        assert status == A2A_CONTRADICTS

    def test_a_nested_failure_is_still_failed(self):
        status = self._status(
            self._tool_result("NOT_ENOUGH_INFO", execution_status="failed"))
        assert status == A2A_FAILED

    def test_a_filing_that_predates_the_event_stays_out_of_scope(self):
        """NOT_APPLICABLE_YET is decided before delegation and must survive."""
        from finvet.models.a2a import reclassify_corroboration

        result = A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_NOT_APPLICABLE_YET, verdict="NOT_ENOUGH_INFO",
        ).model_dump()
        assert (reclassify_corroboration("SUPPORTS", result)["status"]
                == A2A_NOT_APPLICABLE_YET)


class TestTheToolDoesNotClassify:
    """The tool runs before the parent verdict exists, so any status it sets is
    a placeholder. It used to borrow NO_MATCHING_DISCLOSURE for that, which is
    a real audit-facing outcome -- a result that never reached
    reclassification would have read as a filing saying nothing."""

    def test_the_placeholder_is_neutral(self):
        def fake_scoped(state, **kwargs):
            return {"agent_evidence": {
                "verdict": "SUPPORTS", "confidence": 0.9, "provenance": [],
                "tools_called": [], "execution_status": "completed"}}

        with patch("finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
                   fake_scoped):
            result = corroborate_sec.corroborate_with_filing.invoke({
                "finding": "f", "ticker": "AAPL", "metric": "fine_amount",
                "claimed_value": 5e8, "operator": "eq", "period": "2025"})

        assert result["status"] == A2A_PENDING_CLASSIFICATION
