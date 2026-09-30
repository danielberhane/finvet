"""A claim about a fiscal period, from the parsed claim to the published verdict.

Claims whose filed facts defeat a simpler selection rule, and the claims
either side of them. Everything between the parser and the
response runs for real: the period resolver, the period target the node sets,
the statement tool, the client's selection, the agent's extraction of tool
results, the trusted-observation resolver, the comparator. Two things are
stubbed, both of them model calls -- the tool loop returns the messages the
real tool produced, and the verdict step returns a verdict that is wrong, so
the published one can only have come from the comparison.

SEC's feed is served from the recorded fixtures.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.graph.nodes.period_resolver import period_resolver
from finvet.mcp.sec_edgar import SECEdgarClient
from finvet.models.claim import ParsedClaim
from finvet.tools.sec_tools import (
    fiscal_target_for,
    get_balance_sheet,
    get_income_statement,
    period_target_for,
    use_period_target,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sec_companyfacts"
CIK = {"HD": "0000354950", "FDX": "0001048911", "MET": "0001099219",
       "BAC": "0000070858", "GE": "0000040545", "JPM": "0000019617"}
WRONG = {"SUPPORTS": "REFUTES", "REFUTES": "SUPPORTS", "NOT_ENOUGH_INFO": "REFUTES"}


class _Agent(BaseVerificationAgent):
    def _get_source_description(self):
        return "SEC EDGAR"


def _claim(ticker, metric, operator, value, period):
    return ParsedClaim(claim_type="sec", ticker=ticker, metric=metric,
                       operator=operator, value=value, period=period)


def _verify(monkeypatch, claim, *, expecting, tool=get_income_statement,
            verdict_step_fails=False):
    client = SECEdgarClient()
    raw = json.loads((FIXTURES / f"{claim.ticker}.json").read_text())
    monkeypatch.setattr(client, "_download_company_facts", lambda cik: raw)

    def no_filing(*a, **k):
        raise AssertionError("a fiscal claim opens no filing")
    monkeypatch.setattr(client._mcp, "call_tool", no_filing)
    monkeypatch.setattr("finvet.tools.sec_tools._get_client", lambda: client)

    state = {"parsed_claim": claim, "request_id": "r", "claim_raw": "claim"}
    period = period_resolver(state)["canonical_period"]
    state["canonical_period"] = period

    args = {"cik": CIK[claim.ticker], "accession_number": "0000000000-00-000000"}
    if tool is get_income_statement:
        args["period"] = period.period_type
    with use_period_target(*period_target_for(period),
                           fiscal=fiscal_target_for(period)):
        result = tool.invoke(args)

    messages = [
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": tool.name, "args": args}]),
        ToolMessage(content=str(result), tool_call_id="call_1", name=tool.name),
        AIMessage(content="done"),
    ]

    agent = _Agent.__new__(_Agent)
    agent.agent_type = "sec"
    agent.max_iterations = 8
    monkeypatch.setattr(agent, "_build_context", lambda s: "ctx", raising=False)
    monkeypatch.setattr(
        agent, "react_agent",
        type("R", (), {"invoke": staticmethod(
            lambda *a, **k: {"messages": messages})})(),
        raising=False)

    def verdict(messages, state):
        if verdict_step_fails:
            raise RuntimeError("the verdict model ran out of tokens")
        return VerdictOutput(verdict=WRONG[expecting], confidence=0.95,
                             reasoning="the model's reading", retrieved_value=1.0,
                             source_description="10-K")
    monkeypatch.setattr(agent, "_extract_verdict", verdict, raising=False)
    return agent.execute(state)


class TestClaimsASimplerRuleGetsWrong:

    def test_home_depot_fiscal_2025_revenue_is_supported(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("HD", "revenue", "eq", 164.5e9, "FY2025"),
            expecting="SUPPORTS")
        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["retrieved_value"] == 164_683_000_000
        assert evidence["trusted_observation"]["period_end"] == "2026-02-01"
        assert evidence["override_applied"] is True

    def test_home_depot_outside_the_band_is_refuted(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("HD", "revenue", "eq", 161.0e9, "FY2025"),
            expecting="REFUTES")
        assert evidence["verdict"] == "REFUTES"

    def test_fedex_net_income_below_a_bound_it_exceeds_is_refuted(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("FDX", "net_income", "lt", 4.2e9, "FY2026"),
            expecting="REFUTES")
        assert evidence["verdict"] == "REFUTES"
        assert evidence["retrieved_value"] == 4_433_000_000

    def test_metlife_net_income_is_the_annual_reports(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("MET", "net_income", "eq", 3.2e9, "FY2025"),
            expecting="REFUTES")
        assert evidence["verdict"] == "REFUTES"
        assert evidence["retrieved_value"] == 3_379_000_000

    def test_jpmorgan_total_assets_is_answered(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("JPM", "total_assets", "lte", 4.2e12, "FY2025"),
            expecting="REFUTES", tool=get_balance_sheet)
        assert evidence["verdict"] == "REFUTES"
        assert evidence["retrieved_value"] == 4_424_900_000_000


class TestARestatedPeriod:
    """Bank of America reported 26.463B for Q2 2025 and recast it to 27.443B
    a year later. General Electric reported 67.954B for 2023 and recast it to
    35.348B after the Vernova spin."""

    def test_a_claim_matching_the_value_of_record_is_supported(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("BAC", "revenue", "eq", 27.4e9, "Q2 2025"),
            expecting="SUPPORTS")
        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["limitation"] is None

    def test_a_claim_matching_only_what_was_restated_is_declined(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("BAC", "revenue", "eq", 26.4e9, "Q2 2025"),
            expecting="NOT_ENOUGH_INFO")
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["limitation"] == "matches_superseded_value"
        assert evidence["confidence"] <= 0.5

    def test_the_decline_shows_both_figures(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("BAC", "revenue", "eq", 26.4e9, "Q2 2025"),
            expecting="NOT_ENOUGH_INFO")
        assert evidence["retrieved_value"] == 27_443_000_000
        assert evidence["trusted_observation"]["superseded_values"] == [26_463_000_000]
        assert "26,463,000,000" in evidence["reasoning"]
        assert "27,443,000,000" in evidence["reasoning"]

    def test_a_claim_matching_neither_is_refuted(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("BAC", "revenue", "eq", 25.0e9, "Q2 2025"),
            expecting="REFUTES")
        assert evidence["verdict"] == "REFUTES"
        assert evidence["limitation"] is None

    def test_the_recast_year_is_supported(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("GE", "revenue", "eq", 35.3e9, "FY2023"),
            expecting="SUPPORTS")
        assert evidence["verdict"] == "SUPPORTS"

    def test_the_year_as_first_reported_is_declined_not_refuted(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("GE", "revenue", "eq", 68.0e9, "FY2023"),
            expecting="NOT_ENOUGH_INFO")
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["limitation"] == "matches_superseded_value"


class TestWhenTheVerdictModelFails:
    """The comparison needs nothing from the model, so its failure changes
    nothing about the answer -- including the decline."""

    def test_the_comparison_still_answers(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("HD", "revenue", "eq", 164.5e9, "FY2025"),
            expecting="SUPPORTS", verdict_step_fails=True)
        assert evidence["verdict"] == "SUPPORTS"

    def test_a_superseded_match_is_still_declined(self, monkeypatch):
        evidence = _verify(
            monkeypatch, _claim("BAC", "revenue", "eq", 26.4e9, "Q2 2025"),
            expecting="NOT_ENOUGH_INFO", verdict_step_fails=True)
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["limitation"] == "matches_superseded_value"


class TestAStatementForAnotherPeriodIsNotEvidence:

    def test_the_wrong_fiscal_year_yields_no_observation(self, monkeypatch):
        """The statement is fiscal 2024's; the claim is about fiscal 2025."""
        from finvet.models.evidence import (
            resolve_trusted_observation,
            tool_record_from_result,
        )
        client = SECEdgarClient()
        raw = json.loads((FIXTURES / "HD.json").read_text())
        monkeypatch.setattr(client, "_download_company_facts", lambda cik: raw)
        monkeypatch.setattr("finvet.tools.sec_tools._get_client", lambda: client)
        with use_period_target("2024-12-31", "annual", fiscal=(2024, "FY")):
            result = get_income_statement.invoke(
                {"cik": CIK["HD"], "accession_number": "x", "period": "annual"})
        record = tool_record_from_result("get_income_statement", {}, result)

        claim = _claim("HD", "revenue", "eq", 164.5e9, "FY2025")
        assert resolve_trusted_observation(
            claim, [record], expected_period_end="2025-12-31",
            expected_period_start="2025-01-01",
            expected_fiscal=(2025, "FY")) is None


@pytest.mark.parametrize("period,expected", [
    ("FY2025", (2025, "FY")),
    ("Q2 2025", (2025, "Q2")),
])
def test_the_fiscal_target_is_the_claims_own_label(period, expected):
    claim = _claim("HD", "revenue", "eq", 1.0, period)
    resolved = period_resolver({"parsed_claim": claim, "request_id": "r"})
    assert fiscal_target_for(resolved["canonical_period"]) == expected
