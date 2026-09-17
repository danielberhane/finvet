"""A numeric claim stated in another currency is declined before an agent runs.

Filed values and quotes are compared in USD, and the 7-field parser contract
carries no currency. "Apple's revenue was €360 billion" would otherwise reach
the comparator as 360e9 against 391e9 USD and be refuted -- a confident wrong
verdict on a claim that may well be true at the day's exchange rate. That is
the one error the system exists to never make.

The check reads the claim text, like the Q4 check, so it holds for any parser:
a rule in a prompt binds only a model that reads the prompt, and the parser is
about to become fine-tuned weights that read a one-line prompt.

Only sec and market are gated. Fines and settlements are corroborated through
filing_amounts, which reads the currency out of the filing text itself.
"""

import pytest

from finvet.graph.nodes import domain_agents
from finvet.graph.nodes.domain_agents import (
    _names_non_usd_amount, run_market_agent, run_news_agent, run_sec_agent,
)
from finvet.models.claim import ParsedClaim


@pytest.fixture
def no_agent_may_run(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("an agent was constructed for a non-USD claim")
    monkeypatch.setattr(domain_agents, "_run_agent", _refuse)


def _sec(value=360e9):
    return ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                       operator="eq", value=value, period="fiscal 2024")


def _market(value=150.0):
    return ParsedClaim(claim_type="market", ticker="AAPL",
                       metric="closing_price", operator="eq", value=value)


class TestTheClaimTextIsRead:

    @pytest.mark.parametrize("text", [
        "Apple's fiscal 2024 revenue was €360 billion",
        "Apple's fiscal 2024 revenue was 360 billion euros",
        "Apple's fiscal 2024 revenue was EUR 360 billion",
        "Shell's revenue was £250 billion",
        "Toyota's revenue was ¥45 trillion",
        "Reliance's revenue was ₹9 trillion",
    ])
    def test_a_non_usd_marker_is_recognised(self, text):
        assert _names_non_usd_amount(text)

    @pytest.mark.parametrize("text", [
        "Apple's fiscal 2024 revenue was $391 billion",
        "Apple's fiscal 2024 revenue was 391 billion dollars",
        "Apple's fiscal 2024 revenue was USD 391 billion",
        # a word that contains a code is not the code
        "Europe accounted for a quarter of Apple's revenue",
    ])
    def test_usd_and_unrelated_text_pass(self, text):
        assert not _names_non_usd_amount(text)


class TestTheNodesDecline:

    def test_sec_declines_without_running_an_agent(self, no_agent_may_run):
        out = run_sec_agent({
            "request_id": "r", "parsed_claim": _sec(),
            "claim_raw": "Apple's fiscal 2024 revenue was €360 billion"})

        assert out["agent_evidence"]["limitation"] == "non_usd_amount"
        assert out["agent_evidence"]["verdict"] == "NOT_ENOUGH_INFO"
        assert out["agent_evidence"]["tools_called"] == []

    def test_market_declines_and_names_itself(self, no_agent_may_run):
        out = run_market_agent({
            "request_id": "r", "parsed_claim": _market(),
            "claim_raw": "Apple closed at €140 yesterday"})

        assert out["agent_evidence"]["limitation"] == "non_usd_amount"
        assert out["agent_evidence"]["agent"] == "market"

    def test_the_normalised_text_is_preferred(self, no_agent_may_run):
        """input_guardrails writes claim_normalized; that is what the agent
        would have seen, so that is what the guard reads."""
        out = run_sec_agent({
            "request_id": "r", "parsed_claim": _sec(),
            "claim_raw": "Apple's revenue was $391 billion",
            "claim_normalized": "Apple's revenue was €360 billion"})

        assert out["agent_evidence"]["limitation"] == "non_usd_amount"

    def test_a_stated_reason_keeps_it_out_of_the_review_queue(self, no_agent_may_run):
        from finvet.graph.nodes.output_guardrails import output_guardrails

        declined = run_sec_agent({
            "request_id": "r", "parsed_claim": _sec(),
            "claim_raw": "Apple's revenue was €360 billion"})["agent_evidence"]
        out = output_guardrails({
            "request_id": "r", "claim_raw": "Apple's revenue was €360 billion",
            "verdict": declined["verdict"], "confidence": declined["confidence"],
            "agent_evidence": declined})

        assert "low_confidence" not in (out.get("hitl_triggers") or [])


class TestNothingElseIsAffected:

    def test_a_usd_claim_proceeds(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(domain_agents, "_run_agent",
                            lambda *a, **k: seen.setdefault("ran", True) and
                            {"agent_evidence": {"provenance": []}, "agent_type": "sec"})
        run_sec_agent({"request_id": "r", "parsed_claim": _sec(391e9),
                       "claim_raw": "Apple's fiscal 2024 revenue was $391 billion"})
        assert seen.get("ran")

    def test_a_claim_with_no_value_is_not_gated(self, monkeypatch):
        """'Apple's annual report discusses euro exposure' has nothing to
        compare; the narrative path is unaffected."""
        seen = {}
        monkeypatch.setattr(domain_agents, "_run_agent",
                            lambda *a, **k: seen.setdefault("ran", True) and
                            {"agent_evidence": {"provenance": []}, "agent_type": "sec"})
        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric=None)
        run_sec_agent({"request_id": "r", "parsed_claim": claim,
                       "claim_raw": "Apple's annual report discusses euro exposure"})
        assert seen.get("ran")

    def test_news_is_not_gated(self, monkeypatch):
        """A fine in euros is corroborated through filing_amounts, which reads
        the currency from the filing; the news node must not pre-empt that."""
        seen = {}
        monkeypatch.setattr(domain_agents, "_run_agent",
                            lambda *a, **k: seen.setdefault("ran", True) and
                            {"agent_evidence": {"provenance": [], "verdict": "NOT_ENOUGH_INFO",
                                                "confidence": 0.2, "tools_called": []},
                             "agent_type": "news"})
        claim = ParsedClaim(claim_type="news", ticker="AAPL", metric="fine_amount",
                            operator="eq", value=1.8e9)
        run_news_agent({"request_id": "r", "parsed_claim": claim,
                        "claim_raw": "The EU fined Apple €1.8 billion"})
        assert seen.get("ran")
