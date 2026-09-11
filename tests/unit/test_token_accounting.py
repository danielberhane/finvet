"""The token counter the API reports must be the tokens the models consumed.

Before this file, `total_tokens_used` was 0 on every response: the parser read
`response_metadata["usage"]`, a key the provider never sets, and the agent loop
and verdict call recorded nothing at all. LangSmith showed ~21k tokens per
verification against a reported 0. Each test here drives the producer.
"""
import json
from unittest.mock import MagicMock, patch

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.errors import GraphRecursionError

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.graph.nodes.claim_parser import claim_parser
from finvet.graph.nodes.domain_agents import _run_agent

RAW = {"claim_type": "sec", "ticker": "JPM", "value": 158.1e9,
       "comparison": "eq", "period": "fiscal 2024", "currency": "USD",
       "reject_reason": None}


def _ai(total, **kw):
    return AIMessage(content="", usage_metadata={
        "input_tokens": total - 1, "output_tokens": 1, "total_tokens": total},
        response_metadata={"model_name": "deepseek-chat"}, **kw)


class TestTheParserCountsWhatTheProviderReports:

    def test_usage_metadata_is_the_source(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content=json.dumps(RAW),
            usage_metadata={"input_tokens": 2700, "output_tokens": 45,
                            "total_tokens": 2745},
            response_metadata={"token_usage": {"total_tokens": 2745}})
        with patch("finvet.graph.nodes.claim_parser.create_llm", return_value=llm), \
             patch("finvet.graph.nodes.claim_parser.get_audit_logger",
                   return_value=MagicMock()):
            result = claim_parser({
                "claim_raw": "JPMorgan's fiscal 2024 revenue came to $158.1 billion.",
                "request_id": "req_tok1", "total_tokens_used": 10})
        assert result["total_tokens_used"] == 2755


class _Agent(BaseVerificationAgent):
    def _get_source_description(self):
        return "SEC EDGAR"


def _loop(messages, *, then_raise=None):
    """A react agent that reports each AI turn the way a real model run does:
    through the callback handler in the config, then returns or aborts."""
    def invoke(_inputs, config):
        for m in messages:
            for cb in config["callbacks"]:
                cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=m)]]),
                              run_id=uuid4())
        if then_raise:
            raise then_raise
        return {"messages": messages}
    return type("R", (), {"invoke": staticmethod(invoke)})()


def _agent(monkeypatch, messages, *, then_raise=None):
    agent = _Agent.__new__(_Agent)
    agent.agent_type = "sec"
    agent.max_iterations = 5
    monkeypatch.setattr(agent, "_build_context", lambda state: "ctx", raising=False)
    monkeypatch.setattr(agent, "react_agent", _loop(messages, then_raise=then_raise),
                        raising=False)
    monkeypatch.setattr(agent, "_extract_tool_info",
                        lambda messages: ([], [], [], []), raising=False)
    monkeypatch.setattr("finvet.agents.base.resolve_trusted_observation",
                        lambda *a, **k: None)
    return agent


class TestTheAgentSumsEveryModelCall:

    def test_every_ai_message_in_the_loop_is_counted(self, monkeypatch):
        agent = _agent(monkeypatch, [_ai(3302), _ai(3475)])
        monkeypatch.setattr(agent, "_extract_verdict", lambda m, s: VerdictOutput(
            verdict="NOT_ENOUGH_INFO", confidence=0.3, reasoning="x",
            retrieved_value=None, source_description="10-K"), raising=False)
        evidence = agent.execute({"parsed_claim": None, "request_id": "r"})
        assert evidence["tokens_used"] == 3302 + 3475

    def test_the_verdict_call_is_counted_too(self, monkeypatch):
        agent = _agent(monkeypatch, [_ai(3302)])
        parsed = VerdictOutput(verdict="NOT_ENOUGH_INFO", confidence=0.3,
                               reasoning="x", retrieved_value=None,
                               source_description="10-K")
        structured = MagicMock()
        structured.invoke.return_value = {"raw": _ai(2246), "parsed": parsed,
                                          "parsing_error": None}
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        with patch("finvet.agents.base.create_llm", return_value=llm):
            evidence = agent.execute({"parsed_claim": None, "request_id": "r"})
        assert evidence["tokens_used"] == 3302 + 2246

    def test_a_verdict_that_fails_to_parse_still_raises(self, monkeypatch):
        agent = _agent(monkeypatch, [_ai(3302)])
        structured = MagicMock()
        structured.invoke.return_value = {"raw": _ai(50), "parsed": None,
                                          "parsing_error": ValueError("bad json")}
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        with patch("finvet.agents.base.create_llm", return_value=llm), \
             pytest.raises(ValueError):
            agent._extract_verdict([_ai(3302)], {"parsed_claim": None})

    def test_a_loop_that_hits_its_recursion_limit_still_reports_what_it_spent(
            self, monkeypatch):
        # The A2A delegation on a $3B settlement claim did exactly this: three
        # model calls, then GraphRecursionError, and 9.5k tokens vanished
        # because there were no returned messages to sum.
        agent = _agent(monkeypatch, [_ai(3000), _ai(3300), _ai(3200)],
                       then_raise=GraphRecursionError("Recursion limit of 7"))
        evidence = agent.execute({"parsed_claim": None, "request_id": "r"})
        assert evidence["execution_status"] == "failed"
        assert evidence["tokens_used"] == 9500

    def test_a_loop_that_never_answered_reports_zero_not_a_missing_key(
            self, monkeypatch):
        agent = _agent(monkeypatch, [], then_raise=RuntimeError("down"))
        evidence = agent.execute({"parsed_claim": None, "request_id": "r"})
        assert evidence["tokens_used"] == 0


class TestTheNodeCarriesAgentTokensIntoState:

    def test_agent_tokens_are_added_to_the_running_total(self):
        agent = MagicMock()
        agent.execute.return_value = {"verdict": "SUPPORTS", "confidence": 0.9,
                                      "tools_called": [], "tokens_used": 6777}
        result = _run_agent(lambda **kw: agent, "sec", "SEC EDGAR",
                            {"request_id": "r", "total_tokens_used": 2745})
        assert result["total_tokens_used"] == 2745 + 6777

    def test_an_agent_crash_leaves_the_total_unchanged(self):
        def boom(**kw):
            raise RuntimeError("no")
        result = _run_agent(boom, "sec", "SEC EDGAR",
                            {"request_id": "r", "total_tokens_used": 2745})
        assert result["total_tokens_used"] == 2745
