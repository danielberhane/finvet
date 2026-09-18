"""The agent loop admits exactly the tool rounds its budget says, then stops.

`BaseVerificationAgent` runs the model<->tools loop under
`recursion_limit = max_iterations * 2 + 1`: one super-step for the model
turn, one for the tool turn, and one for the final answer. So a budget of N
admits N-1 tool rounds; the Nth round trips the limit. That arithmetic is
what the 2026-09-13 incident turned on -- the model behind the provider's
alias started needing five rounds against a budget of five, and every claim
escalated -- and it is what this test pins.

It is also the guard for swapping the loop's implementation. It drives the
real constructor with a scripted model and asserts on what the loop returns
and raises, not on which library built it: if a replacement keeps these
facts, nothing downstream notices. The facts:

- the returned state holds no SystemMessage (the verdict call drops the
  *last* message as anti-anchoring; it must be the model's, not the prompt)
- the limit raises GraphRecursionError whose text says "Recursion limit",
  the string compose_failure_reasoning keys on
- execute() turns that into _error_evidence, before any verdict call
"""

from typing import Any, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from finvet.agents import base as base_module
from finvet.agents.base import BaseVerificationAgent


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


class ScriptedModel(BaseChatModel):
    """Calls `lookup` n_rounds times, then answers."""
    n_rounds: int = 3
    calls: int = 0
    first_prompt: List[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.first_prompt:
            self.first_prompt = [type(m).__name__ for m in messages]
        self.calls += 1
        if self.calls <= self.n_rounds:
            msg = AIMessage(content="", tool_calls=[
                {"id": f"c{self.calls}", "name": "lookup", "args": {"q": str(self.calls)}}])
        else:
            msg = AIMessage(content="final answer")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


class LookupAgent(BaseVerificationAgent):
    def __init__(self, max_iterations: int):
        super().__init__(agent_type="sec", tools=[lookup],
                         system_prompt="You verify claims.", max_iterations=max_iterations)

    def _get_source_description(self) -> str:
        return "scripted"


@pytest.fixture
def agent_with(monkeypatch):
    def _build(n_rounds: int, budget: int) -> LookupAgent:
        model = ScriptedModel(n_rounds=n_rounds)
        model.first_prompt = []
        monkeypatch.setattr(base_module, "create_llm", lambda role: model)
        return LookupAgent(max_iterations=budget)
    return _build


def _run_loop(agent: LookupAgent):
    return agent.react_agent.invoke(
        {"messages": [HumanMessage(content="claim")]},
        config={"recursion_limit": agent.max_iterations * 2 + 1})


class TestBudgetArithmetic:

    def test_budget_five_admits_four_rounds(self, agent_with):
        out = _run_loop(agent_with(n_rounds=4, budget=5))
        msgs = out["messages"]
        assert isinstance(msgs[-1], AIMessage) and msgs[-1].content == "final answer"
        assert sum(isinstance(m, ToolMessage) for m in msgs) == 4

    def test_budget_five_refuses_the_fifth_round(self, agent_with):
        with pytest.raises(GraphRecursionError) as excinfo:
            _run_loop(agent_with(n_rounds=5, budget=5))
        assert "Recursion limit" in str(excinfo.value)

    def test_the_production_budget_admits_seven_rounds(self, agent_with):
        out = _run_loop(agent_with(n_rounds=7, budget=8))
        assert sum(isinstance(m, ToolMessage) for m in out["messages"]) == 7

    def test_the_production_budget_refuses_the_eighth(self, agent_with):
        with pytest.raises(GraphRecursionError):
            _run_loop(agent_with(n_rounds=8, budget=8))


class TestWhatTheLoopHandsBack:

    def test_the_system_prompt_reaches_the_model_but_not_the_state(self, agent_with):
        agent = agent_with(n_rounds=1, budget=5)
        out = _run_loop(agent)
        assert agent.llm.first_prompt[:2] == ["SystemMessage", "HumanMessage"]
        assert not any(isinstance(m, SystemMessage) for m in out["messages"])

    def test_the_last_message_is_the_models_own(self, agent_with):
        out = _run_loop(agent_with(n_rounds=2, budget=5))
        last = out["messages"][-1]
        assert isinstance(last, AIMessage) and not last.tool_calls


class TestExecuteOnExhaustion:

    def test_execute_returns_error_evidence_naming_the_budget(self, agent_with):
        agent = agent_with(n_rounds=5, budget=5)
        evidence = agent.execute({"request_id": "r", "claim_raw": "a claim"})

        assert evidence["execution_status"] == "failed"
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert "Recursion limit" in (evidence["error"] or "")
        assert "tool-call budget" in evidence["reasoning"]
