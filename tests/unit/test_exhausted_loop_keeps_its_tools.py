"""A loop that hits its budget still reports the tools it ran.

An SEC agent called five tools, had the figure in hand, then kept going until
the recursion limit tripped. The failure evidence recorded `tools_called: []`, so the trajectory
layer, the audit record and the reviewer all saw an agent that did nothing.
LangGraph raises GraphRecursionError without returning state, and the loop
was reading its messages from that return value alone.

Driven through the real loop: a scripted model that never stops calling
tools, the production budget, and execute() on top.
"""

from typing import Any, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from finvet.agents import base as base_module
from finvet.agents.base import BaseVerificationAgent


@tool
def lookup(q: str) -> dict:
    """Look something up."""
    return {"success": True, "items": [{"line_item": "Revenues", "value": 1.0}],
            "q": q}


class NeverStops(BaseChatModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        msg = AIMessage(content="", tool_calls=[
            {"id": f"c{self.calls}", "name": "lookup", "args": {"q": str(self.calls)}}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "NeverStops":
        return self

    @property
    def _llm_type(self) -> str:
        return "never-stops"


class LookupAgent(BaseVerificationAgent):
    def __init__(self):
        super().__init__(agent_type="sec", tools=[lookup],
                         system_prompt="You verify claims.", max_iterations=8)
        self._provenance_tool_names = {"lookup"}

    def _get_source_description(self) -> str:
        return "scripted"


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(base_module, "create_llm", lambda role: NeverStops())
    agent = LookupAgent()
    return agent.execute({"claim_raw": "x", "parsed_claim": None,
                          "canonical_period": None})


class TestExhaustedLoopKeepsItsTools:

    def test_the_run_is_still_recorded_as_failed(self, evidence):
        assert evidence["execution_status"] == "failed"
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"

    def test_every_tool_round_that_ran_is_listed(self, evidence):
        # Budget 8 (recursion limit 17): eight tool rounds execute, and the
        # limit trips on the model turn that would start a ninth.
        assert evidence["tools_called"] == ["lookup"] * 8

    def test_the_detail_carries_each_call_and_its_result(self, evidence):
        detail: List[dict] = evidence["tool_calls_detail"]
        assert len(detail) == 8
        assert [d["args"] for d in detail] == [{"q": str(i)} for i in range(1, 9)]
        assert all(d["success"] for d in detail)

    def test_provenance_is_kept_for_tracked_tools(self, evidence):
        assert len(evidence["provenance"]) == 8
        assert evidence["provenance"][0]["result"]["items"][0]["line_item"] == "Revenues"
