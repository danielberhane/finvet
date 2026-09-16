"""A node handed no parsed claim declines; it does not run the agent.

`_unsupported_claim` is the pre-flight whose whole job is to refuse, for free,
the claims an agent is already known to be unable to answer. Its first guard
handled the one input it cannot inspect -- no parse at all -- by returning
None, which the calling node reads as "proceed". An agent with no ticker,
metric or period has nothing to look up; it would spend its tool budget and
end in NOT_ENOUGH_INFO regardless, which is precisely the outcome the
pre-flight exists to prevent.

The branch is unreachable on the graph path (the router sends a missing parse
to reject_handler), so nothing observed it. The reader who did asked the right
question: why let the agent run when you already know it will not lead to an
answer? Now it declines like the other three reasons, with a stated
limitation, at no cost.
"""

import pytest

from finvet.graph.nodes import domain_agents
from finvet.graph.nodes.domain_agents import (
    run_market_agent, run_news_agent, run_sec_agent,
)


@pytest.fixture
def no_agent_may_run(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("an agent was constructed for a claim with no parse")
    monkeypatch.setattr(domain_agents, "_run_agent", _refuse)


@pytest.mark.parametrize("node,agent_type", [
    (run_sec_agent, "sec"),
    (run_market_agent, "market"),
    (run_news_agent, "news"),
])
class TestAMissingParseIsDeclinedByEveryAgentNode:

    def test_the_node_declines_without_running_an_agent(
            self, node, agent_type, no_agent_may_run):
        out = node({"request_id": "r", "claim_raw": "anything"})

        evidence = out["agent_evidence"]
        assert evidence["limitation"] == "no_parsed_claim"
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["tools_called"] == []
        assert out["agent_type"] == agent_type

    def test_the_decline_names_its_agent(self, node, agent_type, no_agent_may_run):
        out = node({"request_id": "r", "claim_raw": "anything"})

        assert out["agent_evidence"]["agent"] == agent_type

    def test_a_stated_reason_keeps_it_out_of_the_review_queue(
            self, node, agent_type, no_agent_may_run):
        from finvet.graph.nodes.output_guardrails import output_guardrails

        declined = node({"request_id": "r", "claim_raw": "anything"})["agent_evidence"]
        out = output_guardrails({
            "request_id": "r", "claim_raw": "anything",
            "verdict": declined["verdict"], "confidence": declined["confidence"],
            "agent_evidence": declined})

        assert "low_confidence" not in (out.get("hitl_triggers") or [])
