"""Tests for the human-facing reasoning on agent failure.

An agent crash used to surface the raw exception in the HITL review screen —
"Recursion limit of 11 reached ... visit https://docs.langchain.com/..." read
as a stack trace where a reviewer needed a judgment. The raw error still lands
in the audit trail via the node_error event; the reasoning shown to a reviewer
states what happened and what to do, in prose.
"""

from finvet.agents.base import compose_failure_reasoning

RECURSION_MSG = (
    "Recursion limit of 11 reached without hitting a stop condition. "
    "You can increase the limit by setting the `recursion_limit` config key.\n"
    "For troubleshooting, visit: "
    "https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT"
)


class TestRecursionLimitReads:

    def test_no_framework_jargon_reaches_the_reviewer(self):
        reasoning = compose_failure_reasoning(RECURSION_MSG)
        assert "Recursion limit" not in reasoning
        assert "recursion_limit" not in reasoning
        assert "langchain.com" not in reasoning

    def test_states_what_happened_and_where_the_detail_lives(self):
        reasoning = compose_failure_reasoning(RECURSION_MSG)
        assert "human review" in reasoning
        assert "audit trail" in reasoning


class TestOtherErrorsStayHonest:

    def test_first_line_of_an_unknown_error_is_carried(self):
        reasoning = compose_failure_reasoning("Connection refused by upstream\nstack...")
        assert "Connection refused by upstream" in reasoning
        assert "stack..." not in reasoning

    def test_long_errors_are_truncated_not_dumped(self):
        reasoning = compose_failure_reasoning("x" * 2000)
        assert len(reasoning) < 400

    def test_empty_error_still_produces_a_sentence(self):
        assert compose_failure_reasoning("").strip()
