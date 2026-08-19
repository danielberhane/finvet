"""Tests for the LangGraph audit callback handler."""

import uuid
from unittest.mock import MagicMock

import pytest

from src.finvet.audit.callbacks import AuditCallbackHandler


@pytest.fixture
def mock_audit():
    """Mock AuditLogger with a log_event method."""
    return MagicMock()


@pytest.fixture
def request_id():
    return "req_test123abc"


@pytest.fixture
def handler(mock_audit, request_id):
    return AuditCallbackHandler(mock_audit, request_id)


class TestNodeTracking:
    """Test that node start/end events are logged correctly."""

    def test_logs_node_started_for_pipeline_node(self, handler, mock_audit, request_id):
        """on_chain_start logs node_started for registered pipeline nodes."""
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=run_id,
            parent_run_id=uuid.uuid4(),
            name="claim_parser",
            metadata={"langgraph_node": "claim_parser", "langgraph_step": 2},
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "node_started"
        assert event_call.kwargs["request_id"] == request_id
        assert event_call.kwargs["data"]["node"] == "claim_parser"
        assert event_call.kwargs["data"]["step"] == 2

    def test_skips_internal_langgraph_events(self, handler, mock_audit):
        """on_chain_start ignores top-level LangGraph wrapper and internal events."""
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=uuid.uuid4(),
            parent_run_id=None,
            name="LangGraph",
            metadata={},
        )

        mock_audit.log_event.assert_not_called()

    def test_skips_unknown_node_names(self, handler, mock_audit):
        """on_chain_start ignores nodes not in TRACKED_NODES."""
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=uuid.uuid4(),
            parent_run_id=uuid.uuid4(),
            name="__pregel_internal",
            metadata={},
        )

        mock_audit.log_event.assert_not_called()

    def test_logs_node_completed_with_duration(self, handler, mock_audit, request_id):
        """on_chain_end logs node_completed with duration_ms."""
        run_id = uuid.uuid4()

        # Start the node first (so duration can be calculated)
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=run_id,
            parent_run_id=uuid.uuid4(),
            name="sec_agent",
            metadata={"langgraph_node": "sec_agent", "langgraph_step": 4},
        )
        mock_audit.reset_mock()

        # End the node
        handler.on_chain_end(
            outputs={"agent_evidence": {}},
            run_id=run_id,
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "node_completed"
        assert event_call.kwargs["data"]["node"] == "sec_agent"
        assert "duration_ms" in event_call.kwargs["data"]
        assert isinstance(event_call.kwargs["data"]["duration_ms"], int)

    def test_skips_chain_end_for_untracked_nodes(self, handler, mock_audit):
        """on_chain_end ignores events for nodes we didn't track in on_chain_start."""
        handler.on_chain_end(
            outputs={},
            run_id=uuid.uuid4(),  # never started
        )

        mock_audit.log_event.assert_not_called()

    def test_logs_node_error(self, handler, mock_audit, request_id):
        """on_chain_error logs node_error with truncated error message."""
        run_id = uuid.uuid4()

        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=run_id,
            parent_run_id=uuid.uuid4(),
            name="input_guardrails",
            metadata={"langgraph_node": "input_guardrails", "langgraph_step": 1},
        )
        mock_audit.reset_mock()

        handler.on_chain_error(
            error=ValueError("something went wrong"),
            run_id=run_id,
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "node_error"
        assert event_call.kwargs["data"]["node"] == "input_guardrails"
        assert "something went wrong" in event_call.kwargs["data"]["error"]


class TestToolTracking:
    """Test that tool call events are logged correctly."""

    def test_logs_tool_called(self, handler, mock_audit, request_id):
        """on_tool_start logs tool_called with name and truncated input."""
        handler.on_tool_start(
            serialized={"name": "get_xbrl_concepts"},
            input_str="ticker=AAPL, concept=Revenues",
            run_id=uuid.uuid4(),
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "tool_called"
        assert event_call.kwargs["data"]["tool"] == "get_xbrl_concepts"
        assert "AAPL" in event_call.kwargs["data"]["input"]

    def test_logs_tool_completed(self, handler, mock_audit, request_id):
        """on_tool_end logs tool_completed with truncated output."""
        handler.on_tool_end(
            output="{'revenue': 383285000000, 'period': '2023'}",
            run_id=uuid.uuid4(),
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "tool_completed"
        assert "383285000000" in event_call.kwargs["data"]["output_preview"]

    def test_logs_tool_error(self, handler, mock_audit, request_id):
        """on_tool_error logs tool_error with truncated error message."""
        handler.on_tool_error(
            error=ConnectionError("API timeout"),
            run_id=uuid.uuid4(),
        )

        mock_audit.log_event.assert_called_once()
        event_call = mock_audit.log_event.call_args
        assert event_call.kwargs["event_type"] == "tool_error"
        assert "API timeout" in event_call.kwargs["data"]["error"]

    def test_truncates_long_tool_output(self, handler, mock_audit):
        """Tool output longer than MAX_CALLBACK_DATA_CHARS is truncated."""
        long_output = "x" * 5000

        handler.on_tool_end(output=long_output, run_id=uuid.uuid4())

        event_call = mock_audit.log_event.call_args
        output_preview = event_call.kwargs["data"]["output_preview"]
        assert len(output_preview) <= 1003  # 1000 + "..."


class TestAllTrackedNodes:
    """Verify all 12 pipeline nodes are tracked."""

    @pytest.mark.parametrize("node_name", [
        "input_guardrails", "claim_parser", "period_resolver",
        "sec_agent", "market_agent", "news_agent", "reject_handler",
        "consensus", "output_guardrails", "hitl_checkpoint",
        "apply_hitl_decision", "response_generator",
    ])
    def test_tracks_all_pipeline_nodes(self, handler, mock_audit, node_name):
        """Every registered pipeline node should be tracked."""
        handler.on_chain_start(
            serialized=None,
            inputs={},
            run_id=uuid.uuid4(),
            parent_run_id=uuid.uuid4(),
            name=node_name,
            metadata={"langgraph_node": node_name, "langgraph_step": 1},
        )

        mock_audit.log_event.assert_called_once()
        assert mock_audit.log_event.call_args.kwargs["data"]["node"] == node_name
