"""LangGraph callback handler for automatic pipeline audit logging.

Hooks into LangGraph's callback system to log node timing, tool calls,
and errors to the existing audit_events table — without any changes to
node code.

Usage in verify.py:
    callback = AuditCallbackHandler(audit, request_id)
    config = {
        "configurable": {"thread_id": request_id},
        "callbacks": [callback],
    }
    result = graph.invoke(initial_state, config)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from ..config.constants import MAX_CALLBACK_DATA_CHARS
from ..utils.logging import get_logger

logger = get_logger(__name__)

# The 12 nodes registered in workflow.py. Only events matching these names
# are logged — all internal LangGraph plumbing (graph wrapper, conditional
# edges, state merges) is silently skipped.
TRACKED_NODES = frozenset({
    "input_guardrails",
    "claim_parser",
    "period_resolver",
    "sec_agent",
    "market_agent",
    "news_agent",
    "reject_handler",
    "consensus",
    "output_guardrails",
    "hitl_checkpoint",
    "apply_hitl_decision",
    "response_generator",
})


def _truncate(text: str, max_chars: int = MAX_CALLBACK_DATA_CHARS) -> str:
    """Truncate text to max_chars, appending '...' if truncated."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class AuditCallbackHandler(BaseCallbackHandler):
    """Logs LangGraph pipeline events to the audit trail.

    Tracks three types of events:
    - Node lifecycle: started, completed (with duration), errors
    - Tool calls: name, input args, output preview
    - Errors: node or tool failures with truncated messages

    All data is truncated to MAX_CALLBACK_DATA_CHARS before storage.
    """

    def __init__(self, audit_logger, request_id: str):
        """Initialize with an AuditLogger instance and request ID.

        Args:
            audit_logger: The AuditLogger to write events to.
            request_id: The request ID to associate all events with.
        """
        super().__init__()
        self._audit = audit_logger
        self._request_id = request_id
        # Maps run_id -> (node_name, start_time) for duration calculation
        self._active_nodes: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a graph node starts executing."""
        node_name = kwargs.get("name", "")
        metadata = kwargs.get("metadata", {})

        # Only track our pipeline nodes, skip LangGraph internals
        langgraph_node = metadata.get("langgraph_node", "")
        if langgraph_node not in TRACKED_NODES and node_name not in TRACKED_NODES:
            return

        resolved_name = langgraph_node or node_name
        step = metadata.get("langgraph_step")

        self._active_nodes[str(run_id)] = (resolved_name, datetime.utcnow())

        self._audit.log_event(
            event_type="node_started",
            request_id=self._request_id,
            data={"node": resolved_name, "step": step},
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a graph node finishes executing."""
        entry = self._active_nodes.pop(str(run_id), None)
        if entry is None:
            return  # Not a tracked node

        node_name, start_time = entry
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        self._audit.log_event(
            event_type="node_completed",
            request_id=self._request_id,
            data={"node": node_name, "duration_ms": duration_ms},
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a graph node throws an exception."""
        entry = self._active_nodes.pop(str(run_id), None)
        if entry is None:
            return  # Not a tracked node

        node_name, start_time = entry
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        self._audit.log_event(
            event_type="node_error",
            request_id=self._request_id,
            data={
                "node": node_name,
                "duration_ms": duration_ms,
                "error": _truncate(str(error)),
            },
        )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool is invoked."""
        tool_name = (serialized or {}).get("name", "unknown")

        self._audit.log_event(
            event_type="tool_called",
            request_id=self._request_id,
            data={
                "tool": tool_name,
                "input": _truncate(input_str),
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool returns a result."""
        self._audit.log_event(
            event_type="tool_completed",
            request_id=self._request_id,
            data={"output_preview": _truncate(str(output))},
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool throws an exception."""
        self._audit.log_event(
            event_type="tool_error",
            request_id=self._request_id,
            data={"error": _truncate(str(error))},
        )
