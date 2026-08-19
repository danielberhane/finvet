"""Data models for audit trail and audit events."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class AuditEvent(BaseModel):
    """Individual audit event in the verification pipeline."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}", description="Unique event ID")
    parent_event: Optional[str] = Field(None, description="Parent event ID for chaining")
    event_type: str = Field(..., description="Type of event")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO format timestamp")

    # Event-specific data
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")

    # Tool call tracking
    tool_name: Optional[str] = Field(None, description="Tool name if this is a tool call")
    tool_parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")
    tool_response: Optional[Any] = Field(None, description="Tool response")
    tool_duration_ms: Optional[int] = Field(None, description="Tool execution time")

    # LLM tracking
    llm_model: Optional[str] = Field(None, description="LLM model used")
    llm_tokens: Optional[int] = Field(None, description="Tokens used")
    llm_prompt: Optional[str] = Field(None, description="Prompt sent to LLM")
    llm_response: Optional[str] = Field(None, description="Response from LLM")

    # User and request context
    user_id: Optional[str] = Field(None, description="User identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")

    # Flags and warnings
    flags: List[str] = Field(default_factory=list, description="Flags raised during this event")
    warnings: List[str] = Field(default_factory=list, description="Warnings during this event")
    errors: List[str] = Field(default_factory=list, description="Errors during this event")


class AuditTrail(BaseModel):
    """Complete audit trail for a verification request."""

    request_id: str = Field(..., description="Unique request identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp_start: str = Field(..., description="Start time")
    timestamp_end: Optional[str] = Field(None, description="End time")

    # Original request
    claim_raw: str = Field(..., description="Original claim text")
    claim_hash: str = Field(..., description="SHA256 hash of claim")

    # All events
    events: List[AuditEvent] = Field(default_factory=list, description="All audit events")

    # Summary counts
    total_events: int = Field(0, description="Total number of events")
    total_tool_calls: int = Field(0, description="Total tool calls made")
    total_tokens: int = Field(0, description="Total LLM tokens used")
    execution_time_ms: Optional[int] = Field(None, description="Total execution time")

    # Configuration versions
    config_version: str = Field(..., description="Configuration version used")
    router_policy_version: str = Field(..., description="Routing policy version")

    # Final outcome
    final_verdict: Optional[str] = Field(None, description="Final verdict")
    final_confidence: Optional[float] = Field(None, description="Final confidence")
    hitl_triggered: bool = Field(False, description="Whether HITL was triggered")

    # Integrity
    execution_hash: Optional[str] = Field(None, description="Hash of entire execution trace")

    def add_event(self, event: AuditEvent) -> None:
        """Add an event to the audit trail."""
        self.events.append(event)
        self.total_events += 1
        if event.tool_name:
            self.total_tool_calls += 1
        if event.llm_tokens:
            self.total_tokens += event.llm_tokens

    def compute_execution_time(self) -> None:
        """Compute total execution time if end timestamp is set."""
        if self.timestamp_end:
            start = datetime.fromisoformat(self.timestamp_start)
            end = datetime.fromisoformat(self.timestamp_end)
            self.execution_time_ms = int((end - start).total_seconds() * 1000)
