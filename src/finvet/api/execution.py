"""Exactly-once terminalization for a verification request.

Both `/verify` and `/verify-stream` run the same graph and must end a request
the same way. They did not: the streaming success path emitted its verdict
without committing an execution at all, streaming HITL omitted the data_sources
the synchronous path sent, and no error path released the request's event
buffer. Four copies of the lifecycle had drifted four ways.

This module owns that lifecycle. A route builds one `ExecutionFinalizer`, calls
`finish` exactly once, and lets `release` guarantee cleanup even when the
request ends in an exception.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Mapping, Optional

from ..audit.logger import AuditLogger
from ..utils.exceptions import AuditPersistenceError
from ..utils.logging import get_logger

logger = get_logger(__name__)

# How a request ended. Kept distinct from the verdict: a claim blocked at a
# guardrail and a claim answered NOT_ENOUGH_INFO are different outcomes that a
# verdict column cannot tell apart.
TerminalStatus = Literal[
    "success", "pending_review", "rejected", "guardrail_blocked", "error"
]

_AGENT_DISPLAY_NAMES = {"sec": "SEC", "news": "News", "market": "Market"}


def elapsed_ms(started_at: datetime) -> int:
    """Milliseconds since `started_at`."""
    return int((datetime.utcnow() - started_at).total_seconds() * 1000)


def agents_run_from_state(state: Mapping[str, Any]) -> list:
    """Display names of the agents that ran, from graph state."""
    agent = state.get("agent_type")
    if not agent:
        # At a HITL interrupt the node update may not have reached state, but
        # the evidence always names the agent that produced it.
        agent = (state.get("agent_evidence") or {}).get("agent")
    if not agent or agent == "unknown":
        return []
    return [_AGENT_DISPLAY_NAMES.get(agent, agent)]


@dataclass
class ExecutionFinalizer:
    """Commits a request's execution record exactly once.

    `finish` raises `AuditPersistenceError` when the record could not be
    written. That is deliberate: a verdict served with no durable audit row is
    the state the system claims cannot occur, so the caller must surface a
    failure rather than return an unaudited success.
    """

    audit: AuditLogger
    request_id: str
    claim_text: str
    started_at: datetime
    _finished: bool = field(default=False, init=False)

    @property
    def finished(self) -> bool:
        return self._finished

    def finish(
        self,
        *,
        status: TerminalStatus,
        state: Mapping[str, Any],
        final_response: Mapping[str, Any],
    ) -> None:
        if self._finished:
            raise RuntimeError(f"execution {self.request_id} finalized twice")
        self._finished = True

        metadata = final_response.get("metadata") or {}
        committed = self.audit.commit_execution(
            request_id=self.request_id,
            claim_text=self.claim_text,
            verdict=final_response.get("verdict"),
            confidence=final_response.get("confidence", 0.0),
            agents_run=agents_run_from_state(state),
            execution_time_ms=elapsed_ms(self.started_at),
            final_response=dict(final_response),
            data_sources=metadata.get("data_sources"),
            terminal_status=status,
        )

        if not committed:
            self.audit.discard_buffer(self.request_id)
            raise AuditPersistenceError(self.request_id)

    def release(self) -> None:
        """Drop any buffered events for a request that will never be committed.

        Safe to call unconditionally in a `finally`: a successful `finish`
        already cleared the buffer, so this is a no-op after it.
        """
        self.audit.discard_buffer(self.request_id)


def store_completed_claim(claim_memory, *, request_id: str, claim_text: str,
                          final_response: Mapping[str, Any],
                          state: Mapping[str, Any]) -> None:
    """Record a finished verification in episodic memory.

    Non-terminal by design — a memory write must never fail a verification that
    already produced a verdict — but the failure is logged rather than
    swallowed, so a memory backend that is down is visible in the logs instead
    of silently degrading recall.
    """
    if not claim_memory:
        return

    try:
        parsed_claim = state.get("parsed_claim")
        agent_evidence = state.get("agent_evidence") or {}
        claim_memory.store_claim(
            request_id=request_id,
            claim_text=claim_text,
            verdict=final_response.get("verdict", ""),
            confidence=final_response.get("confidence", 0),
            ticker=getattr(parsed_claim, "ticker", None),
            metric=getattr(parsed_claim, "metric", None),
            agent_type=agent_evidence.get("agent"),
        )
    except Exception as exc:
        logger.warning(f"Claim memory store failed (request: {request_id}): {exc}")


def build_pending_response(request_id: str, claim_text: str,
                           state: Mapping[str, Any],
                           preliminary_analysis: Optional[Dict[str, Any]] = None,
                           ) -> Dict[str, Any]:
    """The response body for a claim paused at the HITL checkpoint.

    Shared so both routes describe a pending review identically; they returned
    different shapes for the same graph result before.
    """
    agent_evidence = state.get("agent_evidence") or {}
    hitl_triggers = state.get("hitl_triggers", [])
    response = {
        "status": "pending_review",
        "request_id": request_id,
        "claim": claim_text,
        "verdict": "PENDING",
        "confidence": 0.0,
        "confidence_label": "PENDING",
        "summary": "This claim requires human review before a verdict can be provided.",
        "explanation": agent_evidence.get("reasoning", ""),
        "sources": [],
        "disclosures": ["Human review required for this claim"],
        "hitl_triggers": hitl_triggers,
        "metadata": {
            "hitl_required": True,
            "hitl_triggers": hitl_triggers,
            "agent": agent_evidence.get("agent"),
            "tools_called": agent_evidence.get("tools_called", []),
        },
    }
    if preliminary_analysis is not None:
        response["preliminary_analysis"] = preliminary_analysis
    return response
