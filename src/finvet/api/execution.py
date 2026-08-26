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

from fastapi import HTTPException

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


def resolve_memory_context(claim_memory, request_id: Optional[str],
                           ) -> Optional[Dict[str, Any]]:
    """Read the prior episode the caller named, from the server's own store.

    Returns None when no id was given. Raises HTTPException(404) when an id was
    given and names nothing: the caller asked for specific context, and
    silently proceeding without it would verify a different question than the
    one they submitted.
    """
    if not request_id:
        return None
    if claim_memory is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "memory_context_not_found",
                "message": "Claim memory is disabled; no prior verification "
                           "to reuse.",
            })

    # Three failures, three answers. They used to collapse into one 404, which
    # told a caller their episode did not exist when the store was down --
    # sending them to re-run a claim whose answer was sitting in it.
    from ..memory.store_service import ClaimMemoryCorrupt, ClaimMemoryUnavailable

    try:
        match = claim_memory.get_claim(request_id)
    except ClaimMemoryUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "memory_context_unavailable",
                "message": f"The memory store could not be reached: {exc}",
            })
    except ClaimMemoryCorrupt as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "memory_context_corrupt",
                "message": f"The stored verification cannot be read: {exc}",
            })

    if match is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "memory_context_not_found",
                "message": f"No stored verification for {request_id}",
            })
    return match


def begin_request(audit, *, request_id: str, claim_text: str, user_id: str,
                  started_at: datetime,
                  memory_context: Optional[Dict[str, Any]] = None,
                  ) -> Dict[str, Any]:
    """Open a request: record its start and build the graph's initial state.

    Shared because the two routes had drifted here as well as at the end. Both
    built the same state dict by hand, but only /verify logged the
    memory_context_injected decision -- and the UI always streams, so that
    audit event never fired in practice. Only /verify put a timestamp on
    input_received. One builder means one answer.
    """
    audit.log_event(
        event_type="input_received",
        request_id=request_id,
        data={
            "claim_raw": claim_text,
            "user_id": user_id,
            "timestamp": started_at.isoformat(),
        },
    )

    if memory_context:
        # The user chose "Verify With Context" at /memory-check. That decision
        # changes what the agent sees, so it belongs in the trail.
        audit.log_event(
            event_type="memory_context_injected",
            request_id=request_id,
            data={
                "user_decision": "with_context",
                # An exact lookup by id. There is no similarity to record --
                # the field was always None and suggested a fuzzy match had
                # been scored. What matters is the role the text plays.
                "prior_request_id": getattr(memory_context, "request_id", None),
                "context_role": "untrusted_historical_context",
            },
        )

    return {
        "claim_raw": claim_text,
        "user_id": user_id,
        "request_id": request_id,
        "timestamp_received": started_at.isoformat(),
        "total_tokens_used": 0,
        "memory_context": memory_context,
    }


def derive_terminal_status(state: Mapping[str, Any],
                           final_response: Optional[Mapping[str, Any]]) -> TerminalStatus:
    """What actually happened, read from the pipeline's own output.

    Derived rather than passed in by each call site: the routes used to label
    every completed run "success", so a parser rejection -- a claim the system
    declined to verify -- was stored as a successful verification.
    """
    if state.get("hitl_required") and not state.get("hitl_checkpoint_passed"):
        return "pending_review"
    if not final_response:
        return "error"
    if final_response.get("status") == "rejected" or \
            final_response.get("verdict") == "REJECTED":
        return "rejected"
    return "success"


def build_error_response(request_id: str, claim_text: str, message: str,
                         *, error_code: str = "internal_error") -> Dict[str, Any]:
    """The canonical body for a run that produced no verdict.

    A terminal outcome needs a response shape even when nothing was verified,
    or the run leaves no record of having happened.
    """
    return {
        "status": "error",
        "request_id": request_id,
        "claim": claim_text,
        "verdict": None,
        "confidence": 0.0,
        "summary": "Verification did not complete.",
        "explanation": message,
        "sources": [],
        "disclosures": [],
        "metadata": {"error_code": error_code},
    }


def build_guardrail_response(request_id: str, claim_text: str,
                             violation_type: str, message: str) -> Dict[str, Any]:
    """The canonical body for a claim refused at a guardrail.

    A refusal is a terminal outcome and belongs on the record: without it, the
    audit trail cannot show what the system declined or why.
    """
    return {
        "status": "guardrail_blocked",
        "request_id": request_id,
        "claim": claim_text,
        "verdict": None,
        "confidence": 0.0,
        "summary": "This claim was refused by an input guardrail.",
        "explanation": message,
        "sources": [],
        "disclosures": [],
        "metadata": {"error_code": violation_type},
    }


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

    # -- coordinator entry points -------------------------------------------
    #
    # One per terminal outcome, so a route never decides for itself what a
    # status means or whether an outcome is worth recording. Each derives the
    # status, builds the canonical body, commits once, and returns the body
    # for the caller to hand to the client. The caller emits only after this
    # returns, which is what makes "on record before the client is told" true.

    def finish_pipeline_outcome(
        self,
        state: Mapping[str, Any],
        final_response: Optional[Mapping[str, Any]],
        *,
        claim_memory=None,
        preliminary_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Terminalize a run that reached the graph: success, pending, rejected,
        or error when it produced no response."""
        status = derive_terminal_status(state, final_response)

        if status == "pending_review":
            body = build_pending_response(
                self.request_id, self.claim_text, state,
                preliminary_analysis=preliminary_analysis)
        elif status == "error":
            body = build_error_response(
                self.request_id, self.claim_text,
                "The pipeline completed without producing a response.",
                error_code="no_final_response")
        else:
            body = dict(final_response or {})

        self.finish(status=status, state=state, final_response=body)

        # After the commit: a memory write must never decide whether a
        # verified run is on record.
        if status == "success":
            store_completed_claim(
                claim_memory, request_id=self.request_id,
                claim_text=self.claim_text, final_response=body, state=state)
        return body

    def finish_guardrail(self, violation_type: str, message: str) -> Dict[str, Any]:
        """Terminalize a claim refused at a guardrail."""
        body = build_guardrail_response(
            self.request_id, self.claim_text, violation_type, message)
        self.finish(status="guardrail_blocked", state={}, final_response=body)
        return body

    def finish_error(self, message: str, *, error_code: str = "internal_error",
                     ) -> Dict[str, Any]:
        """Terminalize a run that failed unexpectedly."""
        body = build_error_response(self.request_id, self.claim_text, message,
                                    error_code=error_code)
        self.finish(status="error", state={}, final_response=body)
        return body


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
