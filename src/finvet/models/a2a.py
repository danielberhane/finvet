"""One contract for agent-to-agent corroboration, both directions.

FinVet has two delegations: the SEC agent asking News to confirm a filing
disclosure, and the News agent asking SEC whether the issuer's own filing
corroborates a reported event. They previously returned different shapes
(`news_verdict` vs a filing equivalent), which forced every consumer —
guardrails, audit trail, UI, evals — to know which direction it was reading.

One shape instead. `direction` says who asked whom; `status` is the audit-facing
outcome and is deliberately richer than the raw verdict, because "the filing does
not mention this" and "no filing could yet cover this event" are different facts
and only one of them is evidence of anything.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Audit-facing outcome. Kept separate from `verdict` because a verdict answers
# "what did the target agent conclude" while status answers "what does that mean
# for the claim under review" — and silence is not contradiction.
A2A_CORROBORATES = "CORROBORATES"
A2A_CONTRADICTS = "CONTRADICTS"
A2A_NO_MATCHING_DISCLOSURE = "NO_MATCHING_DISCLOSURE"
A2A_NOT_APPLICABLE_YET = "NOT_APPLICABLE_YET"
A2A_FAILED = "FAILED"

A2AStatus = Literal[
    "CORROBORATES",
    "CONTRADICTS",
    "NO_MATCHING_DISCLOSURE",
    "NOT_APPLICABLE_YET",
    "FAILED",
]


class A2AResult(BaseModel):
    """Result of one agent delegating verification to another."""

    success: bool = Field(..., description="Whether the delegation completed")
    # One legal value today: the SEC -> News direction was removed after it
    # fired 0 times in 496 runs. Kept as a field, not hardcoded downstream, so
    # a future direction is a contract change rather than a consumer rewrite.
    direction: Literal["news_to_sec"] = Field(
        "news_to_sec", description="Who asked whom"
    )
    source_agent: str = Field(..., description="Agent that delegated")
    target_agent: str = Field(..., description="Agent that was asked")

    status: A2AStatus = Field(
        A2A_FAILED,
        description="Audit-facing outcome; only CONTRADICTS is a source conflict",
    )
    verdict: str = Field(
        "NOT_ENOUGH_INFO", description="Target agent's verdict: SUPPORTS/REFUTES/NOT_ENOUGH_INFO"
    )
    confidence: float = Field(0.0, description="Target agent's confidence, 0.0-1.0")
    reasoning: str = Field("", description="Target agent's reasoning")

    # Both numbers are kept so a reviewer can redo the comparison by hand. Without
    # claimed_value the retrieved figure is unfalsifiable in the audit trail.
    retrieved_value: Optional[float] = Field(
        None, description="Value the target agent found in its own sources"
    )
    claimed_value: Optional[float] = Field(
        None, description="Value asserted by the claim under review"
    )

    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Where the evidence came from: filing type, period, section, query",
    )
    provenance: List[Dict[str, Any]] = Field(
        default_factory=list, description="The target agent's own tool provenance"
    )
    tools_used: List[str] = Field(default_factory=list)

    # "agent" = the model chose to delegate. "policy" = the node did, because the
    # claim met a rule. The trigger is deterministic; the agent it invokes is not.
    trigger_mode: Literal["agent", "policy"] = Field(
        "agent", description="What caused the delegation"
    )

    finding: str = Field("", description="What the source agent asked about")
    # The claim's original metric. Kept here because it cannot ride on the
    # delegated ParsedClaim: fine_amount and settlement_amount are *news*
    # metrics and ParsedClaim rejects them on a sec-typed claim. The audit trail
    # still needs to record what kind of assertion was being checked.
    metric: str = Field("", description="Metric of the claim under review")
    error: Optional[str] = Field(None)


def classify_status(
    claim_verdict: str,
    target_verdict: str,
    *,
    applicable: bool = True,
) -> str:
    """Map the target agent's verdict onto an audit-facing status.

    `applicable=False` means no source could plausibly cover the event yet — a
    filing that predates it, for instance. That is reported separately because
    silence from a document written before the event is not evidence of absence.
    """
    if not applicable:
        return A2A_NOT_APPLICABLE_YET
    decisive = {"SUPPORTS", "REFUTES"}
    if claim_verdict not in decisive or target_verdict not in decisive:
        return A2A_NO_MATCHING_DISCLOSURE
    return A2A_CORROBORATES if claim_verdict == target_verdict else A2A_CONTRADICTS
