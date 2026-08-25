"""The contract for one-hop bounded agent delegation.

FinVet has exactly one delegation: the News agent asks the SEC agent whether an
issuer's own filing corroborates a reported fine or settlement. It runs in that
direction only, in-process, one hop deep. This is **not** a network A2A
protocol and no interoperability with external agents is implied — the SEC
agent simply holds no delegation tool, which is what makes the call terminate
by construction rather than by a guard someone could forget.

`status` is the audit-facing outcome and is deliberately richer than the raw
verdict, because "the filing does not mention this", "no filing could yet cover
this event", and "the nested agent failed" are three different facts and none
of them is a contradiction.

Classification is **not** performed here. A status describes the relationship
between the parent claim's verdict and the target's, and the tool runs before
the parent verdict exists — comparing the target's verdict with itself is what
recorded contradictions as agreement. `reclassify_corroboration` below is the
only place status is decided, and `run_news_agent` is its only caller.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..config.constants import CORROBORATION_METRICS

# Audit-facing outcome. Kept separate from `verdict` because a verdict answers
# "what did the target agent conclude" while status answers "what does that mean
# for the claim under review" — and silence is not contradiction.
A2A_CORROBORATES = "CORROBORATES"
A2A_CONTRADICTS = "CONTRADICTS"
A2A_NO_MATCHING_DISCLOSURE = "NO_MATCHING_DISCLOSURE"
A2A_NOT_APPLICABLE_YET = "NOT_APPLICABLE_YET"
A2A_FAILED = "FAILED"

# A narrower, more serious case of NO_MATCHING_DISCLOSURE: the claim asserts a
# material amount the issuer would have had to disclose, a filing exists that
# covers the period, and that filing does not mention it. Generic silence is
# uninformative -- a periodic report omits most things -- but silence about a
# fine at an identifiable issuer, in a filing that could have carried it, is
# worth a person's attention.
#
# This exists because neither side of the delegation can be decisive on these
# metrics: a fine amount is a narrative fact with no XBRL concept, so after the
# trusted-observation boundary both the news claim and the filing check resolve
# to NOT_ENOUGH_INFO. Escalating on disagreement is therefore unreachable;
# escalating on unsupported assertion is not.
A2A_UNDISCLOSED_MATERIAL_CLAIM = "UNDISCLOSED_MATERIAL_CLAIM"

A2AStatus = Literal[
    "CORROBORATES",
    "CONTRADICTS",
    "NO_MATCHING_DISCLOSURE",
    "UNDISCLOSED_MATERIAL_CLAIM",
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

    # Whether the event date was known well enough to apply the temporal gate.
    # "unknown" means the filing's silence was not weighed against the event's
    # timing, so NO_MATCHING_DISCLOSURE from this run is weaker than it looks.
    temporal_scope: Literal["checked", "unknown"] = Field(
        "unknown", description="Whether the event date could be compared to the filing"
    )
    error: Optional[str] = Field(None)


def reclassify_corroboration(
    parent_verdict: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide the delegation's status now that the parent verdict exists.

    The tool cannot do this. It runs inside the News agent's ReAct loop, before
    that agent has reached a verdict, so it has only the target's answer. Left
    to classify, it compared the SEC verdict with itself -- which is agreement
    for any decisive verdict -- and a filing that flatly contradicted the news
    was recorded as CORROBORATES, suppressing the escalation this contract
    exists to trigger.

    Applied to both trigger paths so the model-invoked and policy-invoked
    results cannot diverge again.
    """
    updated = dict(result)

    # A delegation that did not complete has no opinion to compare against.
    if not updated.get("success"):
        updated["status"] = A2A_FAILED
        return updated

    # A filing that could not yet cover the event stays out of scope; that is a
    # fact about the calendar, not about the claim.
    if updated.get("status") == A2A_NOT_APPLICABLE_YET:
        return updated

    status = classify_status(
        parent_verdict,
        updated.get("verdict", "NOT_ENOUGH_INFO"),
    )

    # Promote plain silence to the escalating status when the claim asserted a
    # material amount and a filing that could have covered it said nothing.
    # temporal_scope guards the obvious false positive: a filing that closed
    # before the event was never going to mention it.
    if (status == A2A_NO_MATCHING_DISCLOSURE
            and updated.get("metric") in CORROBORATION_METRICS
            and updated.get("claimed_value") is not None
            and updated.get("temporal_scope") == "checked"):
        status = A2A_UNDISCLOSED_MATERIAL_CLAIM

    updated["status"] = status
    return updated


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
