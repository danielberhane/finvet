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

# Audit-facing outcome. Kept separate from `verdict` because a verdict answers
# "what did the target agent conclude" while status answers "what does that mean
# for the claim under review" — and silence is not contradiction.
#
# Declared before the constants so each can be annotated with it: bare string
# constants are not narrowed to the Literal, which made `Field(A2A_FAILED, ...)`
# a type error on the status field below.
A2AStatus = Literal[
    "PENDING_CLASSIFICATION",
    "CORROBORATES",
    "CONTRADICTS",
    "NO_MATCHING_DISCLOSURE",
    "NOT_APPLICABLE_YET",
    "SOURCE_UNAVAILABLE",
    "NO_CORPUS",
    "FAILED",
]

# The tool runs inside the News agent's ReAct loop, before that agent has a
# verdict, so it has nothing to classify against. This is what it carries until
# reclassify_corroboration decides. It used to borrow NO_MATCHING_DISCLOSURE,
# which is a real outcome -- a result that never reached reclassification read
# as a filing that said nothing.
A2A_PENDING_CLASSIFICATION: A2AStatus = "PENDING_CLASSIFICATION"

A2A_CORROBORATES: A2AStatus = "CORROBORATES"
A2A_CONTRADICTS: A2AStatus = "CONTRADICTS"

# An applicable filing was identified and successfully searched, and it does not
# mention the claim. Only reachable when a search actually succeeded: see
# summarize_filing_search.
A2A_NO_MATCHING_DISCLOSURE: A2AStatus = "NO_MATCHING_DISCLOSURE"

A2A_NOT_APPLICABLE_YET: A2AStatus = "NOT_APPLICABLE_YET"

# The distinction that makes silence meaningful. Nothing was read, so nothing is
# known about what the filing says.
A2A_SOURCE_UNAVAILABLE: A2AStatus = "SOURCE_UNAVAILABLE"
A2A_NO_CORPUS: A2AStatus = "NO_CORPUS"

A2A_FAILED: A2AStatus = "FAILED"


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

    # How the temporal gate was applied, which decides whether the filing's
    # silence means anything.
    #   "event_date"   - the caller supplied a real date for the event
    #   "claim_period" - inferred from the claim's resolved period (its start,
    #                    the earliest moment the event could have occurred)
    #   "unknown"      - no date at all; silence was never weighed against
    #                    timing, so it is weaker evidence than it looks
    # Recorded rather than collapsed to a boolean because an escalation sends
    # work to a person, and they should know whether the date was stated or
    # inferred.
    temporal_scope: Literal["event_date", "claim_period", "unknown"] = Field(
        "unknown", description="How the event date was established"
    )
    error: Optional[str] = Field(None)


def summarize_filing_search(
    provenance: Optional[List[Dict[str, Any]]],
) -> Dict[str, bool]:
    """What the nested agent's filing searches actually established.

    Whether a filing was read is a different question from what the nested
    agent concluded, and only the first licenses the sentence "the issuer's
    filing does not mention this". `search_filing_text` reports its own
    outcome, so this reads that rather than inferring from the verdict.
    """
    summary = {"searched": False, "unavailable": False, "failed": False,
               "no_corpus": False}

    for entry in provenance or []:
        if not isinstance(entry, dict) or entry.get("tool") != "search_filing_text":
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue

        if result.get("success"):
            summary["searched"] = True
            if result.get("reason") == "no_corpus":
                summary["no_corpus"] = True
            continue

        # A failure to reach the source is not a failure of the source to say
        # anything. Which kind matters to a reader deciding whether to re-run.
        error = str(result.get("error") or "").lower()
        if "not available" in error or "unavailable" in error:
            summary["unavailable"] = True
        else:
            summary["failed"] = True

    return summary


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

    # Two decisive verdicts can be compared however each was reached, so the
    # comparison stands on its own. Everything else resolves to "the filing
    # does not mention this" -- a claim about a document, which may only be
    # made if a document was actually read.
    if status == A2A_NO_MATCHING_DISCLOSURE:
        search = summarize_filing_search(updated.get("provenance"))
        if not search["searched"]:
            # Nothing was successfully searched. The delegation completed, but
            # it established nothing about the filing's contents, and calling
            # that silence turns an absence of evidence into evidence of
            # absence.
            status = A2A_FAILED if search["failed"] else A2A_SOURCE_UNAVAILABLE
        elif search["no_corpus"]:
            # There was no filing to be silent.
            status = A2A_NO_CORPUS

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
