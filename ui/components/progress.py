"""The live ledger shown while a claim is being verified.

A run took ten or twenty seconds and displayed a rotating caption — "Resolving
time period…" — that named a stage and said nothing about the reader's claim.
The stage was itself a guess: the view keyword-matched the raw claim text to
choose the caption, so "Apple announced revenue of $391 billion" was captioned
"Searching news sources…" while the SEC agent ran.

The server had the real answers all along and discarded them. Now each step
leaves the artifact it produced, and the step that matters most — the
deterministic comparison — is shown forming. When Python overrules the model
that is the whole argument of this system, and it used to happen silently.

Two module conventions are followed here: a pure shaper (`progress_rows`)
separate from an HTML builder (`progress_html`), as `parsed_claim_rows` and
`source_badges` do; and HTML assembled by list-append and join, never a
template with embedded newlines, because a blank line ends an `st.markdown`
block early.
"""

from components.formatting import _escape, format_value, humanize, verdict_label

# Prose for the nodes a reader will actually see. Anything unmapped falls back
# to `humanize`, so a node added server-side reads imperfectly rather than
# printing `some_new_node` on the page.
_STEP_LABELS = {
    "input_guardrails": "Checked the claim is safe to run",
    "claim_parser": "Read the claim",
    "period_resolver": "Resolved the period",
    "sec_agent": "Read the filings",
    "market_agent": "Fetched the market quote",
    "news_agent": "Searched the news",
    "reject_handler": "Declined the claim",
    "consensus": "Weighed the evidence",
    "output_guardrails": "Checked the answer",
    "hitl_checkpoint": "Queued for review",
    "apply_hitl_decision": "Applied the review decision",
    "response_generator": "Wrote the response",
}

_AGENT_NODES = ("sec_agent", "market_agent", "news_agent")


def _joined(parts):
    return "  ·  ".join(p for p in parts if p)


def _parse_detail(d):
    """How the claim was read — the thing the caption used to guess at."""
    parts = []
    if d.get("ticker"):
        parts.append(str(d["ticker"]))
    metric = d.get("metric")
    if metric:
        parts.append(str(metric))
    elif d.get("claim_type") == "sec":
        # Not a gap: a null metric is the decision that routes the claim to
        # filing text rather than XBRL.
        parts.append("no metric — routed to filing text")
    if d.get("value") is not None:
        parts.append(format_value(d["value"]))
    if d.get("period"):
        parts.append(str(d["period"]))
    return _joined(parts)


def _agent_detail(d, claimed_value):
    """What the agent retrieved, and what the comparison made of it."""
    parts = []
    tools = d.get("tools") or []
    if tools:
        parts.append(f"{len(tools)} tool call{'s' if len(tools) != 1 else ''}")
    if d.get("rag_chunks"):
        parts.append(f"{d['rag_chunks']} filing passages")
    if d.get("a2a_status"):
        parts.append(f"asked the SEC agent — {humanize(d['a2a_status'])}")

    retrieved = d.get("retrieved_value")
    if retrieved is not None:
        found = format_value(retrieved)
        if d.get("concept"):
            found += f" ({d['concept']})"
        if claimed_value is not None:
            found = f"claimed {format_value(claimed_value)}  ·  found {found}"
        parts.append(found)

    diff = d.get("magnitude_difference_percent")
    if diff is not None:
        parts.append(f"{diff:.2f}% apart")

    if d.get("limitation"):
        parts.append(humanize(d["limitation"]))

    detail = _joined(parts)

    # The moment this whole component exists for. Only say the comparison
    # decided otherwise when it visibly did: on the delegation path
    # `override_applied` is set against the news agent's own declined verdict,
    # not against the answer on screen, so the two can agree while the flag is
    # true. Saying "decided otherwise" beside a matching verdict reads as a
    # contradiction in the one place the project cannot afford one.
    original = d.get("llm_original_verdict")
    if d.get("override_applied") and original and original != d.get("verdict"):
        detail += (f" — the model said {verdict_label(original)};"
                   f" the comparison decided otherwise")
    return detail


def _detail_for(node, d, claimed_value):
    if node == "claim_parser":
        return _parse_detail(d)
    if node == "period_resolver":
        window = _joined([d.get("start"), d.get("end")]).replace("  ·  ", " to ")
        return _joined([window, d.get("assumption")])
    if node in _AGENT_NODES:
        return _agent_detail(d, claimed_value)
    if node == "consensus":
        verdict = verdict_label(d.get("verdict")) if d.get("verdict") else ""
        confidence = (f"{d['confidence']:.0%} confident"
                      if d.get("confidence") is not None else "")
        return _joined([verdict, confidence])
    if node == "output_guardrails":
        triggers = d.get("hitl_triggers") or []
        if triggers:
            return "needs review — " + ", ".join(humanize(t) for t in triggers)
        return "no review needed"
    return ""


def progress_rows(events, finished=False):
    """One row per node that reported, in the order they ran.

    The last row is `running` until the stream finishes, because a node
    reports when it *completes* — so the arrival of one step's result is the
    signal that the next has begun.
    """
    rows = []
    claimed_value = None

    for event in events or []:
        if event.get("type") != "progress":
            continue
        node = event.get("node") or ""
        detail = event.get("detail") or {}

        # Carried forward: the agent's evidence has the retrieved value but not
        # the claimed one, so the comparison is only legible by joining the two.
        if node == "claim_parser" and detail.get("value") is not None:
            claimed_value = detail["value"]

        rows.append({
            "node": node,
            "label": _STEP_LABELS.get(node) or humanize(node),
            "detail": _detail_for(node, detail, claimed_value),
            "state": "done",
        })

    if rows and not finished:
        rows[-1]["state"] = "running"
    return rows


def progress_html(rows, claim=None, stopped=False):
    """The ledger as one HTML blob, reusing the audit view's timeline.

    Carries its own heading because it no longer sits inside `st.status`: that
    widget auto-collapsed itself the moment a run completed, closing the ledger
    at exactly the moment a reader wanted it, and no `expanded=True` overrides
    that. Without the widget there is no chrome to hold the claim text.

    The heading also carries the heartbeat. Between steps an agent can think
    for five to ten seconds, and a ledger that sits perfectly still reads as
    stalled rather than working. Three dots animate their opacity on a
    stagger -- pure CSS, because a Streamlit rerun would restart the stream.

    Rendered with no rows at all when a claim is given: `progress_html([])`
    returned "", so between the click and the first event there was nothing on
    screen, which is the moment a reader most needs to see that something
    started.

    `stopped` is the state a failed run settles into. Without it the last frame
    of a run that died would keep animating, claiming to work.

    The trail stays in this slot when the run ends rather than being cleared and
    re-rendered under the verdict: content a reader has been watching for
    fifteen seconds should not relocate. The claim text also appears *only*
    here -- the result panel renders verdict, summary and confidence, never the
    claim -- so moving it below the verdict put the claim after the answer.
    """
    running = bool(rows) and any(r["state"] == "running" for r in rows)
    if not rows and claim:
        running = not stopped

    if not rows and not claim:
        return ""

    parts = []
    if claim:
        if stopped:
            heading = "Stopped"
        elif running:
            heading = ("Verifying"
                       '<span class="verifying-dots">'
                       '<span class="verifying-dot-1"></span>'
                       '<span class="verifying-dot-2"></span>'
                       '<span class="verifying-dot-3"></span>'
                       "</span>")
        else:
            # Not "Verified": this heading sits above the verdict, so on a
            # refuted claim it would read as the verdict itself. "Checked" is
            # true whether the claim was supported, refuted or declined.
            heading = "Checked"
        parts.append(f'<div class="report-title">{heading}</div>')
        parts.append(f'<div class="report-source">{_escape(str(claim))}</div>')

    if rows:
        parts.append('<div class="pipeline-timeline">')
        for row in rows:
            dot = ("pipeline-step-dot-running"
                   if row["state"] == "running" and not stopped
                   else "pipeline-step-dot-ok")
            parts.append('<div class="pipeline-step">')
            parts.append(f'<div class="pipeline-step-dot {dot}"></div>')
            parts.append('<div class="pipeline-step-body">')
            parts.append(
                f'<div class="pipeline-step-label">{_escape(row["label"])}</div>')
            if row["detail"]:
                parts.append(
                    f'<div class="pipeline-step-meta">{_escape(row["detail"])}</div>')
            parts.append('</div>')
            parts.append('</div>')
        parts.append('</div>')
    return "\n".join(parts)
