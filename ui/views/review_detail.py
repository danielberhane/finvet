"""Single review detail + HITL submission page."""

import time

import streamlit as st

from api_client import reconcile_review, submit_review
from components.evidence import render_evidence
from components.formatting import _escape, humanize, verdict_label


def render_review_detail():
    """Render the review detail page for a selected pending review."""
    review_data = st.session_state.get('selected_review_data', {})
    request_id = st.session_state.selected_review_id
    claim = review_data.get("claim", "")
    preliminary = review_data.get("preliminary_analysis", {})
    hitl_triggers = review_data.get("hitl_triggers", [])

    # Back button
    if st.button("← Back to Pending Reviews"):
        st.session_state.selected_review_id = None
        st.session_state.selected_review_data = None
        st.rerun()

    st.markdown("## Review Claim")

    # HITL Header Panel
    st.markdown("""
    <div class="hitl-panel">
        <div class="hitl-header">Human Review Required</div>
        <p style="color: #92400E; margin: 0; font-size: 0.95rem;">
            The AI agents need human verification before issuing a final verdict.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Claim Under Review
    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.25rem; margin: 0.75rem 0;">
        <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #94a3b8; margin-bottom: 0.35rem;">Claim Under Review</div>
        <div style="font-size: 1rem; color: #1E3A5F; line-height: 1.6;">{_escape(claim)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Preliminary Verdict Panel
    prelim_verdict = preliminary.get("verdict")
    prelim_confidence = preliminary.get("confidence", 0)
    prelim_agent = preliminary.get("agent", "unknown")

    if prelim_verdict:
        if prelim_verdict == "SUPPORTS":
            verdict_class = "verdict-supports"
        elif prelim_verdict == "REFUTES":
            verdict_class = "verdict-refutes"
        else:
            verdict_class = "verdict-nei"

        agent_display = {"sec": "SEC", "market": "Market", "news": "News"}.get(prelim_agent, prelim_agent.upper())

        # Why triggered + Preliminary verdict in one banner
        trigger_text = ""
        if hitl_triggers:
            trigger_text = ", ".join(humanize(t) for t in hitl_triggers)

        # The deterministic layer's disagreement with the model is decisive
        # context for the reviewer, so it sits in the banner, not the evidence.
        override_text = ""
        if preliminary.get("override_applied"):
            original = preliminary.get("llm_original_verdict")
            override_text = (
                f"Model concluded {original}; deterministic numeric check overruled it"
                if original
                else "Deterministic numeric check overruled the model verdict"
            )

        # Built as one string rather than two conditional template lines: an
        # unused conditional leaves a whitespace-only line, which ends the HTML
        # block early in st.markdown.
        _sub = 'font-size: 0.82rem; opacity: 0.85; margin-top: 0.25rem;'
        sub_lines = []
        if trigger_text:
            sub_lines.append(f'<div style="{_sub}">Trigger: {trigger_text}</div>')
        if override_text:
            sub_lines.append(f'<div style="{_sub}">{override_text}</div>')
        sub_html = "".join(sub_lines)

        st.markdown(f"""
        <div class="{verdict_class}" style="display: flex; align-items: center; justify-content: space-between; text-align: left; padding: 1.25rem 1.5rem;">
            <div>
                <div style="font-size: 0.72rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.15rem;">AI Preliminary Assessment</div>
                <div class="verdict-text" style="font-size: 1.3rem;">Preliminary: {verdict_label(prelim_verdict)}</div>{sub_html}
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: 800; letter-spacing: -1px;">{prelim_confidence:.0%}</div>
                <div style="font-size: 0.75rem; opacity: 0.8;">{agent_display} Agent</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evidence & Analysis
    render_evidence(
        reasoning=preliminary.get("reasoning"),
        source_description=preliminary.get("source_description"),
        tools_called=preliminary.get("tools_called", []),
        tool_calls_detail=preliminary.get("tool_calls_detail", []),
    )

    st.markdown("---")

    # A row whose graph already ran and whose decision was already made needs
    # its audit write retried, not a second opinion. Offering the decision form
    # here would invite a reviewer to decide again on a claim that has already
    # been decided.
    if review_data.get("review_status") == "finalization_failed":
        _render_finalization_retry(request_id)
        return

    # Submit Your Review Section
    st.markdown('<div class="section-header">Submit Your Review</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        decision = st.radio(
            "Your decision:",
            options=["approve", "override", "reject"],
            format_func=lambda x: {
                "approve": "✅ Approve AI Verdict",
                "override": "✏️ Override Verdict",
                "reject": "❌ Reject Claim"
            }[x],
            horizontal=True,
            label_visibility="collapsed"
        )

    with col2:
        if decision == "override":
            override_verdict = st.selectbox(
                "Select verdict:",
                options=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                format_func=verdict_label,
                label_visibility="collapsed"
            )
        else:
            override_verdict = None

    reviewer_notes = st.text_area(
        "Reviewer Notes:",
        placeholder="Add context for your decision (e.g., why you're overriding, additional research done)...",
        height=100,
        label_visibility="collapsed"
    )

    if st.button("Submit Review", type="primary"):
        review_payload = {
            "decision": decision,
            "override_verdict": override_verdict,
            "reviewer_notes": reviewer_notes
        }
        result, error = submit_review(request_id, review_payload)
        if result:
            st.success(f"✅ Review submitted. Final verdict: **{result.get('verdict')}**")
            time.sleep(1.5)
            st.session_state.selected_review_id = None
            st.session_state.selected_review_data = None
            st.rerun()
        else:
            st.error(error)

    # Request ID for reference
    st.caption(f"Request ID: `{request_id}`")


def _render_finalization_retry(request_id):
    """The action a stuck review actually needs.

    Built with list-append and one join: st.markdown breaks on blank lines
    inside an HTML block.
    """
    panel = [
        '<div class="hitl-panel">',
        '<div class="hitl-header">Audit finalization failed</div>',
        '<p style="color: #92400E; margin: 0; font-size: 0.95rem;">',
        'This review was submitted and the verification finished, but the '
        'result could not be written to the audit trail. The decision still '
        'stands &mdash; retrying records it. No new decision is needed.',
        '</p>',
        '</div>',
    ]
    st.markdown("\n".join(panel), unsafe_allow_html=True)

    st.caption(
        "Reviews are held in memory only, so a restart of the API since the "
        "review was submitted will have discarded the result. In that case "
        "the retry reports that the checkpoint is gone and the claim must be "
        "re-run."
    )

    if st.button("Retry audit finalization", type="primary"):
        result, error = reconcile_review(request_id)
        if result:
            st.success(
                f"✅ Recorded. Final verdict: **{result.get('verdict')}**")
            time.sleep(1.5)
            st.session_state.selected_review_id = None
            st.session_state.selected_review_data = None
            st.rerun()
        else:
            st.error(error)

    st.caption(f"Request ID: `{request_id}`")
