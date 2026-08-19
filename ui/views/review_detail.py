"""Single review detail + HITL submission page."""

import time

import streamlit as st

from api_client import submit_review
from components.evidence import render_evidence
from components.formatting import _escape


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
        conf_color = "#10B981" if prelim_confidence >= 0.7 else "#F59E0B" if prelim_confidence >= 0.5 else "#EF4444"

        # Why triggered + Preliminary verdict in one banner
        trigger_text = ""
        if hitl_triggers:
            trigger_text = ", ".join([t.replace("_", " ").title() for t in hitl_triggers])

        st.markdown(f"""
        <div class="{verdict_class}" style="display: flex; align-items: center; justify-content: space-between; text-align: left; padding: 1.25rem 1.5rem;">
            <div>
                <div style="font-size: 0.72rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.15rem;">AI Preliminary Assessment</div>
                <div class="verdict-text" style="font-size: 1.3rem;">Preliminary: {prelim_verdict}</div>
                {f'<div style="font-size: 0.82rem; opacity: 0.85; margin-top: 0.25rem;">Trigger: {trigger_text}</div>' if trigger_text else ''}
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
