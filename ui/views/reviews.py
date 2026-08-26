"""Pending Reviews list page."""

from datetime import datetime

import streamlit as st

from components.formatting import stage_label, verdict_label

from api_client import get_reviews_detailed


def render_reviews():
    """Render the pending reviews list page."""
    st.markdown("## Pending Reviews")

    pending_reviews, error = get_reviews_detailed()

    if error:
        st.error(error)
        return

    if not pending_reviews:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">✅</div>
            <h3 style="color: #64748b; font-weight: 600;">All caught up!</h3>
            <p>No claims are currently awaiting human review.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**{len(pending_reviews)}** claim{'s' if len(pending_reviews) != 1 else ''} awaiting review")

    for review in pending_reviews:
        request_id = review.get("request_id", "")
        claim = review.get("claim", "")
        timestamp = review.get("timestamp", "")
        preliminary = review.get("preliminary_analysis", {})
        prelim_verdict = preliminary.get("verdict", "UNKNOWN")
        prelim_confidence = preliminary.get("confidence", 0)
        agent = preliminary.get("agent")

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%b %d, %Y %I:%M %p")
        except Exception:
            time_str = timestamp

        # Verdict badge class
        if prelim_verdict == "SUPPORTS":
            verdict_badge = "badge-supports"
        elif prelim_verdict == "REFUTES":
            verdict_badge = "badge-refutes"
        else:
            verdict_badge = "badge-nei"

        # Confidence badge
        if prelim_confidence >= 0.7:
            conf_badge = "badge-high"
            conf_label = "HIGH"
        elif prelim_confidence >= 0.5:
            conf_badge = "badge-medium"
            conf_label = "MEDIUM"
        else:
            conf_badge = "badge-low"
            conf_label = "LOW"

        # Agent display
        # "UNKNOWN" is not a thing that ran; an empty label is honest.
        agent_display = stage_label({"agent": agent}).replace(" Agent", "") or "—"

        # Create clickable card
        with st.container():
            st.markdown(f"""
            <div class="pending-card">
                <div class="pending-claim-text">"{claim}"</div>
                <div class="pending-meta">
                    <div class="pending-meta-item">
                        <span class="badge {verdict_badge}">{verdict_label(prelim_verdict)}</span>
                    </div>
                    <div class="pending-meta-item">
                        <span class="badge {conf_badge}">{conf_label}</span>
                    </div>
                    <div class="pending-meta-item">
                        <strong>Agent:</strong> {agent_display}
                    </div>
                    <div class="pending-meta-item">
                        <strong>Submitted:</strong> {time_str}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Review this claim", key=f"review_{request_id}", use_container_width=True):
                st.session_state.selected_review_id = request_id
                st.session_state.selected_review_data = review
                st.rerun()
