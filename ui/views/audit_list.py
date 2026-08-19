"""Audit fleet view — stats bar, filters, and execution card list."""

from datetime import datetime

import streamlit as st

from api_client import list_audit_executions
from components.formatting import _escape
from components.source_badges import _data_source_badges_html


_VERDICT_OPTIONS = ["", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "PENDING", "REJECTED"]
_AGENT_OPTIONS   = ["", "sec", "market", "news"]
_SOURCE_OPTIONS  = ["", "xbrl", "rag", "a2a"]

_VERDICT_BADGE = {
    "SUPPORTS":        "badge-supports",
    "REFUTES":         "badge-refutes",
    "NOT_ENOUGH_INFO": "badge-nei",
    "PENDING":         "badge-nei",
    "REJECTED":        "badge-refutes",
}


def _compute_stats(items: list) -> dict:
    """Compute summary statistics from a list of execution dicts."""
    total = len(items)
    if total == 0:
        return {"total": 0, "supports_pct": 0.0, "avg_confidence": 0.0, "avg_time_s": 0.0}

    verdicts = [item.get("verdict") for item in items]
    supports_pct = sum(1 for v in verdicts if v == "SUPPORTS") / total

    confidences = [item["confidence"] for item in items if item.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    times = [item["execution_time_ms"] for item in items if item.get("execution_time_ms") is not None]
    avg_time_s = (sum(times) / len(times) / 1000) if times else 0.0

    return {
        "total": total,
        "supports_pct": supports_pct,
        "avg_confidence": avg_confidence,
        "avg_time_s": avg_time_s,
    }


def render_audit_list():
    """Render the audit fleet view."""
    st.markdown("## Audit Trail")

    # ── Filters ───────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            f_verdict = st.selectbox(
                "Verdict", _VERDICT_OPTIONS,
                format_func=lambda x: "All verdicts" if x == "" else x,
                key="audit_filter_verdict",
            )
        with f_col2:
            f_agent = st.selectbox(
                "Agent", _AGENT_OPTIONS,
                format_func=lambda x: "All agents" if x == "" else x.upper(),
                key="audit_filter_agent",
            )
        with f_col3:
            f_source = st.selectbox(
                "Data source", _SOURCE_OPTIONS,
                format_func=lambda x: "All sources" if x == "" else x.upper(),
                key="audit_filter_source",
            )
        with f_col4:
            f_limit = st.select_slider(
                "Max results", options=[25, 50, 100, 200],
                value=50, key="audit_filter_limit",
            )

    # ── Fetch ──────────────────────────────────────────────────────
    data, error = list_audit_executions(
        verdict=f_verdict or None,
        agent=f_agent or None,
        data_source=f_source or None,
        limit=f_limit,
    )

    if error:
        st.error(error)
        return

    items = (data or {}).get("items", [])

    # ── Stats bar ─────────────────────────────────────────────────
    stats = _compute_stats(items)
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Total Verifications", stats["total"])
    s_col2.metric("Supports Rate", f"{stats['supports_pct']:.0%}")
    s_col3.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}")
    s_col4.metric("Avg Execution Time", f"{stats['avg_time_s']:.1f}s")

    st.markdown("---")

    # ── Empty state ───────────────────────────────────────────────
    if not items:
        parts = [
            '<div class="empty-state">',
            '<div class="empty-state-icon">🔍</div>',
            '<h3 style="color: #64748b; font-weight: 600;">No records found</h3>',
            '<p>Try adjusting your filters or run a verification first.</p>',
            '</div>',
        ]
        st.markdown("\n".join(parts), unsafe_allow_html=True)
        return

    # ── Execution list ────────────────────────────────────────────
    for item in items:
        request_id   = item.get("request_id", "")
        claim_text   = item.get("claim_text", "")
        verdict      = item.get("verdict") or "UNKNOWN"
        confidence   = item.get("confidence") or 0.0
        exec_ms      = item.get("execution_time_ms") or 0
        agents_run   = item.get("agents_run") or []
        data_sources = item.get("data_sources") or {}
        timestamp    = item.get("timestamp", "")

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%b %d, %Y %H:%M UTC")
        except Exception:
            time_str = timestamp

        badge_class   = _VERDICT_BADGE.get(verdict, "badge-nei")
        agent_display = "/".join(
            {"sec": "SEC", "market": "Market", "news": "News"}.get(a, a.upper())
            for a in agents_run
        ) or "—"
        exec_s = f"{exec_ms / 1000:.1f}s" if exec_ms else "—"
        badges_html = _data_source_badges_html({"data_sources": data_sources})

        with st.container():
            card_parts = [
                '<div class="audit-card">',
                f'<div class="audit-card-claim">"{_escape(claim_text)}"</div>',
                '<div class="audit-card-meta">',
                f'<span class="badge {badge_class}">{_escape(verdict)}</span>',
                f'<span>{confidence:.0%} confidence</span>',
                f'<span>{_escape(agent_display)} Agent</span>',
                f'<span>{exec_s}</span>',
                f'<span>{badges_html}</span>',
                f'<span style="margin-left:auto;">{_escape(time_str)}</span>',
                '</div>',
                '</div>',
            ]
            st.markdown("\n".join(card_parts), unsafe_allow_html=True)

            if st.button("View audit trail →", key=f"audit_{request_id}", use_container_width=True):
                st.session_state.selected_audit_id = request_id
                st.rerun()
