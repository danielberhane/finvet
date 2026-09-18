"""Audit detail view — pipeline timeline, evidence, and integrity checksum."""

from datetime import datetime

import streamlit as st

import json

import pandas as pd

from api_client import get_audit_detail
from components.evidence import render_evidence
from components.formatting import _escape, humanize, verdict_label
from components.source_badges import _data_source_badges_html


_PIPELINE_ORDER = [
    ("input_guardrails",    "Input Guardrails"),
    ("claim_parser",        "Claim Parser"),
    ("period_resolver",     "Period Resolver"),
    ("sec_agent",           "SEC Agent"),
    ("market_agent",        "Market Agent"),
    ("news_agent",          "News Agent"),
    ("reject_handler",      "Reject Handler"),
    ("confidence_adjuster", "Confidence"),
    ("output_guardrails",   "Output Guardrails"),
    ("hitl_checkpoint",     "HITL Checkpoint"),
    ("apply_hitl_decision", "Apply HITL Decision"),
    ("response_generator",  "Response Generator"),
]


def _build_timeline(events: list) -> list:
    """Build ordered pipeline steps from raw audit events.

    Only includes nodes that have at least one event (i.e., nodes that ran).
    Returns list of dicts: {node, label, status, duration_ms, error}
    """
    node_data = {}
    for event in events:
        event_type = event.get("event_type", "")
        data = event.get("data") or {}
        node = data.get("node")
        if not node:
            continue
        if node not in node_data:
            node_data[node] = {"status": "running", "duration_ms": None, "error": None}
        if event_type == "node_completed":
            node_data[node]["status"] = "ok"
            node_data[node]["duration_ms"] = data.get("duration_ms")
        elif event_type == "node_error":
            node_data[node]["status"] = "error"
            node_data[node]["error"] = data.get("error", "Unknown error")

    steps = []
    for node_key, label in _PIPELINE_ORDER:
        if node_key in node_data:
            steps.append({"node": node_key, "label": label, **node_data[node_key]})
    return steps


def _format_duration(ms):
    """Format milliseconds as human-readable string. Returns empty string for None."""
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def render_audit_detail():
    """Render the full audit detail view for st.session_state.selected_audit_id."""
    request_id = st.session_state.get("selected_audit_id")
    if not request_id:
        return

    if st.button("← Back to Audit Trail"):
        st.session_state.selected_audit_id = None
        st.rerun()

    data, error = get_audit_detail(request_id)
    if error:
        st.error(error)
        return

    execution = data.get("execution") or {}
    events = data.get("events") or []

    claim_text = execution.get("claim_text", "")
    verdict = execution.get("verdict") or "UNKNOWN"
    confidence = execution.get("confidence") or 0.0
    exec_ms = execution.get("execution_time_ms") or 0
    timestamp = execution.get("timestamp", "")
    agents_run = execution.get("agents_run") or []
    data_sources = execution.get("data_sources") or {}
    execution_hash = execution.get("execution_hash", "")
    integrity = data.get("integrity") or {}
    full_trace = execution.get("full_trace") or {}
    final_response = full_trace.get("final_response") or {}

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        time_str = dt.strftime("%b %d, %Y %I:%M:%S %p UTC")
    except Exception:
        time_str = timestamp

    agent_display = "/".join(
        {"sec": "SEC", "market": "Market", "news": "News"}.get(a, humanize(a))
        for a in agents_run
    ) or "—"

    # ── Header card ───────────────────────────────────────────────
    st.markdown("## Audit Detail")

    verdict_class = {
        "SUPPORTS": "verdict-supports",
        "REFUTES": "verdict-refutes",
        "PENDING": "verdict-pending",
        "REJECTED": "verdict-refutes",
    }.get(verdict, "verdict-nei")

    badges_html = _data_source_badges_html({"data_sources": data_sources})
    exec_s = f"{exec_ms / 1000:.1f}s" if exec_ms else "—"

    header_parts = [
        f'<div class="{verdict_class}" style="display:flex;align-items:center;justify-content:space-between;">',
        "<div>",
        '<div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#94A3B8;margin-bottom:0.25rem;">CLAIM UNDER AUDIT</div>',
        f'<div style="font-size:0.95rem;color:#1E3A5F;font-weight:500;">"{_escape(claim_text)}"</div>',
        f'<div style="font-size:0.78rem;color:#94A3B8;margin-top:0.4rem;">{_escape(time_str)} &middot; {_escape(agent_display)} Agent &middot; {exec_s} {badges_html}</div>',
        "</div>",
        '<div style="text-align:right;min-width:120px;">',
        f'<div class="verdict-text">{_escape(verdict_label(verdict))}</div>',
        f'<div class="verdict-confidence" style="font-size:2rem;font-weight:800;">{confidence:.0%}</div>',
        "</div>",
        "</div>",
    ]
    st.markdown("\n".join(header_parts), unsafe_allow_html=True)

    # ── Pipeline Timeline ─────────────────────────────────────────
    st.markdown("### Pipeline Timeline")

    steps = _build_timeline(events)
    if steps:
        dot_class_map = {
            "ok":      "pipeline-step-dot-ok",
            "error":   "pipeline-step-dot-err",
            "warn":    "pipeline-step-dot-warn",
            "running": "pipeline-step-dot-skip",
        }
        step_parts = ['<div class="pipeline-timeline">']
        for step in steps:
            dot_cls = dot_class_map.get(step["status"], "pipeline-step-dot-skip")
            duration_str = _format_duration(step["duration_ms"])
            error_str = f" — {_escape(step['error'])}" if step.get("error") else ""
            meta = f"{duration_str}{error_str}" if (duration_str or error_str) else "&nbsp;"
            step_parts += [
                '<div class="pipeline-step">',
                f'<div class="pipeline-step-dot {dot_cls}"></div>',
                '<div class="pipeline-step-body">',
                f'<div class="pipeline-step-label">{_escape(step["label"])}</div>',
                f'<div class="pipeline-step-meta">{meta}</div>',
                "</div>",
                "</div>",
            ]
        step_parts.append("</div>")
        st.markdown("\n".join(step_parts), unsafe_allow_html=True)
    else:
        st.caption("No pipeline events recorded for this execution.")

    # ── Agent Evidence ────────────────────────────────────────────
    metadata = final_response.get("metadata") or {}
    explanation = final_response.get("explanation", "")
    if explanation or metadata.get("tools_called"):
        st.markdown("### Agent Analysis")
        render_evidence(
            reasoning=explanation,
            source_description=metadata.get("source_description", ""),
            tools_called=metadata.get("tools_called", []),
            tool_calls_detail=metadata.get("tool_calls_detail", []),
            data_sources=metadata,
            # Already shown in the header above.
            show_badges=False,
        )

    # ── HITL Section ──────────────────────────────────────────────
    hitl_meta = metadata.get("hitl") or {}
    if hitl_meta.get("applied"):
        decision = hitl_meta.get("decision", "")
        notes = hitl_meta.get("reviewer_notes", "")
        override = hitl_meta.get("override_verdict", "")
        override_html = f' &rarr; Override: <strong>{_escape(verdict_label(override))}</strong>' if override else ""
        hitl_parts = [
            '<div class="hitl-panel" style="margin-top:1rem;padding:1rem 1.25rem;border-radius:8px;">',
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;color:#92400E;margin-bottom:0.4rem;">Human Review Applied</div>',
            f'<div style="font-size:0.88rem;color:#78350F;">Decision: <strong>{_escape(humanize(decision))}</strong>{override_html}</div>',
        ]
        if notes:
            hitl_parts.append(f'<div style="font-size:0.82rem;color:#92400E;margin-top:0.3rem;">Notes: {_escape(notes)}</div>')
        hitl_parts.append("</div>")
        st.markdown("\n".join(hitl_parts), unsafe_allow_html=True)

    # ── Integrity checksum ────────────────────────────────────────
    # The API recomputes the checksum over the stored envelope and returns the
    # result. This renders that result; it must never infer a verified state
    # from the presence of a hash string, which is what it used to do.
    if execution_hash:
        status = (integrity.get("status") or "unavailable").lower()
        label, css = {
            "verified": ("Integrity verified for the stored snapshot",
                         "hash-verified"),
            "failed": ("Checksum does not match the stored record",
                       "hash-failed"),
        }.get(status, ("Not verifiable — no checksum on record",
                       "hash-unknown"))

        hash_parts = [
            '<div class="hash-row">',
            '<span class="hash-label">Integrity Checksum</span>',
            f'<span class="hash-value">{_escape(execution_hash)}</span>',
            f'<span class="{css}">{_escape(label)}</span>',
            "</div>",
        ]
        st.markdown("\n".join(hash_parts), unsafe_allow_html=True)

    # ── Raw Event Log ─────────────────────────────────────────────
    if events:
        with st.expander(f"Raw Event Log ({len(events)} events)", expanded=False):
            rows = [
                {
                    "Timestamp": e.get("timestamp", ""),
                    "Event Type": e.get("event_type", ""),
                    "Agent": e.get("agent") or "—",
                    "Node": (e.get("data") or {}).get("node", "—"),
                    "Duration ms": (e.get("data") or {}).get("duration_ms", ""),
                }
                for e in events
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Full Trace JSON ───────────────────────────────────────────
    with st.expander("Full Trace JSON", expanded=False):
        st.code(json.dumps(full_trace, indent=2, default=str), language="json")

    st.caption(f"Request ID: `{request_id}`")
