"""FinVet - AI-Powered Financial Claim Verification"""

import streamlit as st

from finvet import __version__

from api_client import get_pending_reviews, health_check
from components.formatting import _escape
from styles import STYLES
from views.verify import render_verify
from views.reviews import render_reviews
from views.review_detail import render_review_detail
from views.audit_list import render_audit_list
from views.audit_detail import render_audit_detail

# Page config
st.set_page_config(
    page_title="FinVet | Financial Claim Verification",
    page_icon="FV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
st.markdown(STYLES, unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'verify'
if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None
if 'session_verifications' not in st.session_state:
    st.session_state.session_verifications = 0
if 'last_verdict' not in st.session_state:
    st.session_state.last_verdict = None
if 'memory_matches' not in st.session_state:
    st.session_state.memory_matches = None
if 'memory_choice' not in st.session_state:
    st.session_state.memory_choice = None
if 'pending_claim' not in st.session_state:
    st.session_state.pending_claim = None
if 'selected_audit_id' not in st.session_state:
    st.session_state.selected_audit_id = None

# HITL redirect: set sidebar_nav BEFORE the radio widget renders
if st.session_state.get('_hitl_redirect'):
    st.session_state.sidebar_nav = 'pending'
    del st.session_state['_hitl_redirect']

# Fetch pending count + system health
api_connected, api_version = health_check()
pending_reviews_list = get_pending_reviews()
pending_count = len(pending_reviews_list)

# Sidebar Navigation
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-text">FinVet</div>
        <div class="sidebar-brand-sub">Verification Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation section
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    reviews_label = f"Pending Reviews  ({pending_count})" if pending_count > 0 else "Pending Reviews"

    # Initialize sidebar_nav from current_page if not already set
    if "sidebar_nav" not in st.session_state:
        st.session_state.sidebar_nav = st.session_state.current_page

    page = st.radio(
        "nav",
        options=["verify", "pending", "audit"],
        format_func=lambda x: {
            "verify": "Verify Claim",
            "pending": reviews_label,
            "audit": "Audit Trail",
        }[x],
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.session_state.selected_review_id = None
        st.session_state.selected_audit_id = None
        st.rerun()

    # Pending count badge
    if pending_count > 0:
        st.markdown(f"""
        <div class="pending-badge">
            <div class="pending-badge-count">{pending_count}</div>
            <div class="pending-badge-label">Awaiting Review</div>
        </div>
        """, unsafe_allow_html=True)

    # System status section
    st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)

    if api_connected:
        dot_class = "status-dot-ok"
        conn_text = "Connected"
    else:
        dot_class = "status-dot-err"
        conn_text = "Offline"

    ver_text = f"v{api_version}" if api_version else "--"
    count = st.session_state.session_verifications
    last = st.session_state.last_verdict

    if last == "SUPPORTS":
        verdict_html = '<span class="status-verdict status-verdict-supports">SUPPORTS</span>'
    elif last == "REFUTES":
        verdict_html = '<span class="status-verdict status-verdict-refutes">REFUTES</span>'
    elif last == "NOT_ENOUGH_INFO":
        verdict_html = '<span class="status-verdict status-verdict-nei">NEI</span>'
    elif last:
        verdict_html = f'<span class="status-verdict status-verdict-none">{_escape(last)}</span>'
    else:
        verdict_html = '<span class="status-verdict status-verdict-none">--</span>'

    status_parts = [
        '<div class="sidebar-status">',
        '<div class="status-row">',
        '<span class="status-label">API</span>',
        f'<span class="status-value"><span class="status-dot {dot_class}"></span>{conn_text}</span>',
        '</div>',
        '<div class="status-row">',
        '<span class="status-label">Version</span>',
        f'<span class="status-value">{ver_text}</span>',
        '</div>',
        '<div class="status-row">',
        '<span class="status-label">Session</span>',
        f'<span class="status-value">{count} verified</span>',
        '</div>',
        '<div class="status-row">',
        '<span class="status-label">Last</span>',
        f'<span class="status-value">{verdict_html}</span>',
        '</div>',
        '</div>',
    ]
    st.markdown("\n".join(status_parts), unsafe_allow_html=True)

    # Footer (version only, no model details)
    st.markdown(
        f'<div class="sidebar-footer">FinVet v{__version__}</div>',
        unsafe_allow_html=True,
    )

# Header - Large centered FinVet
st.markdown("""
<div class="main-header">
    <div class="logo-text">FinVet</div>
    <div class="tagline">AI-powered financial claim verification</div>
</div>
""", unsafe_allow_html=True)

# Page routing
# Review detail must come before Pending Reviews so st.stop() doesn't block it
if st.session_state.selected_review_id:
    render_review_detail()
    st.stop()

if st.session_state.current_page == 'pending':
    render_reviews()
    st.stop()

if st.session_state.current_page == 'audit':
    if st.session_state.get('selected_audit_id'):
        render_audit_detail()
    else:
        render_audit_list()
    st.stop()

# Default: Verify Claim page
render_verify()

# Footer
st.markdown(f"""
<div class="footer">
    <strong>FinVet v{__version__}</strong> — Autonomous financial claim verification<br>
    Powered by SEC EDGAR, Finnhub, Tavily
</div>
""", unsafe_allow_html=True)
