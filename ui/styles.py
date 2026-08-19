"""CSS styles for the FinVet Streamlit UI."""

STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .logo-text {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A5F 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }

    .tagline {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    .verdict-supports {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #10B981;
        color: #334155;
        padding: 1.5rem 2rem;
        border-radius: 0 10px 10px 0;
    }

    .verdict-supports .verdict-text {
        color: #059669;
    }

    .verdict-supports .verdict-confidence {
        color: #059669;
    }

    .verdict-refutes {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #EF4444;
        color: #334155;
        padding: 1.5rem 2rem;
        border-radius: 0 10px 10px 0;
    }

    .verdict-refutes .verdict-text {
        color: #DC2626;
    }

    .verdict-refutes .verdict-confidence {
        color: #DC2626;
    }

    .verdict-nei {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #F59E0B;
        color: #334155;
        padding: 1.5rem 2rem;
        border-radius: 0 10px 10px 0;
    }

    .verdict-nei .verdict-text {
        color: #D97706;
    }

    .verdict-nei .verdict-confidence {
        color: #D97706;
    }

    .verdict-pending {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #8B5CF6;
        color: #334155;
        padding: 1.5rem 2rem;
        border-radius: 0 10px 10px 0;
    }

    .verdict-pending .verdict-text {
        color: #7C3AED;
    }

    .verdict-pending .verdict-confidence {
        color: #7C3AED;
    }

    /* HITL Queued Notification */
    .hitl-queued {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #64748B;
        border-radius: 0 10px 10px 0;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    .hitl-queued-top {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .hitl-queued-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #F59E0B;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
        flex-shrink: 0;
        animation: hitl-pulse 2s ease-in-out infinite;
    }

    @keyframes hitl-pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(245, 158, 11, 0.4); }
        50% { opacity: 0.6; box-shadow: 0 0 4px rgba(245, 158, 11, 0.2); }
    }

    .hitl-queued-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .hitl-queued-body {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .hitl-queued-details {
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .hitl-detail-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.3rem 0.65rem;
        font-size: 0.78rem;
        color: #475569;
    }

    .hitl-detail-label {
        font-weight: 500;
        color: #94A3B8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .hitl-detail-value {
        font-weight: 600;
        color: #1E3A5F;
    }

    .hitl-review-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #1E293B;
        color: #F8FAFC;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.25rem;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        text-decoration: none;
        transition: background 0.15s ease;
        flex-shrink: 0;
    }

    .hitl-review-btn:hover {
        background: #334155;
    }

    .hitl-review-arrow {
        font-size: 1rem;
        opacity: 0.7;
    }

    .hitl-trigger-tag {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .verdict-text {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .hitl-panel {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .hitl-header {
        color: #92400E;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .stTextInput > div > div > input {
        font-size: 1rem;
        line-height: 1.4;
        padding: 0.5rem 1rem !important;
        border-radius: 8px;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        box-sizing: border-box;
    }

    .stButton > button {
        font-size: 1rem;
        line-height: 1.4;
        padding: 0.5rem 2rem !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        box-sizing: border-box;
        font-weight: 600;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        border: none;
        border-radius: 8px;
    }

    [data-testid="stBaseButton-secondary"] {
        background: #E2E8F0 !important;
        border: 1px solid #94A3B8 !important;
        color: #1E293B !important;
        font-size: 0.83rem !important;
        font-weight: 700 !important;
        height: 34px !important;
        min-height: 34px !important;
        max-height: 34px !important;
        padding: 0 1.1rem !important;
        letter-spacing: 0.1px !important;
        box-shadow: none !important;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        background: #CBD5E1 !important;
        border-color: #64748B !important;
        color: #0F172A !important;
        box-shadow: none !important;
    }

    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }

    /* Evidence report */
    .evidence-report {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .report-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
    }

    .report-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
    }

    .report-source {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .report-body {
        padding: 1.25rem 1.5rem;
        line-height: 1.75;
        color: #334155;
        font-size: 0.9rem;
    }

    .report-section-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #1E3A5F;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #f1f5f9;
    }

    .report-section-title:first-child {
        margin-top: 0;
    }

    .report-paragraph {
        margin: 0 0 0.75rem 0;
        line-height: 1.75;
    }

    .report-paragraph:last-child {
        margin-bottom: 0;
    }

    .report-body strong {
        color: #1E3A5F;
        font-weight: 600;
    }

    .report-body em {
        color: #475569;
        font-style: italic;
    }

    .report-body code {
        background: #f1f5f9;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85em;
        color: #3B82F6;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }

    .report-list {
        margin: 0.4rem 0 0.75rem 0;
        padding-left: 1.5rem;
    }

    .report-list li {
        margin-bottom: 0.35rem;
        line-height: 1.65;
    }

    .report-list li::marker {
        color: #94a3b8;
    }

    .report-kv {
        padding: 0.3rem 0;
        line-height: 1.6;
    }

    .report-kv-key {
        font-weight: 600;
        color: #475569;
    }

    .report-footer {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.65rem 1.25rem;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
    }

    .report-footer-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-right: 0.25rem;
    }

    .tool-chip {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.73rem;
        font-weight: 500;
    }

    /* Tool calls detail */
    .report-tools {
        border-top: 1px solid #e2e8f0;
    }

    .report-tools-header {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        padding: 0.65rem 1.25rem 0.4rem;
        background: #f8fafc;
    }

    .tool-call-row {
        padding: 0.5rem 1.25rem;
        border-top: 1px solid #f1f5f9;
    }

    .tool-call-row:last-child {
        padding-bottom: 0.75rem;
    }

    .tool-call-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .tool-call-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4338CA;
        background: #EEF2FF;
        padding: 2px 8px;
        border-radius: 4px;
    }

    .tool-call-args {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }

    .tool-arg-key {
        color: #475569;
        font-weight: 500;
    }

    .tool-arg-none {
        color: #94a3b8;
        font-style: italic;
    }

    .tool-status-ok {
        font-size: 0.65rem;
        font-weight: 700;
        color: #059669;
        background: #D1FAE5;
        padding: 1px 6px;
        border-radius: 3px;
        letter-spacing: 0.3px;
    }

    .tool-status-err {
        font-size: 0.65rem;
        font-weight: 700;
        color: #DC2626;
        background: #FEE2E2;
        padding: 1px 6px;
        border-radius: 3px;
        letter-spacing: 0.3px;
    }

    .tool-call-result {
        font-size: 0.75rem;
        color: #64748b;
        background: #f8fafc;
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
        margin-top: 0.3rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 120px;
        overflow-y: auto;
    }

    /* Similar claims section */
    .similar-section {
        margin-top: 1.5rem;
    }

    .similar-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .similar-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s ease;
    }

    .similar-card:hover {
        border-color: #cbd5e1;
    }

    .similar-card-left {
        flex: 1;
        min-width: 0;
    }

    .similar-claim-text {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .similar-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.35rem;
        font-size: 0.75rem;
        color: #94a3b8;
    }

    .similar-meta-item {
        display: flex;
        align-items: center;
        gap: 0.2rem;
    }

    .similar-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .similar-badge-supports { background: #D1FAE5; color: #065F46; }
    .similar-badge-refutes { background: #FEE2E2; color: #991B1B; }
    .similar-badge-nei { background: #FEF3C7; color: #92400E; }

    /* Data source badges */
    .ds-badges { display: inline-flex; gap: 0.35rem; margin-left: 0.75rem; vertical-align: middle; }
    .ds-badge {
        display: inline-flex; align-items: center; gap: 0.25rem;
        padding: 0.15rem 0.55rem; border-radius: 999px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .ds-xbrl { background: #DBEAFE; color: #1E40AF; }
    .ds-rag  { background: #EDE9FE; color: #6D28D9; }
    .ds-a2a  { background: #FEF3C7; color: #92400E; }

    .similar-similarity {
        font-size: 0.88rem;
        font-weight: 700;
        color: #3B82F6;
        white-space: nowrap;
        margin-left: 1rem;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }

    .pending-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .pending-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }

    .pending-claim-text {
        color: #1E3A5F;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    .pending-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.85rem;
        color: #64748b;
        align-items: center;
    }

    .pending-meta-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .badge-supports {
        background: #D1FAE5;
        color: #065F46;
    }

    .badge-refutes {
        background: #FEE2E2;
        color: #991B1B;
    }

    .badge-nei {
        background: #FEF3C7;
        color: #92400E;
    }

    .badge-low {
        background: #FEE2E2;
        color: #991B1B;
    }

    .badge-medium {
        background: #FEF3C7;
        color: #92400E;
    }

    .badge-high {
        background: #D1FAE5;
        color: #065F46;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #94a3b8;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    [data-testid="stSidebar"] {
        background: #0F172A;
    }

    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 1.5rem 0 1rem;
        text-align: center;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 1.5rem;
    }

    .sidebar-brand-text {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-brand-sub {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #475569 !important;
        margin-top: 0.25rem;
    }

    /* Nav items */
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #94A3B8 !important;
        cursor: pointer;
        transition: all 0.15s ease;
        text-decoration: none;
    }

    .sidebar-nav-item:hover {
        background: #1E293B;
        color: #E2E8F0 !important;
    }

    .sidebar-nav-active {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E40AF 100%);
        color: #FFFFFF !important;
        font-weight: 600;
    }

    .sidebar-nav-active:hover {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E40AF 100%);
    }

    .nav-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }

    .nav-dot-verify { background: #60A5FA; }
    .nav-dot-reviews { background: #A78BFA; }

    /* Sidebar section label */
    .sidebar-section {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #475569 !important;
        padding: 0 1rem;
        margin: 1.5rem 0 0.5rem;
        font-weight: 600;
    }

    /* Pending count badge */
    .pending-badge {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }

    .pending-badge-count {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F59E0B !important;
    }

    .pending-badge-label {
        font-size: 0.75rem;
        color: #64748B !important;
        margin-top: 0.1rem;
    }

    /* Sidebar system status */
    .sidebar-status {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.3rem 0;
    }

    .status-row + .status-row {
        border-top: 1px solid #334155;
        margin-top: 0.2rem;
        padding-top: 0.5rem;
    }

    .status-label {
        font-size: 0.7rem;
        color: #64748B !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-value {
        font-size: 0.78rem;
        font-weight: 600;
        color: #CBD5E1 !important;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        position: relative;
        top: -0.5px;
    }

    .status-dot-ok {
        background: #10B981;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
    }

    .status-dot-err {
        background: #EF4444;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
    }

    .status-verdict {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 3px;
        letter-spacing: 0.3px;
    }

    .status-verdict-supports { background: #065F46; color: #D1FAE5 !important; }
    .status-verdict-refutes { background: #991B1B; color: #FEE2E2 !important; }
    .status-verdict-nei { background: #92400E; color: #FEF3C7 !important; }
    .status-verdict-none { background: #334155; color: #64748B !important; }

    /* Sidebar footer */
    .sidebar-footer {
        position: absolute;
        bottom: 1rem;
        left: 1rem;
        right: 1rem;
        text-align: center;
        font-size: 0.7rem;
        color: #334155 !important;
        border-top: 1px solid #1E293B;
        padding-top: 0.75rem;
    }

    /* Memory match card */
    .memory-card {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 0 10px 10px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .memory-card-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .memory-card-title {
        font-size: 0.82rem; font-weight: 700; color: #334155;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .memory-card-similarity {
        font-size: 0.88rem; font-weight: 700; color: #3B82F6;
    }
    .memory-card-claim {
        font-size: 0.95rem; color: #1E3A5F; line-height: 1.6;
        margin-bottom: 0.75rem; padding: 0.5rem 0;
    }
    .memory-card-meta {
        display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
    }
    .memory-meta-chip {
        display: inline-flex; align-items: center; gap: 0.25rem;
        background: white; border: 1px solid #E2E8F0; border-radius: 6px;
        padding: 0.3rem 0.65rem; font-size: 0.78rem; color: #475569;
    }
    .memory-meta-label {
        font-weight: 500; color: #94A3B8; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.3px;
    }
    .memory-meta-value { font-weight: 600; color: #1E3A5F; }

    /* Memory verdict badge in card */
    .memory-verdict-supports { background: #D1FAE5; color: #065F46; }
    .memory-verdict-refutes { background: #FEE2E2; color: #991B1B; }
    .memory-verdict-nei { background: #FEF3C7; color: #92400E; }

    /* Style sidebar radio as nav items */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 4px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        margin: 0 !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #1E293B !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E40AF 100%) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Hide the radio circle */
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] {
        pointer-events: none;
    }

    [data-testid="stSidebar"] .stRadio > div > div {
        display: none;
    }

    /* ── Audit Trail ─────────────────────────────────────────── */

    .audit-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s, box-shadow 0.15s;
    }

    .audit-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }

    .audit-card-claim {
        font-size: 0.95rem;
        color: #1E3A5F;
        font-weight: 500;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .audit-card-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        font-size: 0.78rem;
        color: #64748b;
    }

    .pipeline-timeline {
        border-left: 2px solid #E2E8F0;
        margin: 1rem 0 1rem 0.75rem;
        padding-left: 1.25rem;
    }

    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
        position: relative;
    }

    .pipeline-step-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
        margin-top: 3px;
        margin-left: -1.6rem;
        border: 2px solid #ffffff;
    }

    .pipeline-step-dot-ok   { background: #10B981; }
    .pipeline-step-dot-err  { background: #EF4444; }
    .pipeline-step-dot-warn { background: #F59E0B; }
    .pipeline-step-dot-skip { background: #CBD5E1; }

    .pipeline-step-body { flex: 1; }

    .pipeline-step-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }

    .pipeline-step-meta {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 1px;
    }

    .hash-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.75rem;
    }

    .hash-label {
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        flex-shrink: 0;
    }

    .hash-value {
        font-family: 'Courier New', monospace;
        color: #334155;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .hash-verified {
        color: #10B981;
        font-weight: 700;
        flex-shrink: 0;
    }
</style>
"""
