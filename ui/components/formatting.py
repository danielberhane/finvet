"""Shared formatting utilities for the FinVet UI."""

import re


def format_value(val):
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if abs(val) >= 1_000_000_000:
        return f"${val / 1_000_000_000:.2f}B"
    elif abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"${val / 1_000:.2f}K"
    else:
        return f"${val:,.2f}"


# Display only. The wire value stays SCREAMING_SNAKE everywhere it means
# something -- the API contract, the audit record, the raw-response panel and
# the review decision submitted back to the server. This maps it to prose at
# the last moment, for a person reading a page.
_VERDICT_LABELS = {
    "SUPPORTS": "Supports",
    "REFUTES": "Refutes",
    "NOT_ENOUGH_INFO": "Not Enough Info",
    "PENDING": "Pending",
    "REJECTED": "Rejected",
    "UNKNOWN": "Unknown",
}


def verdict_label(verdict):
    """Human-readable verdict text.

    An unrecognised value is title-cased rather than dropped: a new verdict
    added server-side should read imperfectly, never disappear from the page.
    """
    if not verdict:
        return "--"
    known = _VERDICT_LABELS.get(verdict)
    if known:
        return known
    return str(verdict).replace("_", " ").title()


def _escape(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(text):
    """Convert inline markdown to HTML (bold, italic, code)."""
    # Bold: **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code: `text`
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text
