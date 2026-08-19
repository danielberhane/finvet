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
