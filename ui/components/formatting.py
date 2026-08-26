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


# Tokens whose casing `.title()` gets wrong.
_PRESERVE_CASE = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4",
                  "sec": "SEC", "xbrl": "XBRL", "rag": "RAG", "a2a": "A2A",
                  "us": "US", "pe": "P/E", "hitl": "HITL", "api": "API",
                  "pii": "PII", "id": "ID", "ui": "UI"}


def humanize(value):
    """Turn a machine value into something a person can read.

    Defence in depth for the whole surface, not one field. `Claim rejected:
    non_financial` reached a user because a serialization format was pasted
    into display text; the same shape exists in limitations
    (`unsupported_q4_derivation`), HITL triggers (`output_safety_violation`)
    and anything added later. Anything rendered goes through here.

    Text already written for a human is returned untouched — it contains
    spaces, so it is prose, and re-casing it would mangle a real sentence.
    """
    if not value:
        return "--"
    text = str(value)
    if " " in text:
        return text
    words = [_PRESERVE_CASE.get(w.lower(), w.capitalize())
             for w in text.split("_") if w]
    return " ".join(words) if words else "--"


_AGENT_NAMES = {"sec": "SEC", "market": "Market", "news": "News"}

# Read from the response rather than inferred: `data_sources.a2a`
# records which way the delegation ran.
_A2A_DIRECTIONS = {"news_to_sec": ("news", "sec"),
                   "sec_to_news": ("sec", "news")}

# A run that never selected an agent still had a stage that ended it, and the
# rejection records which one.
_STAGE_NAMES = {
    "rejected_parser": "Claim Parser",
    "rejected_input_guard": "Input Guardrails",
    "rejected_human": "Human Reviewer",
}


def stage_label(metadata):
    """Who produced this result — an agent, or the stage that stopped it.

    "UNKNOWN Agent · 0.0s" was shown on a rejected claim. Nothing unknown had
    happened: the parser read a question, classified it, and stopped before any
    agent was selected, which is why the metadata carries no `agent`. The UI
    defaulted the absent value to "unknown" and captioned it as an agent that
    could not be identified.

    An empty string when nothing is known, rather than a name that was made up.
    """
    metadata = metadata or {}
    agent = metadata.get("agent")
    if agent and str(agent).lower() != "unknown":
        label = f"{_AGENT_NAMES.get(agent, humanize(agent))} Agent"

        # When one agent asked another, say so. The delegation is the most
        # interesting thing that happened on such a run, and naming only the
        # agent that started it left the feature visible in the badge and the
        # raw response but not in the sentence describing who answered.
        a2a = (metadata.get("data_sources") or {}).get("a2a") or {}
        if a2a.get("used"):
            pair = _A2A_DIRECTIONS.get(a2a.get("direction"))
            if pair:
                source, target = pair
                label = (f"{_AGENT_NAMES.get(source, humanize(source))} Agent"
                         f" → "
                         f"{_AGENT_NAMES.get(target, humanize(target))} Agent")
        return label

    stage = _STAGE_NAMES.get(metadata.get("disposition"))
    return stage or ""


def parsed_claim_rows(parsed):
    """The parse as (label, value) rows a person can read.

    It reaches the response nested inside twenty other metadata keys, which is
    technically visible and practically hidden. This is the same information
    laid out for reading: labels in words, numbers formatted, and empty fields
    dropped rather than printed as null.

    Two absences are kept, because they mean something. A null metric with a
    null value is not a gap — it is the parse that routes a claim to filing
    text instead of XBRL, and it decides which half of the system answers.
    """
    if not parsed:
        return []

    rows = []

    def add(label, value):
        if value not in (None, "", []):
            rows.append((label, value))

    add("Claim type", humanize(parsed.get("claim_type")))
    add("Ticker", parsed.get("ticker"))

    metric, value = parsed.get("metric"), parsed.get("value")
    if metric:
        add("Metric", metric)
    elif parsed.get("claim_type") == "sec":
        # The routing decision, stated rather than left as a blank.
        rows.append(("Metric", "none — routed to filing text"))

    low, high = parsed.get("range_min"), parsed.get("range_max")
    if low is not None and high is not None:
        add("Range", f"{format_value(low)} – {format_value(high)}")
    elif value is not None:
        add("Value", format_value(value))

    add("Comparison", humanize(parsed.get("operator"))
        if parsed.get("operator") else None)
    add("Period", parsed.get("period"))
    if parsed.get("reject_reason"):
        add("Reject reason", humanize(parsed["reject_reason"]))
    return rows


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
