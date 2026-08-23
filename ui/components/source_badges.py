"""Data source badge rendering (XBRL, RAG, A2A)."""


def _data_source_badges_html(metadata: dict) -> str:
    """Generate HTML badges for data sources used (XBRL, RAG, A2A)."""
    ds = metadata.get("data_sources", {})
    if not ds:
        return ""
    badge_map = {
        "xbrl": ("XBRL", "ds-xbrl"),
        "rag": ("RAG", "ds-rag"),
        "a2a": ("A2A", "ds-a2a"),
    }
    badges = []
    for key in ("xbrl", "rag", "a2a"):
        if key in ds:
            label, css_class = badge_map[key]
            badges.append(f'<span class="ds-badge {css_class}">{label}</span>')
    return f'<span class="ds-badges">{"".join(badges)}</span>' if badges else ""


def _override_badge_html(metadata: dict) -> str:
    """Badge shown when the deterministic layer overruled the LLM verdict.

    Deliberately not a data-source badge — it describes how the verdict was
    decided, not where the evidence came from.
    """
    if not metadata.get("override_applied"):
        return ""
    original = metadata.get("llm_original_verdict")
    title = (
        f"Model concluded {original}; deterministic numeric check overruled it"
        if original
        else "Deterministic numeric check overruled the model verdict"
    )
    parts = [
        '<span class="ds-badges">',
        f'<span class="ds-badge ds-override" title="{title}">OVERRIDE</span>',
        '</span>',
    ]
    return "".join(parts)
