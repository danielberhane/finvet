"""Evidence / analysis report rendering."""

import re
import streamlit as st

from components.formatting import _escape, _md_inline
from components.source_badges import _data_source_badges_html


def render_evidence(reasoning, source_description=None, tools_called=None,
                    tool_calls_detail=None, data_sources=None):
    """Render the evidence as a structured analysis report."""
    if not reasoning or not reasoning.strip():
        return

    # Undo any \\$ escaping from response_generator (not needed in HTML)
    clean = reasoning.replace("\\$", "$")[:4000]

    # Split reasoning into paragraphs and identify structure
    sections = _parse_reasoning_sections(clean)

    # Build the evidence report
    html_parts = ['<div class="evidence-report">']

    # Section header
    html_parts.append('<div class="report-header">')
    html_parts.append('<span class="report-title">Analysis Report</span>')
    if data_sources:
        html_parts.append(_data_source_badges_html({"data_sources": data_sources}))
    if source_description:
        html_parts.append(f'<span class="report-source">{_escape(source_description)}</span>')
    html_parts.append('</div>')

    # Main analysis body
    html_parts.append('<div class="report-body">')
    for section in sections:
        if section["type"] == "heading":
            html_parts.append(f'<div class="report-section-title">{_escape(section["text"])}</div>')
        elif section["type"] == "list":
            html_parts.append('<ul class="report-list">')
            for item in section["items"]:
                html_parts.append(f'<li>{_md_inline(_escape(item))}</li>')
            html_parts.append('</ul>')
        elif section["type"] == "numbered_list":
            html_parts.append('<ol class="report-list">')
            for item in section["items"]:
                html_parts.append(f'<li>{_md_inline(_escape(item))}</li>')
            html_parts.append('</ol>')
        elif section["type"] == "key_value":
            html_parts.append(f'<div class="report-kv"><span class="report-kv-key">{_escape(section["key"])}:</span> {_md_inline(_escape(section["value"]))}</div>')
        else:
            html_parts.append(f'<p class="report-paragraph">{_md_inline(_escape(section["text"]))}</p>')
    html_parts.append('</div>')

    # Tool calls detail section
    if tool_calls_detail:
        html_parts.append('<div class="report-tools">')
        html_parts.append('<div class="report-tools-header">Tool Calls</div>')
        for call in tool_calls_detail:
            tool_name = _escape(str(call.get("tool", "")))
            success = call.get("success", True)
            status_class = "tool-status-ok" if success else "tool-status-err"
            status_label = "OK" if success else "FAILED"

            # Format arguments
            args = call.get("args", {})
            if isinstance(args, dict):
                args_parts = []
                for k, v in args.items():
                    v_str = str(v)
                    if len(v_str) > 80:
                        v_str = v_str[:77] + "..."
                    args_parts.append(f'<span class="tool-arg-key">{_escape(k)}</span>={_escape(v_str)}')
                args_html = ", ".join(args_parts) if args_parts else '<span class="tool-arg-none">no args</span>'
            else:
                args_html = _escape(str(args)[:120])

            # Format result (truncated)
            result_text = call.get("result", call.get("error", ""))
            if len(result_text) > 300:
                result_text = result_text[:297] + "..."

            html_parts.append('<div class="tool-call-row">')
            html_parts.append('<div class="tool-call-header">')
            html_parts.append(f'<span class="tool-call-name">{tool_name}</span>')
            html_parts.append(f'<span class="tool-call-args">{args_html}</span>')
            html_parts.append(f'<span class="{status_class}">{status_label}</span>')
            html_parts.append('</div>')
            html_parts.append(f'<div class="tool-call-result">{_escape(result_text)}</div>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    # Tools summary footer (only if no detail available)
    elif tools_called:
        seen = set()
        unique_tools = [t for t in tools_called if not (t in seen or seen.add(t))]
        tools_html = "".join(f'<span class="tool-chip">{_escape(t)}</span>' for t in unique_tools)
        html_parts.append(f'<div class="report-footer"><span class="report-footer-label">Tools used</span>{tools_html}</div>')

    html_parts.append('</div>')

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def _parse_reasoning_sections(text):
    """Parse LLM reasoning text into structured sections for rendering."""
    sections = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Detect markdown headers (## Header or **Header**)
        header_match = re.match(r'^#{1,3}\s+(.+)', line)
        bold_header_match = re.match(r'^\*\*([^*]+)\*\*:?\s*$', line)
        if header_match:
            sections.append({"type": "heading", "text": header_match.group(1)})
            i += 1
            continue
        if bold_header_match:
            sections.append({"type": "heading", "text": bold_header_match.group(1)})
            i += 1
            continue

        # Detect bullet lists (- item or * item)
        if re.match(r'^[-*]\s+', line):
            items = []
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
                items.append(re.sub(r'^[-*]\s+', '', lines[i].strip()))
                i += 1
            sections.append({"type": "list", "items": items})
            continue

        # Detect numbered lists (1. item)
        if re.match(r'^\d+[\.\)]\s+', line):
            items = []
            while i < len(lines) and re.match(r'^\d+[\.\)]\s+', lines[i].strip()):
                items.append(re.sub(r'^\d+[\.\)]\s+', '', lines[i].strip()))
                i += 1
            sections.append({"type": "numbered_list", "items": items})
            continue

        # Detect key: value patterns (e.g., "Revenue: $3.4B")
        kv_match = re.match(r'^([A-Z][A-Za-z\s]{2,30}):\s+(.+)', line)
        if kv_match and not line.endswith(':'):
            sections.append({"type": "key_value", "key": kv_match.group(1).strip(), "value": kv_match.group(2).strip()})
            i += 1
            continue

        # Regular paragraph - collect consecutive non-empty, non-special lines
        para_lines = []
        while i < len(lines):
            l = lines[i].strip()
            if not l:
                i += 1
                break
            if re.match(r'^#{1,3}\s+', l) or re.match(r'^\*\*[^*]+\*\*:?\s*$', l):
                break
            if re.match(r'^[-*]\s+', l) or re.match(r'^\d+[\.\)]\s+', l):
                break
            para_lines.append(l)
            i += 1
        if para_lines:
            sections.append({"type": "paragraph", "text": " ".join(para_lines)})

    return sections
