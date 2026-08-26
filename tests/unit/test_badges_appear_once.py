"""A data-source badge belongs in one place on a page, not two.

`render_evidence` renders the XBRL / RAG / A2A badges into the Analysis Report
header. The verify page and the audit page each *already* render the same
badges beside the verdict, so both showed them twice within a few centimetres —
once under the confidence figure and again on the report below it. Repeating a
signal does not reinforce it; it makes the reader wonder whether the two are
saying different things.

Removing the line outright was not available: the review page calls the same
component and the Analysis Report is the **only** place badges appear there.
Deleting it would have stripped provenance from the page whose entire purpose
is judging provenance.

So it is a parameter, defaulting to *showing* them. Of the two ways a future
call site can get this wrong, a duplicated badge is cosmetic and a missing one
loses information.
"""

import sys
from pathlib import Path

UI = Path(__file__).resolve().parents[2] / "ui"
sys.path.insert(0, str(UI))

# The inner map: render_evidence wraps it as {"data_sources": ...} itself.
DATA_SOURCES = {"xbrl": {"used": True, "tools": []},
                "rag": {"used": True, "chunks_retrieved": 3}}


class TestTheComponentCanBeToldNotToRepeatThem:

    def _html(self, **kwargs):
        """Capture what render_evidence would emit, without a live Streamlit."""
        from unittest.mock import patch

        import components.evidence as evidence

        captured = {}
        with patch.object(evidence.st, "markdown",
                          side_effect=lambda html, **kw: captured.setdefault("html", html)):
            evidence.render_evidence(
                reasoning="Some reasoning.",
                source_description="SEC EDGAR",
                tools_called=["get_income_statement"],
                tool_calls_detail=[],
                data_sources=DATA_SOURCES,
                **kwargs)
        return captured.get("html", "")

    def test_badges_show_by_default(self):
        """The review page relies on this; a silent default of False would
        strip provenance from the page that needs it most."""
        assert "ds-badge" in self._html()

    def test_they_can_be_suppressed(self):
        assert "ds-badge" not in self._html(show_badges=False)

    def test_suppressing_them_keeps_everything_else(self):
        html = self._html(show_badges=False)

        assert "Analysis Report" in html
        assert "SEC EDGAR" in html
        assert "Some reasoning." in html


class TestThePagesThatAlreadyShowBadgesDoNotRepeatThem:
    """Asserted against the source, because these are call-site decisions and
    the duplication is invisible in any single component's output."""

    def _source(self, name):
        return (UI / "views" / name).read_text()

    def test_the_verify_page_suppresses_the_second_copy(self):
        source = self._source("verify.py")

        assert "_data_source_badges_html(metadata)" in source, (
            "the verify page should still show badges beside the verdict")
        assert "show_badges=False" in source, (
            "and should not show them again on the report below")

    def test_the_audit_page_suppresses_the_second_copy(self):
        source = self._source("audit_detail.py")

        assert "_data_source_badges_html(" in source
        assert "show_badges=False" in source

    def test_the_review_page_still_shows_them(self):
        """It has no other badge site, so the Analysis Report is the only
        provenance a reviewer sees."""
        source = self._source("review_detail.py")

        assert "show_badges=False" not in source
        assert "_data_source_badges_html" not in source, (
            "if this page grows its own badge row, the report copy should be "
            "suppressed here too")
