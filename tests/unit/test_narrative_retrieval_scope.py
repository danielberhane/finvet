"""A disclosure is scoped forward from its event, not onto a period.

Commit 5b86cff ("reject unscoped or irrelevant filing evidence") gave filing
search the resolved period, reasoning that "a FY2024 query could fill its
candidate set with chunks from other years -- wrong evidence, not weak
evidence". That is right, and the example it reaches for says what it was aimed
at: a *numeric* query, where a FY2023 figure satisfying a FY2024 claim is
genuinely wrong.

Applied to narrative text it inverts. A number belongs to one period; a
disclosure describes an *event*, and appears in whichever filings were current
while the matter was live -- often for years. Apple's March 2024 European
Commission investigation is disclosed in the FY2025 10-K and carried across
three filings. Exact matching guarantees a miss:

    period_end = 2024-12-31 (resolved)     -> 0 chunks
    period_end = 2024-09-28 (fiscal-exact) -> 0 chunks
    period_start = 2024-01-01              -> 4 chunks

That was not hypothetical. The News -> SEC delegation failed 100% of the time:
the nested agent searched, got nothing, reworded, got nothing, and died at its
recursion limit with no provenance at all. With forward scoping the first
search returns evidence and the run completes.

Safe because filing prose can never become a trusted observation
(SUPPORTING_EVIDENCE_TOOLS). No number can rest on it, so "wrong-period
narrative text" and "wrong-period figure" are not the same risk -- which is
exactly why the constraint was misplaced here. Numeric retrieval through
sec_tools keeps exact matching, untouched.

The known imprecision is stated rather than discovered later: a claim naming a
specific filing also matches later filings carrying the disclosure forward. The
chunks carry their own period_end and filing_type, so the attribution stays
visible.
"""

from unittest.mock import MagicMock, patch

import pytest


def _search(period_target, **tool_kwargs):
    """Invoke the tool with a resolved period and capture what reached RAG."""
    from finvet.tools.filing_search import search_filing_text

    rag = MagicMock()
    rag.available = True
    rag.search.return_value = []
    rag.has_filings_for.return_value = True

    with patch("finvet.tools.filing_search.get_rag_service", return_value=rag), \
         patch("finvet.tools.filing_search._current_period_target",
               return_value=period_target):
        search_filing_text.invoke(
            {"query": "European Commission fine", "ticker": "AAPL",
             **tool_kwargs})
    return rag.search.call_args.kwargs


class TestTheResolvedPeriodBecomesALowerBound:

    def test_the_period_is_passed_as_a_start_not_an_end(self):
        """The change. A filing can only describe events that happened before
        it closed, so the resolved period bounds the search from below."""
        kwargs = _search(("2024-12-31", "annual"))

        assert kwargs.get("period_start") == "2024-12-31"
        assert not kwargs.get("period_end"), (
            "an exact period_end excludes the later filing that carries the "
            "disclosure forward")

    def test_no_resolved_period_constrains_nothing(self):
        """A claim naming no period searches the whole corpus, as before."""
        kwargs = _search((None, None))

        assert not kwargs.get("period_start")
        assert not kwargs.get("period_end")

    def test_no_upper_bound_is_imposed(self):
        """A disclosure may be restated for years; the newest filing is still
        a valid source for an older event."""
        assert not _search(("2024-12-31", "annual")).get("period_end_max")

    @pytest.mark.parametrize("passthrough", ["section", "filing_type"])
    def test_the_other_filters_are_untouched(self, passthrough):
        kwargs = _search(("2024-12-31", "annual"), **{passthrough: "x"})

        assert kwargs.get(passthrough) == "x"

    def test_the_ticker_still_scopes_the_search(self):
        """The filter that actually prevents cross-company evidence is
        unaffected, and it is the one that matters most."""
        assert _search(("2024-12-31", "annual")).get("ticker") == "AAPL"


class TestNumericRetrievalKeepsExactMatching:
    """The guard 5b86cff added is right where it was aimed. Only the narrative
    tool changes; a figure from the wrong fiscal year is still wrong evidence."""

    def test_the_xbrl_path_still_targets_one_period(self):
        from finvet.tools.sec_tools import period_target_for

        class _P:
            period_type = "annual"
            end_date = "2024-09-28"

        assert period_target_for(_P()) == ("2024-09-28", "annual")

    def test_filing_text_can_never_become_a_trusted_observation(self):
        """Why relaxing this scope costs nothing in numeric rigour."""
        from finvet.models.evidence import SUPPORTING_EVIDENCE_TOOLS

        assert "search_filing_text" in SUPPORTING_EVIDENCE_TOOLS
