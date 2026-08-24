"""Unit tests for the hybrid RAG layer.

No database and no embedder are required: the parser tests run on inline HTML,
and the search tests stub out both the embedding call and the DB session.
"""

from unittest.mock import MagicMock, patch

import pytest

from finvet.rag.parser import (
    _ITEM_PATTERN,
    _ITEM_PATTERN_NO_SPACE,
    chunk_sections,
    parse_filing_html,
)


# ---------------------------------------------------------------------------
# Filing fixtures — the two heading layouts seen in real EDGAR documents
# ---------------------------------------------------------------------------

def _inline_heading_filing() -> str:
    """AAPL/MSFT/TSLA layout: item number and title inside one element."""
    return """
    <html><body>
      <div><span>Item 1.    Business</span></div>
      <p>We design and market smartphones and personal computers worldwide, and we
         sell them through retail stores, a direct sales force, and third party
         wholesalers, retailers and resellers in every major market we serve.</p>
      <div><span>Item 1A.    Risk Factors</span></div>
      <p>Our business is subject to intense competition and supply concentration,
         and a significant interruption in the supply of components would have a
         material adverse effect on our results of operations and cash flows.</p>
      <div><span>Item 3.    Legal Proceedings</span></div>
      <p>We are party to various legal proceedings arising in the ordinary course
         of business, including the matters described in the notes to the
         consolidated financial statements included elsewhere in this report.</p>
      <div><span>Item 9A.    Controls and Procedures</span></div>
      <p>Disclosure controls were effective as of the end of the period covered by
         this report, and there were no changes in internal control over financial
         reporting during the most recent fiscal quarter that materially affected
         our internal control over financial reporting in any respect.</p>
      <div><span>Item 15.    Exhibits</span></div>
      <p>The exhibits filed with this report are listed in the exhibit index that
         immediately precedes the signature page of this annual report, and each
         is incorporated herein by reference where expressly indicated below.</p>
      <div><span>SIGNATURES</span></div>
      <p>Pursuant to the requirements of the Securities Exchange Act of 1934, the
         registrant has duly caused this report to be signed on its behalf by the
         undersigned, thereunto duly authorized officers and directors.</p>
    </body></html>
    """


def _table_cell_heading_filing() -> str:
    """AMZN/Workiva layout: number and title in separate table cells.

    Concatenation leaves no space after the period ("Item 1A.Risk Factors"),
    and the table of contents repeats the numbers with no title at all.
    """
    return """
    <html><body>
      <table><tr><td><span>Item 1.</span></td><td><span>Business</span></td>
                 <td><span>3</span></td></tr>
             <tr><td><span>Item 1A.</span></td><td><span>Risk Factors</span></td>
                 <td><span>6</span></td></tr></table>
      <div><table><tr><td><span>Item 1.</span></td>
                      <td><div><span>Business</span></div></td></tr></table></div>
      <p>We serve consumers through online and physical stores worldwide, and we
         seek to offer low prices, vast selection and convenience across every
         segment and geography in which we choose to operate our business.</p>
      <div><table><tr><td><span>Item 1A.</span></td>
                      <td><div><span>Risk Factors</span></div></td></tr></table></div>
      <p>We face intense competition across geographies and business segments, and
         many of our current and potential competitors have greater resources,
         longer histories and larger customer bases than we currently have.</p>
      <div><table><tr><td><span>Item 3.</span></td>
                      <td><div><span>Legal Proceedings</span></div></td></tr></table></div>
      <p>See Note 7 Commitments and Contingencies for legal proceedings detail, as
         well as the discussion of regulatory matters and government inquiries
         described elsewhere in this annual report on Form 10-K.</p>
    </body></html>
    """


@pytest.fixture
def inline_filing(tmp_path):
    p = tmp_path / "AAPL_10-K_2024-09-28.html"
    p.write_text(_inline_heading_filing(), encoding="utf-8")
    return p


@pytest.fixture
def table_filing(tmp_path):
    p = tmp_path / "AMZN_10-K_2024-12-31.html"
    p.write_text(_table_cell_heading_filing(), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

class TestItemPatterns:
    def test_strict_pattern_matches_spaced_heading(self):
        m = _ITEM_PATTERN.match("Item 1A. Risk Factors")
        assert m and m.group(1) == "1A" and m.group(2) == "Risk Factors"

    def test_strict_pattern_matches_nbsp_separated_heading(self):
        m = _ITEM_PATTERN.match("Item 1A.    Risk Factors")
        assert m and m.group(1) == "1A"

    def test_strict_pattern_rejects_unspaced_heading(self):
        """Why the fallback pattern exists at all."""
        assert _ITEM_PATTERN.match("Item 1A.Risk Factors") is None

    def test_fallback_pattern_matches_unspaced_heading(self):
        m = _ITEM_PATTERN_NO_SPACE.match("Item 1A.Risk Factors")
        assert m and m.group(1) == "1A" and m.group(2) == "Risk Factors"

    def test_fallback_pattern_rejects_bare_toc_entry(self):
        """A title-less cell is a table-of-contents row, not a heading."""
        assert _ITEM_PATTERN_NO_SPACE.match("Item 1A.") is None

    def test_fallback_pattern_rejects_toc_row_with_page_number(self):
        """'Item 1.Business4' would otherwise win the first-occurrence dedup."""
        assert _ITEM_PATTERN_NO_SPACE.match("Item 1.4") is None


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

class TestParseFilingHtml:
    def test_extracts_indexable_sections(self, inline_filing):
        names = {s.name for s in parse_filing_html(inline_filing)}
        assert {"business", "risk_factors", "legal_proceedings"} <= names

    def test_sections_do_not_bleed_into_each_other(self, inline_filing):
        """Regression: _iter_after only yields leaf elements, so a stop tag
        tested against yielded elements alone never matched and every section
        ran to the end of the document."""
        sections = {s.name: s.text for s in parse_filing_html(inline_filing)}
        assert "smartphones" in sections["business"]
        assert "intense competition" not in sections["business"]
        assert "ordinary course" not in sections["business"]

    def test_no_section_runs_to_end_of_document(self, inline_filing):
        for section in parse_filing_html(inline_filing):
            assert "Securities Exchange Act" not in section.text, (
                f"{section.name} bled through to the signature page"
            )

    def test_later_section_is_still_bounded(self, inline_filing):
        sections = {s.name: s.text for s in parse_filing_html(inline_filing)}
        assert "ordinary course" in sections["legal_proceedings"]
        assert "Disclosure controls" not in sections["legal_proceedings"]

    def test_parses_table_cell_headings(self, table_filing):
        """Workiva/iXBRL filers previously yielded zero sections."""
        sections = parse_filing_html(table_filing)
        assert sections, "table-cell heading layout produced no sections"
        assert "business" in {s.name for s in sections}

    def test_table_cell_sections_are_bounded(self, table_filing):
        sections = {s.name: s.text for s in parse_filing_html(table_filing)}
        assert "physical stores" in sections["business"]
        assert "intense competition" not in sections["business"]

    def test_returns_empty_when_no_headings(self, tmp_path):
        p = tmp_path / "X_10-K_2024-01-01.html"
        p.write_text("<html><body><p>No item headings here.</p></body></html>")
        assert parse_filing_html(p) == []


class TestChunkSections:
    def test_chunks_carry_section_metadata(self, inline_filing):
        chunks = chunk_sections(parse_filing_html(inline_filing))
        assert chunks
        assert all(c.section_name for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_indices_restart_per_section(self, inline_filing):
        chunks = chunk_sections(parse_filing_html(inline_filing))
        by_section = {}
        for c in chunks:
            by_section.setdefault(c.section_name, []).append(c.chunk_index)
        for indices in by_section.values():
            assert indices[0] == 0

    def test_respects_token_ceiling(self, inline_filing):
        chunks = chunk_sections(parse_filing_html(inline_filing), max_tokens=50)
        # Overlap can push a chunk slightly past the target; allow headroom.
        assert all(c.token_count <= 120 for c in chunks)


# ---------------------------------------------------------------------------
# Hybrid search — degradation and fusion
# ---------------------------------------------------------------------------

def _row(chunk_id, text="chunk text"):
    row = MagicMock()
    row.id = chunk_id
    row.chunk_text = text
    row.section = "risk_factors"
    row.section_title = "Risk Factors"
    row.filing_type = "10-K"
    row.period_end = "2024-09-28"
    row.ticker = "AAPL"
    return row


class TestHybridSearchDegradation:
    """The keyword arm is pure Postgres. An embedder outage must not take it out."""

    def _service(self):
        from finvet.rag.service import RAGService
        svc = RAGService()
        svc._table_ready = True
        return svc

    @patch("finvet.rag.service.get_db_session")
    @patch("finvet.rag.service._embed_single")
    def test_returns_keyword_results_when_embedder_fails(self, embed, db_session):
        embed.side_effect = RuntimeError("connection refused")
        session = MagicMock()
        session.execute.return_value.fetchall.return_value = [_row(1), _row(2)]
        db_session.return_value.__enter__.return_value = session

        results = self._service().search("supply chain risk", ticker="AAPL")

        assert len(results) == 2, "keyword arm should still return results"
        # Only the keyword query should have run.
        assert session.execute.call_count == 1

    @patch("finvet.rag.service.get_db_session")
    @patch("finvet.rag.service._embed_single")
    def test_runs_both_arms_when_embedder_works(self, embed, db_session):
        embed.return_value = [0.1] * 768
        session = MagicMock()
        session.execute.return_value.fetchall.return_value = [_row(1)]
        db_session.return_value.__enter__.return_value = session

        self._service().search("supply chain risk")

        assert session.execute.call_count == 2

    @patch("finvet.rag.service.get_db_session")
    @patch("finvet.rag.service._embed_single")
    def test_chunk_in_both_arms_outranks_chunk_in_one(self, embed, db_session):
        embed.return_value = [0.1] * 768
        session = MagicMock()
        # Vector arm returns 2 then 1; keyword arm returns 1 only.
        session.execute.return_value.fetchall.side_effect = [
            [_row(2, "vector only"), _row(1, "in both")],
            [_row(1, "in both")],
        ]
        db_session.return_value.__enter__.return_value = session

        results = self._service().search("supply chain risk")

        assert results[0]["chunk_text"] == "in both"
        assert results[0]["score"] > results[1]["score"]


class TestEmbedContract:
    """The embedder must fail loudly on a dimension mismatch rather than let
    pgvector reject every insert halfway through an ingest."""

    @patch("finvet.rag.service.httpx.post")
    def test_rejects_wrong_dimension(self, post):
        from finvet.rag.service import _embed_texts
        post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"embeddings": [[0.1] * 1536]}),
        )
        with pytest.raises(RuntimeError, match="768"):
            _embed_texts(["one"])

    @patch("finvet.rag.service.httpx.post")
    def test_rejects_truncated_batch(self, post):
        from finvet.rag.service import _embed_texts
        post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"embeddings": [[0.1] * 768]}),
        )
        with pytest.raises(RuntimeError, match="2 inputs"):
            _embed_texts(["one", "two"])
