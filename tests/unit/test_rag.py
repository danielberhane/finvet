"""Unit tests for the hybrid RAG layer.

No database and no embedder are required: the parser tests run on inline HTML,
and the search tests stub out both the embedding call and the DB session.
"""

from pathlib import Path
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
        names = {s.name for s in parse_filing_html(inline_filing, filing_type="10-K")}
        assert {"business", "risk_factors", "legal_proceedings"} <= names

    def test_sections_do_not_bleed_into_each_other(self, inline_filing):
        """Regression: _iter_after only yields leaf elements, so a stop tag
        tested against yielded elements alone never matched and every section
        ran to the end of the document."""
        sections = {s.name: s.text for s in parse_filing_html(inline_filing, filing_type="10-K")}
        assert "smartphones" in sections["business"]
        assert "intense competition" not in sections["business"]
        assert "ordinary course" not in sections["business"]

    def test_no_section_runs_to_end_of_document(self, inline_filing):
        for section in parse_filing_html(inline_filing, filing_type="10-K"):
            assert "Securities Exchange Act" not in section.text, (
                f"{section.name} bled through to the signature page"
            )

    def test_later_section_is_still_bounded(self, inline_filing):
        sections = {s.name: s.text for s in parse_filing_html(inline_filing, filing_type="10-K")}
        assert "ordinary course" in sections["legal_proceedings"]
        assert "Disclosure controls" not in sections["legal_proceedings"]

    def test_parses_table_cell_headings(self, table_filing):
        """Workiva/iXBRL filers previously yielded zero sections."""
        sections = parse_filing_html(table_filing, filing_type="10-K")
        assert sections, "table-cell heading layout produced no sections"
        assert "business" in {s.name for s in sections}

    def test_table_cell_sections_are_bounded(self, table_filing):
        sections = {s.name: s.text for s in parse_filing_html(table_filing, filing_type="10-K")}
        assert "physical stores" in sections["business"]
        assert "intense competition" not in sections["business"]

    def test_returns_empty_when_no_headings(self, tmp_path):
        p = tmp_path / "X_10-K_2024-01-01.html"
        p.write_text("<html><body><p>No item headings here.</p></body></html>")
        assert parse_filing_html(p, filing_type="10-K") == []


class TestChunkSections:
    def test_chunks_carry_section_metadata(self, inline_filing):
        chunks = chunk_sections(parse_filing_html(inline_filing, filing_type="10-K"))
        assert chunks
        assert all(c.section_name for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_indices_restart_per_section(self, inline_filing):
        chunks = chunk_sections(parse_filing_html(inline_filing, filing_type="10-K"))
        by_section = {}
        for c in chunks:
            by_section.setdefault(c.section_name, []).append(c.chunk_index)
        for indices in by_section.values():
            assert indices[0] == 0

    def test_respects_token_ceiling(self, inline_filing):
        """A hard bound, not a target.

        This allowed 120 tokens for a max of 50 -- 2.4x -- on the theory that
        overlap could push a chunk past it. Overlap now yields to the ceiling
        instead, so the bound is exact.
        """
        chunks = chunk_sections(
            parse_filing_html(inline_filing, filing_type="10-K"),
            max_tokens=50, overlap_tokens=10)
        assert chunks
        assert all(c.token_count <= 50 for c in chunks)

    def test_overlap_must_be_smaller_than_the_chunk(self):
        """overlap >= max_tokens cannot make progress; refuse it at entry."""
        with pytest.raises(ValueError):
            chunk_sections([], max_tokens=50, overlap_tokens=100)


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


# ---------------------------------------------------------------------------
# Chunking bounds the ingest pipeline actually has to honour
# ---------------------------------------------------------------------------

def _ten_q_html() -> str:
    """A 10-Q repeats Item 1 across Part I and Part II.

    Part I: 1 Financial Statements, 2 MD&A, 3 Market Risk, 4 Controls.
    Part II: 1 Legal Proceedings, 1A Risk Factors.
    """
    body = " ".join(f"w{i}" for i in range(80))
    return f"""
    <html><body>
      <p>PART I - FINANCIAL INFORMATION</p>
      <p>Item 1. Financial Statements</p>
      <p>Condensed consolidated balance sheets. {body}</p>
      <p>Item 2. Management's Discussion and Analysis</p>
      <p>MDA_MARKER revenue increased twelve percent. {body}</p>
      <p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
      <p>MARKETRISK_MARKER interest rate exposure. {body}</p>
      <p>PART II - OTHER INFORMATION</p>
      <p>Item 1. Legal Proceedings</p>
      <p>LEGAL_MARKER the Commission fined the Company. {body}</p>
      <p>Item 1A. Risk Factors</p>
      <p>RISK_MARKER our business faces risks. {body}</p>
    </body></html>
    """


@pytest.fixture
def ten_q_filing(tmp_path):
    p = tmp_path / "AAPL_10-Q_2025-06-28.html"
    p.write_text(_ten_q_html(), encoding="utf-8")
    return p


class TestHardChunkBounds:
    """The advertised 500/100 contract must actually hold.

    Oversized paragraphs are split by sentence, and a sentence longer than the
    ceiling is never split further -- so one long run of text becomes one
    oversized chunk. Overlap keeps only whole trailing parts, so when the last
    part exceeds the overlap budget the chunks share nothing.
    """

    def test_single_oversized_sentence_is_split(self):
        from finvet.rag.parser import Section

        section = Section(item_number="7", name="mda", title="MD&A",
                          text="token " * 3000)
        chunks = chunk_sections([section], max_tokens=500, overlap_tokens=100)

        assert chunks
        assert max(c.token_count for c in chunks) <= 500

    def test_consecutive_chunks_actually_overlap(self):
        from finvet.rag.parser import Section

        text = ("alpha " * 300).strip() + "\n\n" + ("beta " * 300).strip()
        section = Section(item_number="7", name="mda", title="MD&A", text=text)
        chunks = chunk_sections([section], max_tokens=500, overlap_tokens=100)

        assert len(chunks) >= 2
        shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
        assert shared, "consecutive chunks share no tokens despite overlap_tokens=100"


class TestTenQSectionIdentity:
    """Part I and Part II both contain an "Item 1"; they are different sections.

    Section identity is the item number alone and the map is the 10-K one, so
    Part I Item 1 is labelled `business`, Part I Item 2 (MD&A) maps to
    `properties` and is dropped as non-indexable, Part I Item 3 is labelled
    `legal_proceedings`, and Part II's real Legal Proceedings heading is
    discarded as a duplicate of Item 1.
    """

    def test_mda_is_retained_and_named(self, ten_q_filing):
        sections = parse_filing_html(ten_q_filing, filing_type="10-Q")
        mda = [s for s in sections if "MDA_MARKER" in s.text]
        assert mda, "MD&A was dropped entirely"
        assert mda[0].name == "mda"

    def test_legal_proceedings_holds_the_legal_text(self, ten_q_filing):
        sections = parse_filing_html(ten_q_filing, filing_type="10-Q")
        legal = [s for s in sections if s.name == "legal_proceedings"]
        assert legal, "no legal_proceedings section"
        assert "LEGAL_MARKER" in legal[0].text
        assert "MARKETRISK_MARKER" not in legal[0].text, \
            "market-risk text was filed under legal_proceedings"

    def test_market_risk_is_named_correctly(self, ten_q_filing):
        sections = parse_filing_html(ten_q_filing, filing_type="10-Q")
        risk = [s for s in sections if "MARKETRISK_MARKER" in s.text]
        assert risk and risk[0].name == "market_risk"


# ---------------------------------------------------------------------------
# Tracked fixtures: real-shaped filings with a table of contents
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "rag"


class TestTableOfContentsDoesNotWin:
    """A contents page repeats every heading before the body.

    Taking the first occurrence of each item handed every section the few
    nodes between two adjacent contents rows, and the body text attached to
    whichever contents entry happened to be last -- a 10-K collapsed to a
    single mislabelled section. Selection is by substantive text in the span,
    which a contents row does not have.
    """

    def test_10k_sections_carry_their_body_text(self):
        sections = parse_filing_html(FIXTURE_DIR / "sample-10-k.html",
                                     filing_type="10-K")
        by_name = {s.name: s.text for s in sections}

        assert "BUSINESS_MARKER" in by_name["business"]
        assert "RISK_MARKER" in by_name["risk_factors"]
        assert "MDA_MARKER" in by_name["mda"]

    def test_10q_parts_stay_distinct(self):
        sections = parse_filing_html(FIXTURE_DIR / "sample-10-q.html",
                                     filing_type="10-Q")
        keys = {(s.part, s.item_number, s.name) for s in sections}

        assert ("I", "2", "mda") in keys
        assert ("II", "1", "legal_proceedings") in keys
        assert ("I", "1", "financial_statements_and_notes") in keys

    def test_10q_legal_proceedings_is_not_market_risk(self):
        """Both defects in one assertion: Part II Item 1 must hold the legal
        text, and Part I Item 3 must not be filed under it."""
        sections = parse_filing_html(FIXTURE_DIR / "sample-10-q.html",
                                     filing_type="10-Q")
        legal = next(s for s in sections if s.name == "legal_proceedings")

        assert "LEGAL_MARKER" in legal.text
        assert "MARKETRISK_MARKER" not in legal.text

    def test_form_type_changes_the_mapping(self):
        """The same document read as the wrong form must not silently produce
        the 10-K labels -- which is what filename inference used to do."""
        as_10q = {s.name for s in parse_filing_html(
            FIXTURE_DIR / "sample-10-q.html", filing_type="10-Q")}
        as_10k = {s.name for s in parse_filing_html(
            FIXTURE_DIR / "sample-10-q.html", filing_type="10-K")}

        assert "financial_statements_and_notes" in as_10q
        assert as_10q != as_10k

    def test_chunks_from_a_real_fixture_respect_the_ceiling(self):
        sections = parse_filing_html(FIXTURE_DIR / "sample-10-k.html",
                                     filing_type="10-K")
        chunks = chunk_sections(sections, max_tokens=60, overlap_tokens=15)

        assert chunks
        assert all(c.token_count <= 60 for c in chunks)
