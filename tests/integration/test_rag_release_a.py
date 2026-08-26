"""Filing identity and scoped retrieval, against the real migrated corpus.

Unit tests assert the contracts in isolation; only Postgres plus a live
embedder can show that a passage retrieved today carries the identity needed to
find it in the filing tomorrow, and that a query scoped to the wrong period
returns nothing rather than the nearest available text.

Opt in with -m integration; skipped when Postgres, the embedder or the corpus
is unavailable.
"""

import pytest

pytestmark = pytest.mark.integration


def _rag_or_skip():
    try:
        from finvet.rag.service import get_rag_service

        rag = get_rag_service()
        if not rag.available:
            pytest.skip("filing corpus is empty; run finvet.rag.ingest")
        return rag
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"RAG unavailable: {exc}")


@pytest.fixture(scope="module")
def rag():
    return _rag_or_skip()


class TestTheMigratedSchemaCarriesIdentity:

    def test_every_chunk_has_an_evidence_id(self):
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        with get_db_session() as session:
            total, identified, distinct = session.execute(text(
                "SELECT count(*), count(evidence_id), count(DISTINCT evidence_id) "
                "FROM filing_chunks")).fetchone()

        assert total > 0
        assert identified == total, f"{total - identified} chunks have no identity"
        assert distinct == total, "evidence ids collide, so a citation is ambiguous"

    def test_every_chunk_has_an_item_number(self):
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        with get_db_session() as session:
            missing = session.execute(text(
                "SELECT count(*) FROM filing_chunks WHERE item_number IS NULL"
            )).scalar()

        assert missing == 0

    def test_10q_chunks_carry_a_part_and_10k_chunks_do_not(self):
        """The ambiguity the part resolves: Item 1 is Business in a 10-K,
        Financial Statements in 10-Q Part I, and Legal Proceedings in Part II."""
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        with get_db_session() as session:
            rows = session.execute(text(
                "SELECT filing_type, part, count(*) FROM filing_chunks "
                "GROUP BY 1, 2")).fetchall()

        by_form = {}
        for filing_type, part, count in rows:
            by_form.setdefault(filing_type, set()).add(part)

        assert by_form.get("10-Q"), "no 10-Q chunks in the corpus"
        assert by_form["10-Q"] - {None}, "10-Q chunks carry no part"

    def test_the_same_item_number_appears_in_both_parts(self):
        """Proof the part is load-bearing rather than decorative."""
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        with get_db_session() as session:
            rows = session.execute(text(
                "SELECT item_number, count(DISTINCT part) FROM filing_chunks "
                "WHERE filing_type = '10-Q' AND part IS NOT NULL "
                "GROUP BY 1 HAVING count(DISTINCT part) > 1")).fetchall()

        assert rows, (
            "no item number appears in more than one part, so this corpus "
            "cannot demonstrate the ambiguity the part exists to resolve")


class TestRetrievedEvidenceIsAttributable:

    def test_a_result_carries_everything_needed_to_find_it_again(self, rag):
        results = rag.search(query="risk factors affecting our business",
                             ticker="AAPL", top_k=3)

        assert results, "the corpus should answer a plainly on-topic query"
        top = results[0]
        assert len(top.evidence_id) == 64
        assert top.ticker == "AAPL"
        assert top.cik
        assert top.filing_type in {"10-K", "10-Q"}
        assert top.filing_date and top.period_end
        assert top.section and top.section_title
        assert isinstance(top.chunk_index, int)
        assert len(top.content_sha256) == 64
        assert top.hash_scope == "filing_chunks.chunk_text"
        assert top.evidence_role == "supporting"

    def test_the_content_hash_matches_the_stored_row(self, rag):
        """Recomputable from the database, which is what makes it checkable."""
        from sqlalchemy import text

        from finvet.config.database import get_db_session
        from finvet.rag.service import content_sha256

        top = rag.search(query="risk factors affecting our business",
                         ticker="AAPL", top_k=1)[0]

        with get_db_session() as session:
            stored = session.execute(text(
                "SELECT chunk_text FROM filing_chunks WHERE evidence_id = :e"),
                {"e": top.evidence_id}).scalar()

        assert stored is not None, "the evidence id does not resolve to a row"
        assert content_sha256(stored) == top.content_sha256

    def test_the_evidence_id_resolves_to_exactly_one_row(self, rag):
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        top = rag.search(query="revenue and operating income", ticker="MSFT",
                         top_k=1)[0]

        with get_db_session() as session:
            count = session.execute(text(
                "SELECT count(*) FROM filing_chunks WHERE evidence_id = :e"),
                {"e": top.evidence_id}).scalar()

        assert count == 1

    def test_scores_and_ranks_are_distinguishable(self, rag):
        """keyword_score is a ts_rank value; keyword_rank is a position."""
        results = rag.search(query="cloud services revenue growth",
                             ticker="MSFT", top_k=5)

        assert results
        for result in results:
            assert isinstance(result.vector_similarity, float)
            assert isinstance(result.keyword_score, float)
            assert result.vector_rank is None or isinstance(result.vector_rank, int)
            assert result.keyword_rank is None or isinstance(result.keyword_rank, int)
            assert isinstance(result.rrf_score, float)


class TestRetrievalIsScopedToTheResolvedPeriod:

    def test_a_wrong_period_returns_nothing(self, rag):
        """On topic, wrong year. The relevance floor cannot catch this; the
        period filter must, or the agent reads the wrong fiscal year."""
        results = rag.search(
            query="risks from supplier concentration",
            ticker="AAPL", period_end="2019-09-28", top_k=5)

        assert results == []

    def test_a_wrong_form_returns_nothing(self, rag):
        results = rag.search(query="annual risk factors discussion",
                             ticker="AAPL", filing_type="10-Q",
                             period_end="2025-09-27", top_k=5)

        assert results == []

    def test_the_right_period_still_answers(self, rag):
        """The control: without it the two assertions above pass on a corpus
        that answers nothing at all."""
        from sqlalchemy import text

        from finvet.config.database import get_db_session

        with get_db_session() as session:
            period = session.execute(text(
                "SELECT period_end FROM filing_chunks "
                "WHERE ticker = 'AAPL' AND filing_type = '10-K' LIMIT 1"
            )).scalar()

        results = rag.search(query="risks from supplier concentration",
                             ticker="AAPL", filing_type="10-K",
                             period_end=period, top_k=5)

        assert results, f"no evidence for AAPL 10-K {period}"


class TestAnEmptyCorpusIsDistinctFromSilence:

    def test_an_unindexed_issuer_reports_no_corpus(self, rag):
        """A filing that was searched and said nothing, and no filing at all,
        are opposite facts. Collapsing them let a delegation report an unread
        corpus as a filing that stayed silent."""
        from finvet.tools import filing_search

        out = filing_search.search_filing_text.invoke(
            {"query": "any disclosure at all", "ticker": "ZZZZ"})

        assert out["success"] is True
        assert out["chunks"] == []
        assert out["reason"] == "no_corpus"

    def test_an_indexed_issuer_with_no_match_reports_silence(self, rag):
        from finvet.tools import filing_search

        out = filing_search.search_filing_text.invoke(
            {"query": "sourdough bread starter hydration ratios",
             "ticker": "AAPL"})

        assert out["success"] is True
        assert out["chunks"] == []
        assert out["reason"] == "no_relevant_evidence"

    def test_the_corpus_check_knows_an_indexed_issuer(self, rag):
        assert rag.has_filings_for("AAPL") is True
        assert rag.has_filings_for("ZZZZ") is False
