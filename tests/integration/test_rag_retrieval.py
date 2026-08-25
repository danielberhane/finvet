"""Retrieval behaviour against a real corpus. Opt in with -m integration.

Skipped unless Postgres holds ingested chunks and the Ollama embedder answers,
so the default unit run stays hermetic. What the unit tests cannot show is
whether the calibrated floor and the period filter behave on real embeddings
rather than on mocks, which is exactly where a retrieval change goes wrong.
"""

import pytest

pytestmark = pytest.mark.integration


def _service_or_skip():
    try:
        from finvet.rag.service import get_rag_service

        rag = get_rag_service()
        if not rag.available:
            pytest.skip("RAG service unavailable")
        if not rag.search(query="revenue", ticker="AAPL", top_k=1,
                          min_vector_similarity=-1.0):
            pytest.skip("no ingested corpus")
        return rag
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"RAG unavailable: {exc}")


class TestRelevanceFloorOnRealEmbeddings:

    def test_a_real_disclosure_is_found(self):
        rag = _service_or_skip()
        hits = rag.search(query="risks from supplier concentration",
                          ticker="AAPL", top_k=5)
        assert hits
        assert all(h["vector_similarity"] > 0 for h in hits)

    def test_an_unrelated_query_returns_nothing(self):
        """The floor's whole purpose: a subject no filing discusses must not
        come back with the closest available passages."""
        rag = _service_or_skip()
        assert rag.search(query="beginner watercolour brush techniques",
                          ticker="AAPL", top_k=5) == []

    def test_every_hit_clears_the_calibrated_floor(self):
        from finvet.config.constants import RAG_MIN_VECTOR_SIMILARITY

        rag = _service_or_skip()
        for query in ("data center revenue", "cybersecurity risk",
                      "segment operating income"):
            for hit in rag.search(query=query, ticker="MSFT", top_k=5):
                assert hit["vector_similarity"] >= RAG_MIN_VECTOR_SIMILARITY


class TestPeriodScope:

    def test_wrong_period_returns_nothing(self):
        rag = _service_or_skip()
        assert rag.search(query="revenue", ticker="AAPL",
                          period_end="1990-01-01", top_k=5) == []

    def test_right_period_is_returned(self):
        rag = _service_or_skip()
        any_hit = rag.search(query="revenue", ticker="AAPL", top_k=1)
        if not any_hit:
            pytest.skip("no AAPL evidence above the floor")
        period = any_hit[0]["period_end"]

        scoped = rag.search(query="revenue", ticker="AAPL",
                            period_end=period, top_k=5)
        assert scoped
        assert {h["period_end"] for h in scoped} == {period}


class TestEvidenceIdentity:

    def test_results_carry_enough_to_find_them_again(self):
        rag = _service_or_skip()
        hits = rag.search(query="legal proceedings", ticker="AAPL", top_k=3)
        if not hits:
            pytest.skip("no AAPL legal-proceedings evidence above the floor")

        for hit in hits:
            for field in ("chunk_id", "ticker", "cik", "filing_type",
                          "period_end", "section", "chunk_index",
                          "vector_similarity", "keyword_rank", "score",
                          "content_sha256"):
                assert hit.get(field) is not None, f"missing {field}"

    def test_content_hash_matches_the_text(self):
        import hashlib

        rag = _service_or_skip()
        hits = rag.search(query="revenue", ticker="AAPL", top_k=2)
        if not hits:
            pytest.skip("no AAPL evidence above the floor")

        for hit in hits:
            expected = hashlib.sha256(
                hit["chunk_text"].encode("utf-8")).hexdigest()
            assert hit["content_sha256"] == expected


class TestKeywordOnlyDegradation:

    def test_search_survives_an_embedder_outage(self, monkeypatch):
        """Losing the embedder costs the dense arm, not the whole query."""
        from finvet.rag import service as svc

        rag = _service_or_skip()
        monkeypatch.setattr(
            svc, "_embed_single",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ollama down")))

        hits = rag.search(query="revenue", ticker="AAPL", top_k=5)
        assert all(h["vector_similarity"] == 0.0 for h in hits)
