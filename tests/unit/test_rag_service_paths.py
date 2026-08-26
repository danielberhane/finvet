"""Ingestion, availability, and what happens when the corpus or embedder is down.

`search` has its degradation path tested elsewhere. These are the paths around
it: deciding whether the service can answer at all, deciding whether an issuer
has any indexed filing, and the ingest pass that gives every chunk the identity
a citation later resolves.

The recurring contract is that **unknown is not empty**. A corpus check that
cannot reach the database must not report "no filings for this issuer" — that
answer would be handed to the delegation as `NO_CORPUS`, which reads as a
settled fact about the issuer rather than a failure to look.

Sessions and HTTP are mocked: the behaviour under test is what the service does
when its dependencies misbehave, which live ones will not do on request. The
happy paths run against real Postgres and Ollama in
`tests/integration/test_rag_release_a.py`.
"""

from unittest.mock import MagicMock, patch

import pytest

from finvet.rag.service import RAGService


def _service():
    service = RAGService.__new__(RAGService)
    service._table_ready = True
    service._tsv_ready = True
    return service


def _session(**behaviour):
    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = lambda s, *a: False
    for attr, value in behaviour.items():
        setattr(session, attr, value)
    return session


def _broken(exc=RuntimeError("connection reset")):
    def boom():
        raise exc
    return boom


class TestAvailabilityIsAboutTheCorpusNotTheEmbedder:
    """Search degrades to keyword-only without embeddings, so an embedder
    outage does not make the service useless. An empty corpus does."""

    def test_a_populated_corpus_is_available(self):
        session = _session()
        session.query.return_value.scalar.return_value = 1398
        with patch("finvet.rag.service.get_db_session", lambda: session):
            assert _service().available is True

    def test_an_empty_corpus_is_not_available(self):
        session = _session()
        session.query.return_value.scalar.return_value = 0
        with patch("finvet.rag.service.get_db_session", lambda: session):
            assert _service().available is False

    def test_an_unreachable_store_is_not_available(self):
        with patch("finvet.rag.service.get_db_session", _broken()):
            assert _service().available is False


class TestTheCorpusCheckTreatsUnknownAsPresent:
    """`has_filings_for` decides whether an empty search means "the filing says
    nothing" or "there is no filing". Getting that backwards turns a failed
    lookup into a claim about the issuer."""

    def test_an_indexed_issuer_is_reported(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = (1,)
        with patch("finvet.rag.service.get_db_session", lambda: session):
            assert _service().has_filings_for("AAPL") is True

    def test_an_unindexed_issuer_is_reported(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = None
        with patch("finvet.rag.service.get_db_session", lambda: session):
            assert _service().has_filings_for("ZZZZ") is False

    def test_a_failed_lookup_does_not_claim_an_empty_corpus(self):
        """The asymmetry that matters. Returning False here would report
        NO_CORPUS -- a settled fact about the issuer -- on a database blip."""
        with patch("finvet.rag.service.get_db_session", _broken()):
            assert _service().has_filings_for("AAPL") is True

    def test_the_ticker_is_matched_case_insensitively(self):
        session = _session()
        session.query.return_value.filter.return_value.first.return_value = (1,)
        with patch("finvet.rag.service.get_db_session", lambda: session):
            _service().has_filings_for("aapl")

        criteria = " ".join(
            str(c.compile(compile_kwargs={"literal_binds": True}))
            for c in session.query.return_value.filter.call_args.args)
        assert "AAPL" in criteria, criteria


class TestIngestionGivesEveryChunkAnIdentity:

    def _ingest(self, monkeypatch, *, existing=None, sections=None, chunks=None):
        from finvet.rag import service as svc

        session = _session()
        session.query.return_value.filter_by.return_value.first.return_value = existing
        added = []
        session.add = added.append

        monkeypatch.setattr(svc, "get_db_session", lambda: session)
        monkeypatch.setattr(svc, "parse_filing_html",
                            lambda *a, **k: sections if sections is not None else [object()])
        monkeypatch.setattr(svc, "chunk_sections",
                            lambda *a, **k: chunks if chunks is not None else [])
        monkeypatch.setattr(svc, "_embed_texts",
                            lambda texts: [[0.1] * 768 for _ in texts])

        count = _service().ingest_filing(
            filepath="x.html", ticker="aapl", cik="0000320193",
            filing_type="10-Q", filing_date="2024-08-02",
            period_end="2024-06-29")
        return count, added

    def _chunk(self, index=0, part="I", item="2", text="Revenue grew."):
        chunk = MagicMock()
        chunk.text = text
        chunk.section_name = "mda"
        chunk.section_title = "MD&A"
        chunk.chunk_index = index
        chunk.token_count = 5
        chunk.part = part
        chunk.item_number = item
        return chunk

    def test_an_already_ingested_filing_is_skipped(self, monkeypatch):
        count, added = self._ingest(monkeypatch, existing=object())
        assert count == 0
        assert added == []

    def test_a_filing_with_no_sections_produces_nothing(self, monkeypatch):
        count, added = self._ingest(monkeypatch, sections=[])
        assert count == 0
        assert added == []

    def test_a_filing_that_chunks_to_nothing_produces_nothing(self, monkeypatch):
        count, added = self._ingest(monkeypatch, chunks=[])
        assert count == 0
        assert added == []

    def test_every_stored_chunk_carries_its_position_and_identity(self, monkeypatch):
        count, added = self._ingest(
            monkeypatch, chunks=[self._chunk(0), self._chunk(1)])

        assert count == 2
        assert len(added) == 2
        for record in added:
            assert record.part == "I"
            assert record.item_number == "2"
            assert len(record.evidence_id) == 64
            assert record.ticker == "AAPL", "the ticker is normalised"

    def test_two_chunks_of_one_filing_get_distinct_identities(self, monkeypatch):
        _, added = self._ingest(
            monkeypatch, chunks=[self._chunk(0), self._chunk(1)])
        assert added[0].evidence_id != added[1].evidence_id

    def test_the_same_chunk_ingested_twice_gets_the_same_identity(self, monkeypatch):
        _, first = self._ingest(monkeypatch, chunks=[self._chunk(0)])
        _, second = self._ingest(monkeypatch, chunks=[self._chunk(0)])
        assert first[0].evidence_id == second[0].evidence_id, (
            "a re-ingested corpus must not renumber its citations")


class TestStatsAndAccessor:

    def test_get_stats_summarises_the_corpus(self):
        session = _session()
        session.query.return_value.scalar.return_value = 1398
        # Two different queries share this mock: tickers returns 1-tuples,
        # filings returns (ticker, type, period). Sequenced so each gets the
        # shape it actually unpacks.
        session.query.return_value.distinct.return_value.all.side_effect = [
            [("AAPL",), ("MSFT",)],
            [("AAPL", "10-K", "2025-09-27"), ("MSFT", "10-Q", "2025-09-30")],
        ]
        with patch("finvet.rag.service.get_db_session", lambda: session):
            stats = _service().get_stats()

        assert stats["total_chunks"] == 1398
        assert "AAPL" in stats["tickers"]
        assert {"ticker": "AAPL", "type": "10-K",
                "period": "2025-09-27"} in stats["filings"]

    def test_the_accessor_returns_one_shared_service(self):
        from finvet.rag.service import get_rag_service

        assert get_rag_service() is get_rag_service()


class TestEmbeddingFailuresAreLoud:
    """A wrong-dimension or short batch would corrupt the index silently."""

    def _post(self, payload):
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = lambda: None
        return response

    def test_a_short_batch_raises(self, monkeypatch):
        from finvet.rag import service as svc

        monkeypatch.setattr(svc.httpx, "post",
                            lambda *a, **k: self._post({"embeddings": [[0.1] * 768]}))
        with pytest.raises(RuntimeError, match="embeddings"):
            svc._embed_texts(["a", "b"])

    def test_a_wrong_dimension_raises(self, monkeypatch):
        from finvet.rag import service as svc

        monkeypatch.setattr(svc.httpx, "post",
                            lambda *a, **k: self._post({"embeddings": [[0.1] * 512]}))
        with pytest.raises(RuntimeError, match="dim"):
            svc._embed_texts(["a"])

    def test_embed_single_returns_one_vector(self, monkeypatch):
        from finvet.rag import service as svc

        monkeypatch.setattr(svc.httpx, "post",
                            lambda *a, **k: self._post({"embeddings": [[0.1] * 768]}))
        assert len(svc._embed_single("a")) == 768
