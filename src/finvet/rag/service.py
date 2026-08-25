"""RAG service for SEC filing search.

Handles ingestion (parse → chunk → embed → store) and hybrid search
(vector similarity + full-text keyword search with Reciprocal Rank Fusion).
"""

import hashlib
from typing import Optional
from pathlib import Path

import httpx
from sqlalchemy import text, func

from ..config.constants import (
    EMBED_BATCH_SIZE,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    RAG_MIN_VECTOR_SIMILARITY,
    RRF_K,
)
from ..config.database import get_db_session, Base, engine
from ..config.settings import settings
from ..utils.logging import get_logger
from .models import FilingChunk
from .parser import parse_filing_html, chunk_sections

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Embeddings — served locally by Ollama, so no API key and no per-call billing.
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the local Ollama embedding model."""
    response = httpx.post(
        f"{settings.ollama_url.rstrip('/')}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=settings.embedding_timeout_s,
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings") or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"{EMBEDDING_MODEL} returned {len(embeddings)} embeddings "
            f"for {len(texts)} inputs"
        )
    if embeddings and len(embeddings[0]) != EMBEDDING_DIMS:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} returned {len(embeddings[0])}-dim vectors, "
            f"but filing_chunks.embedding is vector({EMBEDDING_DIMS})"
        )
    return embeddings


def _embed_single(text: str) -> list[float]:
    """Embed a single text."""
    return _embed_texts([text])[0]


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

def rrf_component(rank: int | None) -> float:
    """One arm's contribution to a chunk's fused score.

    Absence contributes nothing. The previous code substituted a fixed distant
    rank (1000), which handed every unmatched document the same small positive
    score -- a participation credit for not matching, and one that mattered
    more as the candidate pool grew.

    Module level rather than a closure so a test can exercise this function
    instead of re-implementing it.
    """
    return 0.0 if rank is None else 1.0 / (RRF_K + rank)


class RAGService:
    """Handles ingestion and hybrid search over SEC filing chunks."""

    def __init__(self):
        self._table_ready = False
        self._tsv_ready = False

    @property
    def available(self) -> bool:
        """True when search can return results at all.

        Deliberately not a check on the embedder: search degrades to keyword-only
        when embeddings are down, so the service is still useful. It is the corpus
        being empty that makes every query pointless.
        """
        try:
            with get_db_session() as session:
                return (session.query(func.count(FilingChunk.id)).scalar() or 0) > 0
        except Exception as e:
            logger.warning(f"RAG availability check failed: {e}")
            return False

    def _ensure_table(self):
        """Create the filing_chunks table if it doesn't exist."""
        if self._table_ready:
            return
        Base.metadata.create_all(bind=engine, tables=[FilingChunk.__table__])

        # Add tsvector generated column and GIN index for full-text search.
        # These can't be declared in SQLAlchemy ORM easily, so we use raw SQL.
        with get_db_session() as session:
            # Check if tsv column already exists
            result = session.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'filing_chunks' AND column_name = 'tsv'"
            )).fetchone()

            if not result:
                session.execute(text(
                    "ALTER TABLE filing_chunks ADD COLUMN tsv tsvector "
                    "GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED"
                ))
                session.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_filing_chunks_tsv "
                    "ON filing_chunks USING gin(tsv)"
                ))
                logger.info("Created tsvector column and GIN index on filing_chunks")

        self._table_ready = True

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_filing(
        self,
        filepath: str | Path,
        ticker: str,
        cik: str,
        filing_type: str,
        filing_date: str,
        period_end: str,
    ) -> int:
        """Parse, chunk, embed, and store a single SEC filing.

        Returns the number of chunks created. Skips if already ingested.
        """
        self._ensure_table()

        # Check if this filing is already ingested
        with get_db_session() as session:
            existing = session.query(FilingChunk).filter_by(
                ticker=ticker, filing_type=filing_type, period_end=period_end,
            ).first()
            if existing:
                logger.info(f"Skipping {ticker} {filing_type} {period_end} — already ingested")
                return 0

        # Parse and chunk
        sections = parse_filing_html(filepath, filing_type=filing_type)
        if not sections:
            logger.warning(f"No sections found in {filepath}")
            return 0

        chunks = chunk_sections(sections)
        if not chunks:
            logger.warning(f"No chunks produced from {filepath}")
            return 0

        logger.info(f"Parsed {len(sections)} sections, {len(chunks)} chunks from {filepath}")

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            batch_texts = [c.text for c in batch]
            embeddings = _embed_texts(batch_texts)
            all_embeddings.extend(embeddings)
            logger.info(f"  Embedded batch {i // EMBED_BATCH_SIZE + 1} ({len(batch)} chunks)")

        # Store in database
        with get_db_session() as session:
            for chunk, embedding in zip(chunks, all_embeddings):
                record = FilingChunk(
                    ticker=ticker.upper(),
                    cik=cik,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    period_end=period_end,
                    section=chunk.section_name,
                    section_title=chunk.section_title,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                session.add(record)

        logger.info(f"Stored {len(chunks)} chunks for {ticker} {filing_type} {period_end}")
        return len(chunks)

    # ------------------------------------------------------------------
    # Hybrid Search (Vector + BM25 with Reciprocal Rank Fusion)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        ticker: str | None = None,
        section: str | None = None,
        filing_type: str | None = None,
        period_end: str | None = None,
        period_start: str | None = None,
        period_end_max: str | None = None,
        top_k: int = 5,
        min_vector_similarity: float | None = None,
    ) -> list[dict]:
        """Search filing chunks using hybrid vector + keyword search.

        Combines pgvector cosine similarity with PostgreSQL full-text search
        using Reciprocal Rank Fusion (RRF) for robust retrieval.

        Args:
            query: The search query text.
            ticker: Filter by company ticker (e.g., "AAPL").
            section: Filter by section name (e.g., "risk_factors", "mda").
            filing_type: Filter by filing type (e.g., "10-K", "10-Q").
            period_end: Exact filing period to restrict to.
            period_start: Earliest acceptable period_end (inclusive).
            period_end_max: Latest acceptable period_end (inclusive).
            top_k: Number of results to return.
            min_vector_similarity: Floor for the dense arm; defaults to
                RAG_MIN_VECTOR_SIMILARITY. A chunk below it is not evidence.

        The period filters exist because a chunk from the wrong fiscal year is
        not weak evidence, it is the wrong evidence, and nothing downstream can
        tell. The resolved retrieval scope supplies them; the model cannot
        substitute an unrelated date.

        Returns:
            List of dicts with chunk_text, section, filing_type, period_end,
            ticker, section_title, and score.
        """
        self._ensure_table()

        # The keyword arm is pure Postgres and needs no embedding. Letting an
        # embedder outage propagate would take it down too, so degrade to
        # keyword-only search instead of failing the whole query.
        try:
            query_embedding = _embed_single(query)
        except Exception as e:
            logger.warning(
                f"Embedding unavailable ({type(e).__name__}: {e}) — "
                f"falling back to keyword-only search"
            )
            query_embedding = None

        # Build WHERE clause filters
        filters = []
        params = {}
        if ticker:
            filters.append("ticker = :ticker")
            params["ticker"] = ticker.upper()
        if section:
            filters.append("section = :section")
            params["section"] = section
        if filing_type:
            filters.append("filing_type = :filing_type")
            params["filing_type"] = filing_type
        if period_end:
            filters.append("period_end = :period_end")
            params["period_end"] = period_end
        if period_start:
            filters.append("period_end >= :period_start")
            params["period_start"] = period_start
        if period_end_max:
            filters.append("period_end <= :period_end_max")
            params["period_end_max"] = period_end_max

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        # Vector search: top 20 by cosine similarity
        # Use CAST() instead of ::vector to avoid conflict with SQLAlchemy :param syntax
        vec_sql = text(f"""
            SELECT id, chunk_text, section, section_title, filing_type, period_end,
                   ticker, cik, chunk_index,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS vec_score
            FROM filing_chunks
            {where_clause}
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT 20
        """)

        # Keyword search: top 20 by ts_rank
        kw_filter = "AND" if filters else "WHERE"
        kw_sql = text(f"""
            SELECT id, chunk_text, section, section_title, filing_type, period_end,
                   ticker, cik, chunk_index,
                   ts_rank(tsv, plainto_tsquery('english', :query_text)) AS kw_score
            FROM filing_chunks
            {where_clause}
            {kw_filter} tsv @@ plainto_tsquery('english', :query_text)
            ORDER BY ts_rank(tsv, plainto_tsquery('english', :query_text)) DESC
            LIMIT 20
        """)

        with get_db_session() as session:
            if query_embedding is not None:
                vec_params = {**params, "query_vec": str(query_embedding)}
                vec_results = session.execute(vec_sql, vec_params).fetchall()
            else:
                vec_results = []

            kw_params = {**params, "query_text": query}
            kw_results = session.execute(kw_sql, kw_params).fetchall()

        # Step 5: the dense arm always returns a nearest neighbour, however far
        # away. Without a floor, a ticker with any corpus yields passages for a
        # disclosure that does not exist, and the tool reports success. The
        # lexical arm needs no floor: tsv @@ plainto_tsquery is already a
        # predicate, so a row only appears if it matched.
        floor = (RAG_MIN_VECTOR_SIMILARITY if min_vector_similarity is None
                 else min_vector_similarity)
        vec_scores = {row.id: float(row.vec_score) for row in vec_results}
        vec_results = [row for row in vec_results if float(row.vec_score) >= floor]
        kw_scores = {row.id: float(row.kw_score) for row in kw_results}

        # Build rank maps (id → rank position, 1-indexed)
        vec_ranks = {row.id: rank + 1 for rank, row in enumerate(vec_results)}
        kw_ranks = {row.id: rank + 1 for rank, row in enumerate(kw_results)}

        # Only chunks that cleared an arm are candidates.
        all_ids = set(vec_ranks.keys()) | set(kw_ranks.keys())

        # Build result lookup from both result sets
        result_map = {}
        for row in vec_results:
            result_map[row.id] = row
        for row in kw_results:
            if row.id not in result_map:
                result_map[row.id] = row
        result_map = {cid: row for cid, row in result_map.items() if cid in all_ids}

        scored = []
        for chunk_id in all_ids:
            rrf_score = (rrf_component(vec_ranks.get(chunk_id))
                         + rrf_component(kw_ranks.get(chunk_id)))
            scored.append((chunk_id, rrf_score))

        # Sort by RRF score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = scored[:top_k]

        # Format output
        output = []
        for chunk_id, score in top_results:
            row = result_map[chunk_id]
            output.append({
                # Identity: enough to find this passage in the filing again
                # without trusting the pipeline's copy of it.
                "chunk_id": chunk_id,
                "cik": row.cik,
                "chunk_index": row.chunk_index,
                "content_sha256": hashlib.sha256(
                    row.chunk_text.encode("utf-8")).hexdigest(),
                "chunk_text": row.chunk_text,
                "section": row.section,
                "section_title": row.section_title,
                "filing_type": row.filing_type,
                "period_end": row.period_end,
                "ticker": row.ticker,
                # Raw arm scores alongside the fused rank: RRF discards them,
                # and without them nothing downstream can judge how strong a
                # result actually was.
                "vector_similarity": round(vec_scores.get(chunk_id, 0.0), 6),
                "keyword_rank": round(kw_scores.get(chunk_id, 0.0), 6),
                "score": round(score, 6),
            })

        return output

    def get_stats(self) -> dict:
        """Get summary statistics about ingested filings."""
        self._ensure_table()
        with get_db_session() as session:
            total = session.query(func.count(FilingChunk.id)).scalar() or 0
            tickers = session.query(FilingChunk.ticker).distinct().all()
            filings = session.query(
                FilingChunk.ticker, FilingChunk.filing_type, FilingChunk.period_end
            ).distinct().all()

        return {
            "total_chunks": total,
            "tickers": [t[0] for t in tickers],
            "filings": [
                {"ticker": f[0], "type": f[1], "period": f[2]}
                for f in filings
            ],
        }


# ---------------------------------------------------------------------------
# Singleton accessor (same pattern as memory/service.py)
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get the singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
