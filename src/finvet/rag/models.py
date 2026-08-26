"""Database model for SEC filing chunks (pgvector)."""

from sqlalchemy import Column, Integer, String, Text, Index
from pgvector.sqlalchemy import Vector

from ..config.constants import EMBEDDING_DIMS
from ..config.database import Base


class FilingChunk(Base):
    """A text chunk from an SEC filing, stored with its embedding for RAG search."""

    __tablename__ = "filing_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    cik = Column(String(20), nullable=False)
    filing_type = Column(String(10), nullable=False)        # '10-K', '10-Q'
    filing_date = Column(String(20), nullable=False)
    period_end = Column(String(20), nullable=False)
    section = Column(String(50), nullable=False, index=True) # 'risk_factors', 'mda', etc.
    section_title = Column(String(100), nullable=False)      # 'Risk Factors', 'MD&A', etc.
    # Where in the filing. A 10-Q restarts item numbering in each part, so an
    # item number without its part names two different sections.
    part = Column(String(10), nullable=True)                 # 'I'/'II'; None for a 10-K
    item_number = Column(String(10), nullable=True)          # '1A', '2', ...
    chunk_index = Column(Integer, nullable=False)
    # Derived from what the chunk is, not from insertion order: the
    # autoincrement id changes on every re-ingestion, so an audit record citing
    # it would point at a different passage after the next corpus rebuild.
    evidence_id = Column(String(64), nullable=True, unique=True, index=True)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMS))               # pgvector cosine search

    __table_args__ = (
        Index("idx_filing_chunks_ticker_period", "ticker", "filing_type", "period_end"),
        Index(
            "idx_filing_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
