"""Database model for SEC filing chunks (pgvector)."""

from sqlalchemy import Column, Integer, String, Text, Index
from pgvector.sqlalchemy import Vector

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
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding = Column(Vector(1536))                         # pgvector cosine search

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
