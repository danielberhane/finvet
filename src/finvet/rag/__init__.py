"""RAG module for SEC filing search.

Provides ingestion and hybrid search (vector + keyword) over SEC filing
text using pgvector and PostgreSQL full-text search.
"""

from .models import FilingChunk
from .service import RAGService, get_rag_service

__all__ = [
    "FilingChunk",
    "RAGService",
    "get_rag_service",
]
