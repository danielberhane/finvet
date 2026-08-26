"""The typed contract for one retrieved filing passage.

Search used to return bare dicts, and two things went wrong in them that a type
would have caught.

`ts_rank` returns a relevance *score*; it was emitted as `keyword_rank`, a name
that promises an ordinal position. Anything reading it as "the 1st keyword hit"
was reading a float around 0.06 that never meant that. Meanwhile the dense
arm's actual position and the fused RRF score were not emitted at all, so a
reader could not see how a result had been reached.

Scores and ranks are separate fields here, and the ranks are integers, so the
confusion is not expressible.

`evidence_role` is fixed to "supporting" rather than being a caller's choice.
Filing text can show what a company said; it cannot become the number a verdict
rests on. Release A enforces that at the trust boundary, and this field records
it in the audit trail.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RAGSearchResult(BaseModel):
    """One passage, with enough identity to find it in the filing again."""

    # Stable across re-ingestion, unlike the autoincrement row id.
    evidence_id: str = Field(..., description="Content-derived passage identity")

    ticker: str
    cik: str
    filing_type: str = Field(..., description="10-K, 10-Q, ...")
    filing_date: str
    period_end: str

    # A 10-Q restarts item numbering in each part, so an item number without
    # its part names two different sections. None for a 10-K, which has no
    # parts -- distinct from "", which would read as a part that is empty.
    part: Optional[str] = Field(None, description="'I'/'II' for a 10-Q")
    item_number: str = Field("", description="'1A', '2', ...")

    section: str
    section_title: str
    chunk_index: int

    chunk_text: str
    content_sha256: str = Field(..., description="Digest of the stored text")
    # Named so a reader knows which bytes to hash to check it. The tool wraps
    # the model-facing excerpt in <filing_excerpt> tags; the hash covers the
    # stored column, not the wrapper.
    hash_scope: Literal["filing_chunks.chunk_text"] = "filing_chunks.chunk_text"

    # Raw arm signals. RRF discards the underlying strengths when it fuses, so
    # without these nothing downstream can judge how strong a hit really was.
    vector_similarity: float = Field(..., description="Cosine similarity, 0-1")
    vector_rank: Optional[int] = Field(
        None, description="1-indexed position in the dense arm, after the floor")
    keyword_score: float = Field(..., description="PostgreSQL ts_rank value")
    keyword_rank: Optional[int] = Field(
        None, description="1-indexed position in the lexical arm")
    rrf_score: float = Field(..., description="Fused reciprocal-rank score")

    evidence_role: Literal["supporting"] = "supporting"
