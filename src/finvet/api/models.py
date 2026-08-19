"""Pydantic request/response models for the FinVet API."""

from typing import Optional
from pydantic import BaseModel, Field


class VerifyClaimRequest(BaseModel):
    """Request model for claim verification."""
    claim: str = Field(..., description="Financial claim to verify", min_length=10, max_length=2000)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    memory_context: Optional[dict] = Field(None, description="Prior verification to use as agent context")


class MemoryCheckRequest(BaseModel):
    """Request model for pre-pipeline memory check."""
    claim: str = Field(..., min_length=10, max_length=2000)


class MemoryAcceptRequest(BaseModel):
    """Request model for accepting a cached verification result."""
    original_request_id: str = Field(..., description="Request ID of the cached verification")
    claim: str = Field(..., description="Current claim text")
    similarity: float = Field(..., description="Cosine similarity score")


class HITLReviewRequest(BaseModel):
    """Request model for HITL review submission."""
    decision: str = Field(..., description="Decision: 'approve', 'override', or 'reject'")
    override_verdict: Optional[str] = Field(None, description="New verdict if decision is 'override'")
    reviewer_notes: Optional[str] = Field(None, description="Optional notes from reviewer")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
