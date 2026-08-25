"""Pydantic request/response models for the FinVet API."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


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
    """Request model for HITL review submission.

    Validation lives here rather than in the route: a malformed decision should
    be rejected by the contract before any audit event is written or any
    pending row is claimed.
    """

    decision: Literal["approve", "override", "reject"] = Field(
        ..., description="Reviewer's decision")
    override_verdict: Optional[Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]] = Field(
        None, description="New verdict; required for, and only for, 'override'")
    reviewer_notes: Optional[str] = Field(
        None, max_length=2000, description="Optional notes from reviewer")

    @model_validator(mode="after")
    def validate_override(self):
        """A verdict without an override is as wrong as an override without one.

        The first silently discards the reviewer's intent; the second leaves the
        route guessing what they meant.
        """
        if (self.decision == "override") != (self.override_verdict is not None):
            raise ValueError(
                "override_verdict is required for decision='override' "
                "and must be omitted otherwise")
        return self


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
