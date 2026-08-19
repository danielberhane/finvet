"""Guard provider protocol and result model."""

from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class GuardResult(BaseModel):
    """Result from a guard classification check."""

    safe: bool
    categories: list[str] = Field(default_factory=list)
    scrubbed_text: Optional[str] = None
    flags: list[str] = Field(default_factory=list)
    violation_type: Optional[str] = None
    provider: str
    latency_ms: float = 0.0


@runtime_checkable
class GuardProvider(Protocol):
    """Protocol that all guard providers must implement."""

    def classify_input(self, text: str) -> GuardResult: ...

    def classify_output(self, response_text: str, original_claim: str) -> GuardResult: ...
