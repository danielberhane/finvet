"""Data models for claims and parsed claim information."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ParsedClaim(BaseModel):
    """
    Simplified parsed claim with 6 fields.

    This is the output from the fine-tuned Qwen parser.
    The agent will infer the specific metric from the claim text.
    """

    claim_type: Literal["sec", "market", "news", "reject"] = Field(
        ...,
        description="Claim category for routing"
    )

    ticker: Optional[str] = Field(
        None,
        description="Stock ticker symbol (e.g., 'AAPL', 'TSLA')"
    )

    value: Optional[float] = Field(
        None,
        description="Numeric value being claimed (e.g., 94000000000 for $94B)"
    )

    comparison: Optional[Literal["eq", "gt", "gte", "lt", "lte"]] = Field(
        None,
        description="Comparison operator: eq (equals/was/is), gt (exceeds/above/more than), gte (at least), lt (below/under/less than), lte (at most)"
    )

    period: Optional[str] = Field(
        None,
        description="Time period as mentioned (e.g., 'Q4 FY2024', 'fiscal 2023')"
    )

    currency: Optional[str] = Field(
        None,
        description="Currency code (e.g., 'USD', 'EUR', 'JPY', 'KRW')"
    )

    reject_reason: Optional[Literal[
        "non_financial",  # Valid text, not a financial claim
        "question",       # Asking a question, not making a claim
        "incomplete"      # Missing ticker or value, unverifiable
    ]] = Field(
        None,
        description="Reason for rejection (only when claim_type == 'reject')"
    )

    @model_validator(mode='after')
    def validate_reject_reason(self):
        """Ensure reject_reason is set when claim_type is reject."""
        if self.claim_type == "reject" and self.reject_reason is None:
            raise ValueError("reject_reason must be set when claim_type is 'reject'")
        if self.claim_type != "reject" and self.reject_reason is not None:
            raise ValueError("reject_reason should only be set when claim_type is 'reject'")
        return self


class CanonicalPeriod(BaseModel):
    """Resolved time period with exact date ranges."""

    period_type: Literal["quarterly", "half_year", "annual", "date", "date_range", "current", "event_relative"] = Field(
        ...,
        description="Type of period"
    )

    start_date: str = Field(
        ...,
        description="First day of period (YYYY-MM-DD)"
    )

    end_date: str = Field(
        ...,
        description="Last day of period (YYYY-MM-DD)"
    )

    fiscal_year: Optional[int] = Field(
        None,
        description="Fiscal year number (e.g., 2024)"
    )

    fiscal_quarter: Optional[Literal["Q1", "Q2", "Q3", "Q4"]] = Field(
        None,
        description="Fiscal quarter"
    )

    is_assumption: bool = Field(
        False,
        description="Whether assumptions were made during resolution"
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="List of assumptions made"
    )

    original_mention: Optional[str] = Field(
        None,
        description="Original period text from claim"
    )


class CompanyInfo(BaseModel):
    """Company metadata retrieved during period resolution."""

    cik: str = Field(
        ...,
        description="SEC Central Index Key (10-digit, zero-padded)"
    )

    name: str = Field(
        ...,
        description="Official company name"
    )

    ticker: str = Field(
        ...,
        description="Primary ticker symbol"
    )

    fiscal_year_end: str = Field(
        ...,
        description="Fiscal year end (MMDD format, e.g., '0928' for September 28)"
    )

    sic: Optional[str] = Field(
        None,
        description="Standard Industrial Classification code"
    )

    sic_description: Optional[str] = Field(
        None,
        description="Industry description"
    )
