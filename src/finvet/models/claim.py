"""Data models for claims and parsed claim information."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config.metrics import METRIC_WHITELIST

# The seven comparison operators of the claim-parser contract. approx and
# range both resolve to equality at a widened tolerance downstream (range
# carries the band's midpoint in `value`; the band itself is not in the schema).
Operator = Literal["eq", "gt", "gte", "lt", "lte", "approx", "range"]


class ParsedClaim(BaseModel):
    """A parsed claim, conforming to the fine-tuned claim parser's contract:

        claim_type | ticker | metric | operator | value | period | reject_reason

    CONTRACT phase completed 2026-08-20: `comparison` and `currency` are gone
    and unknown fields are forbidden — a stray legacy kwarg raises instead of
    being silently ignored. The boundary (normalize_parser_output) strips
    legacy keys from raw model output before construction; the model itself
    stays strict.
    """

    model_config = ConfigDict(extra="forbid")

    claim_type: Literal["sec", "market", "news", "reject"] = Field(
        ...,
        description="Claim category for routing"
    )

    ticker: Optional[str] = Field(
        None,
        description="Stock ticker symbol (e.g., 'AAPL', 'TSLA')"
    )

    metric: Optional[str] = Field(
        None,
        description="Canonical snake_case metric from METRIC_WHITELIST[claim_type]; "
                    "null when no whitelisted metric fits (the agent then infers "
                    "from claim text, as before the field existed)"
    )

    operator: Optional[Operator] = Field(
        None,
        description="How the claimed value relates to the actual value; "
                    "null when the claim carries no numeric value"
    )

    value: Optional[float] = Field(
        None,
        description="Numeric value being claimed (e.g., 94000000000 for $94B); "
                    "for range claims, the band's midpoint"
    )

    period: Optional[str] = Field(
        None,
        description="Time period as mentioned (e.g., 'Q4 FY2024', 'fiscal 2023')"
    )

    reject_reason: Optional[str] = Field(
        None,
        description="snake_case reason, only when claim_type == 'reject'. Open "
                    "vocabulary: the gold uses 21+ distinct values and the tail "
                    "is genuinely open, so a Literal would be brittle"
    )

    @model_validator(mode='after')
    def validate_contract(self):
        """The contract's cross-field rules, in one validator so their order
        is explicit.

        Enforced here: the reject/reason pairing, metric scoping, the §8
        reject contract (a reject carries nothing but its reason), and
        operator-iff-value pairing.
        """
        if self.claim_type == "reject" and self.reject_reason is None:
            raise ValueError("reject_reason must be set when claim_type is 'reject'")
        if self.claim_type != "reject" and self.reject_reason is not None:
            raise ValueError("reject_reason should only be set when claim_type is 'reject'")

        if self.metric is not None:
            allowed = METRIC_WHITELIST.get(self.claim_type, frozenset())
            if self.metric not in allowed:
                raise ValueError(
                    f"metric {self.metric!r} is not whitelisted for "
                    f"claim_type {self.claim_type!r}"
                )

        if self.claim_type == "reject":
            stray = [f for f in ("ticker", "metric", "operator",
                                 "value", "period")
                     if getattr(self, f) is not None]
            if stray:
                raise ValueError(
                    f"a reject carries nothing but its reason; stray fields: {stray}"
                )
        else:
            if (self.value is None) != (self.operator is None):
                raise ValueError(
                    "operator and value come as a pair: a value needs a "
                    "comparator and a comparator needs a value "
                    f"(value={self.value!r}, operator={self.operator!r})"
                )

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
