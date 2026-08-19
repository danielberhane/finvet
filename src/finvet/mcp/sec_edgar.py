"""SEC EDGAR MCP server adapter — typed methods backed by the real MCP server."""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..config.settings import settings
from ..utils.logging import get_logger
from .mcp_client import MCPClient, MCPError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (contract with tools/sec_tools.py — do NOT change fields)
# ---------------------------------------------------------------------------


class CompanyInfo(BaseModel):
    """Company information from SEC EDGAR."""
    cik: str = Field(..., description="SEC Central Index Key")
    name: str = Field(..., description="Official company name")
    ticker: Optional[str] = Field(None, description="Primary ticker symbol")
    fiscal_year_end: str = Field(..., description="Fiscal year end (MMDD format)")
    sic: Optional[str] = Field(None, description="Standard Industrial Classification code")
    sic_description: Optional[str] = Field(None, description="Industry description")


class Filing(BaseModel):
    """SEC filing information."""
    accession_number: str = Field(..., description="SEC accession number")
    filing_date: str = Field(..., description="Date the filing was submitted")
    period_of_report: str = Field(..., description="Period covered by the report")
    form_type: str = Field(..., description="Type of form (10-K, 10-Q, 8-K, etc.)")
    file_number: Optional[str] = Field(None, description="SEC file number")
    url: Optional[str] = Field(None, description="URL to the filing")


class FinancialItem(BaseModel):
    """Financial line item from SEC filing."""
    line_item: str = Field(..., description="Name of the line item")
    concept: Optional[str] = Field(None, description="XBRL concept name")
    value: float = Field(..., description="Numeric value")
    units: str = Field(default="USD", description="Currency and scale (e.g., 'USD')")
    period: str = Field(..., description="Period the value applies to")
    period_end: Optional[str] = Field(None, description="End date of the period (YYYY-MM-DD)")
    decimals: Optional[str] = Field(None, description="Precision indicator")
    context_ref: Optional[str] = Field(None, description="XBRL context reference")


# ---------------------------------------------------------------------------
# XBRL concept lists by statement type
# ---------------------------------------------------------------------------

CONCEPTS_BY_TYPE = {
    "income": [
        # Revenue — companies use different concepts depending on era and industry.
        # ASC 606 (post-2018): RevenueFromContractWithCustomer...
        # General: Revenues
        # Pre-2018: SalesRevenueNet, SalesRevenueGoodsNet
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "GrossProfit",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "ProfitLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
    ],
    "balance": [
        "Assets",
        "AssetsCurrent",
        "Liabilities",
        "LiabilitiesCurrent",
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "PropertyPlantAndEquipmentNet",
    ],
    "cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsOfDividends",
    ],
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SECEdgarClient:
    """Typed adapter for the SEC EDGAR MCP server (streamable-http)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.sec_edgar_mcp_url
        self._mcp = MCPClient(self.base_url)

    # -- company info --------------------------------------------------------

    def get_company_info(self, identifier: str) -> CompanyInfo:
        """Look up company metadata (CIK, fiscal year end, SIC, etc.)."""
        result = self._mcp.call_tool("get_company_info", {"identifier": identifier})

        if isinstance(result, dict):
            if not result.get("success", True):
                raise MCPError(f"SEC EDGAR error: {result.get('error', 'Unknown')}")
            if "company" in result:
                data = result["company"]
                if "cik" in data and isinstance(data["cik"], int):
                    data["cik"] = f"{data['cik']:010d}"
                return CompanyInfo(**data)
        return CompanyInfo(**result)

    # -- filings -------------------------------------------------------------

    def get_recent_filings(
        self,
        identifier: str,
        form_type: str,
        limit: int = 10,
    ) -> List[Filing]:
        """List recent SEC filings for a company."""
        result = self._mcp.call_tool(
            "get_recent_filings",
            {"identifier": identifier, "form_type": form_type, "limit": limit},
        )

        if isinstance(result, dict) and "filings" in result:
            return [Filing(**f) for f in result["filings"]]
        if isinstance(result, list):
            return [Filing(**f) for f in result]
        return []

    # -- financials via XBRL concepts ----------------------------------------

    def get_financials(
        self,
        identifier: str,
        accession_number: Optional[str] = None,
        statement_type: str = "income",
        period: str = "quarterly",
    ) -> List[FinancialItem]:
        """Extract financial data from a specific filing using XBRL concepts.

        Uses get_xbrl_concepts on the MCP server (not get_financials) because
        get_xbrl_concepts accepts an accession_number for period-specific data
        while the server's get_financials only returns the latest filing.
        """
        # Normalise cashflow → cashflow key
        key = "cashflow" if statement_type in ("cash", "cashflow") else statement_type
        concepts = CONCEPTS_BY_TYPE.get(key, CONCEPTS_BY_TYPE["income"])

        args = {"identifier": identifier, "concepts": concepts}
        if accession_number:
            args["accession_number"] = accession_number

        result = self._mcp.call_tool("get_xbrl_concepts", args)
        return self._parse_xbrl_result(result)

    @staticmethod
    def _parse_xbrl_result(result) -> List[FinancialItem]:
        """Convert get_xbrl_concepts response into FinancialItem list."""
        items: List[FinancialItem] = []

        if not isinstance(result, dict):
            return items

        concepts = result.get("concepts", {})
        filing_ref = result.get("filing_reference", {})
        filing_date = filing_ref.get("filing_date", "")

        # Server returns concepts as a dict keyed by concept name
        if isinstance(concepts, dict):
            entries = [
                (name, data) for name, data in concepts.items()
                if isinstance(data, dict)
            ]
        elif isinstance(concepts, list):
            entries = [(c.get("concept", ""), c) for c in concepts if isinstance(c, dict)]
        else:
            return items

        for concept_name, data in entries:
            value = data.get("value")
            if value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                logger.debug(f"Skipping non-numeric concept {concept_name}: value={value!r}")
                continue

            items.append(FinancialItem(
                line_item=concept_name,
                concept=f"us-gaap:{concept_name}" if concept_name else None,
                value=value,
                units=data.get("unit") or "USD",
                period=data.get("period", ""),
                period_end=data.get("period") or filing_date,
                decimals=data.get("decimals"),
                context_ref=data.get("context"),
            ))

        return items

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        self._mcp.close()
