"""SEC EDGAR tools for the SEC verification agent.

These tools provide access to SEC filings and financial data through the
SEC EDGAR MCP server. The agent uses these tools to verify claims about
company financials, revenue, earnings, and other GAAP accounting metrics.

Tool Selection Guide for the LLM:
---------------------------------
1. get_company_info: Always call first to resolve ticker to CIK and get fiscal year end
2. get_recent_filings: Find the specific filing for the claimed period
3. get_income_statement: For revenue, net income, earnings, EPS, operating income
4. get_balance_sheet: For assets, liabilities, equity, debt, cash positions
5. get_cash_flow: For operating cash flow, free cash flow, capex, dividends
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..mcp.sec_edgar import SECEdgarClient
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Singleton client instance (lazy initialization)
_sec_client: Optional[SECEdgarClient] = None


def _get_client() -> SECEdgarClient:
    """Get or create the SEC EDGAR client singleton."""
    global _sec_client
    if _sec_client is None:
        _sec_client = SECEdgarClient()
    return _sec_client


# The period the claim actually refers to, resolved upstream by period_resolver.
# Carried out-of-band rather than as a tool argument: it is already known
# deterministically, so routing it through the model would only add a channel
# for it to be transcribed wrongly. A ContextVar keeps it per-request, so
# concurrent verifications cannot see each other's period.
_period_target: ContextVar[Tuple[Optional[str], Optional[str]]] = ContextVar(
    "finvet_sec_period_target", default=(None, None)
)

# Period types that name a real reporting date. "current" and "event_relative"
# carry today's date as a placeholder — targeting XBRL with it would match
# nothing and flag every value unverified.
_DATABLE_PERIOD_TYPES = frozenset({"annual", "quarterly", "half_year", "date"})


def period_target_for(canonical_period: Any) -> Optional[Tuple[str, str]]:
    """(end date, period type) for a period that names a real reporting date."""
    if canonical_period is None:
        return None
    period_type = getattr(canonical_period, "period_type", None)
    end_date = getattr(canonical_period, "end_date", None)
    if period_type not in _DATABLE_PERIOD_TYPES or not end_date:
        return None
    return end_date, period_type


@contextmanager
def use_period_target(period_end: Optional[str], period_type: Optional[str]):
    """Apply a resolved period to every SEC tool call made inside the block."""
    token = _period_target.set((period_end, period_type))
    try:
        yield
    finally:
        _period_target.reset(token)


def _current_period_target() -> Tuple[Optional[str], Optional[str]]:
    return _period_target.get()


def _set_client(client: SECEdgarClient) -> None:
    """Inject a client instance (for testing)."""
    global _sec_client
    _sec_client = client


def _format_financial_items(financials: list) -> list[dict]:
    """Convert a list of FinancialItem models to serializable dicts."""
    return [
        {
            "line_item": f.line_item,
            "concept": f.concept,
            "value": f.value,
            "units": f.units,
            "period": f.period,
            "period_end": f.period_end,
            # Fact-level quality, carried to the trust boundary. A segment or
            # subsidiary figure is not the entity-wide number a claim asks
            # about, and dropping this made the two indistinguishable.
            "consolidated": f.consolidated,
        }
        for f in financials
    ]


class CompanyInfoResult(BaseModel):
    """Result from get_company_info tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    cik: Optional[str] = Field(None, description="10-digit SEC CIK")
    name: Optional[str] = Field(None, description="Official company name")
    ticker: Optional[str] = Field(None, description="Primary ticker symbol")
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end (MMDD format)")
    sic: Optional[str] = Field(None, description="Industry classification code")
    sic_description: Optional[str] = Field(None, description="Industry description")
    error: Optional[str] = Field(None, description="Error message if failed")


class FilingsResult(BaseModel):
    """Result from get_recent_filings tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    filings: List[Dict[str, Any]] = Field(default_factory=list, description="List of filings")
    count: int = Field(0, description="Number of filings found")
    error: Optional[str] = Field(None, description="Error message if failed")


class FinancialsResult(BaseModel):
    """Result from financial statement tools."""
    success: bool = Field(..., description="Whether the extraction succeeded")
    statement_type: str = Field(..., description="Type of statement retrieved")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Financial line items")
    filing_accession: Optional[str] = Field(None, description="Filing accession number used")
    period_end: Optional[str] = Field(None, description="Period end date of the data")
    error: Optional[str] = Field(None, description="Error message if failed")


@tool
def get_company_info(ticker_or_name: str) -> Dict[str, Any]:
    """
    Look up company information from SEC EDGAR by ticker symbol or company name.

    ALWAYS call this tool first before any other SEC tool. This provides:
    - The CIK (Central Index Key) needed for other SEC lookups
    - Fiscal year end date (critical for matching quarterly/annual periods)
    - Industry classification

    Examples of when to use:
    - "Apple's Q4 2024 revenue" -> get_company_info("AAPL") first
    - "Tesla earnings" -> get_company_info("TSLA") first
    - "Microsoft net income" -> get_company_info("MSFT") first

    Args:
        ticker_or_name: Stock ticker (e.g., "AAPL") or company name (e.g., "Apple Inc")

    Returns:
        CompanyInfoResult with CIK, fiscal year end, and company metadata
    """
    try:
        client = _get_client()
        info = client.get_company_info(identifier=ticker_or_name)
        return CompanyInfoResult(success=True, **info.model_dump()).model_dump()
    except Exception as e:
        logger.error(f"get_company_info failed for {ticker_or_name}: {e}")
        return CompanyInfoResult(success=False, error=str(e)).model_dump()


@tool
def get_recent_filings(
    cik: str,
    form_type: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    List recent SEC filings for a company to find the one covering the claimed period.

    Use this after get_company_info to find the specific filing that contains
    the financial data for the period mentioned in the claim.

    Form types:
    - "10-K": Annual report (use for fiscal year claims like "FY2024 revenue")
    - "10-Q": Quarterly report (use for Q1, Q2, Q3 claims)
    - "10-K/A": Amended annual (check if original 10-K was restated)
    - "8-K": Current report (for earnings announcements, material events)

    Note: Q4 is not filed separately, and this release does not support
    deriving it. Q4 numeric claims are declined before an agent runs; do not
    attempt to assemble one from an annual and a nine-month figure.

    Args:
        cik: SEC Central Index Key (10-digit, from get_company_info)
        form_type: SEC form type ("10-K", "10-Q", "8-K", "10-K/A")
        limit: Maximum filings to return (default 10)

    Returns:
        FilingsResult with list of filings including accession numbers and periods
    """
    try:
        client = _get_client()
        filings = client.get_recent_filings(
            identifier=cik,
            form_type=form_type,
            limit=limit,
        )
        return FilingsResult(
            success=True,
            filings=[
                {
                    "accession_number": f.accession_number,
                    "filing_date": f.filing_date,
                    "period_of_report": f.period_of_report,
                    "form_type": f.form_type,
                    "url": f.url,
                }
                for f in filings
            ],
            count=len(filings),
        ).model_dump()
    except Exception as e:
        logger.error(f"get_recent_filings failed for CIK {cik}: {e}")
        return FilingsResult(success=False, error=str(e)).model_dump()


@tool
def get_income_statement(
    cik: str,
    accession_number: str,
    period: str = "quarterly",
) -> Dict[str, Any]:
    """
    Extract income statement data from an SEC filing for revenue and earnings claims.

    Use this for claims about:
    - Revenue / Net Sales / Total Revenue
    - Net Income / Net Earnings / Profit
    - Earnings Per Share (EPS)
    - Operating Income / Operating Profit
    - Gross Profit / Gross Margin
    - Cost of Revenue / Cost of Goods Sold

    The income statement shows a company's financial performance over a period.
    For quarterly filings (10-Q), data covers 3 months.
    For annual filings (10-K), data covers 12 months.

    Args:
        cik: SEC Central Index Key
        accession_number: Filing accession number (from get_recent_filings)
        period: "quarterly" for 10-Q or "annual" for 10-K

    Returns:
        FinancialsResult with line items including Revenue, Net Income, EPS, etc.
    """
    try:
        client = _get_client()
        period_end, resolved_period = _current_period_target()
        financials = client.get_financials(
            identifier=cik,
            accession_number=accession_number,
            statement_type="income",
            period=resolved_period or period,
            period_end=period_end,
        )
        if financials:
            logger.info(f"Income statement: {len(financials)} items, periods: {set(f.period for f in financials)}")
        return FinancialsResult(
            success=True,
            statement_type="income",
            items=_format_financial_items(financials),
            filing_accession=accession_number,
            period_end=financials[0].period_end if financials else None,
        ).model_dump()
    except Exception as e:
        logger.error(f"get_income_statement failed: {e}")
        return FinancialsResult(
            success=False,
            statement_type="income",
            error=str(e),
        ).model_dump()


@tool
def get_balance_sheet(
    cik: str,
    accession_number: str,
) -> Dict[str, Any]:
    """
    Extract balance sheet data from an SEC filing for asset and liability claims.

    Use this for claims about:
    - Total Assets
    - Total Liabilities
    - Shareholders' Equity / Stockholders' Equity
    - Cash and Cash Equivalents
    - Total Debt / Long-term Debt
    - Accounts Receivable / Accounts Payable
    - Inventory
    - Property, Plant and Equipment (PP&E)

    The balance sheet shows a company's financial position at a point in time.
    It follows the equation: Assets = Liabilities + Shareholders' Equity

    Args:
        cik: SEC Central Index Key
        accession_number: Filing accession number (from get_recent_filings)

    Returns:
        FinancialsResult with line items for assets, liabilities, and equity
    """
    try:
        client = _get_client()
        period_end, resolved_period = _current_period_target()
        financials = client.get_financials(
            identifier=cik,
            accession_number=accession_number,
            statement_type="balance",
            period=resolved_period or "quarterly",
            period_end=period_end,
        )
        if financials:
            logger.info(f"Balance sheet: {len(financials)} items, periods: {set(f.period for f in financials)}")
        return FinancialsResult(
            success=True,
            statement_type="balance",
            items=_format_financial_items(financials),
            filing_accession=accession_number,
            period_end=financials[0].period_end if financials else None,
        ).model_dump()
    except Exception as e:
        logger.error(f"get_balance_sheet failed: {e}")
        return FinancialsResult(
            success=False,
            statement_type="balance",
            error=str(e),
        ).model_dump()


@tool
def get_cash_flow(
    cik: str,
    accession_number: str,
    period: str = "quarterly",
) -> Dict[str, Any]:
    """
    Extract cash flow statement data from an SEC filing for cash-related claims.

    Use this for claims about:
    - Operating Cash Flow / Cash from Operations
    - Free Cash Flow (Operating Cash Flow - CapEx)
    - Capital Expenditures (CapEx)
    - Dividends Paid
    - Share Repurchases / Buybacks
    - Cash from Investing Activities
    - Cash from Financing Activities

    The cash flow statement shows how cash moves in and out of the business.
    It reconciles net income to actual cash generated/used.

    Args:
        cik: SEC Central Index Key
        accession_number: Filing accession number (from get_recent_filings)
        period: "quarterly" for 10-Q or "annual" for 10-K

    Returns:
        FinancialsResult with cash flow line items
    """
    try:
        client = _get_client()
        period_end, resolved_period = _current_period_target()
        financials = client.get_financials(
            identifier=cik,
            accession_number=accession_number,
            statement_type="cashflow",
            period=resolved_period or period,
            period_end=period_end,
        )
        if financials:
            logger.info(f"Cash flow: {len(financials)} items, periods: {set(f.period for f in financials)}")
        return FinancialsResult(
            success=True,
            statement_type="cashflow",
            items=_format_financial_items(financials),
            filing_accession=accession_number,
            period_end=financials[0].period_end if financials else None,
        ).model_dump()
    except Exception as e:
        logger.error(f"get_cash_flow failed: {e}")
        return FinancialsResult(
            success=False,
            statement_type="cashflow",
            error=str(e),
        ).model_dump()


# Export all tools for the agent
SEC_TOOLS = [
    get_company_info,
    get_recent_filings,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
]
