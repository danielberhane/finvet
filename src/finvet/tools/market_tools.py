"""Market data tools for the Market verification agent.

These tools provide access to market data through the Finnhub MCP server.
The agent uses these tools to verify claims about stock prices, market cap,
P/E ratios, and other market-based metrics.

Tool Selection Guide for the LLM:
---------------------------------
1. get_stock_quote: For current/recent price, today's trading data
2. get_daily_prices: For historical prices on specific dates or date ranges
3. get_company_overview: For market cap, P/E ratio, dividend yield, EPS
4. get_earnings: For quarterly EPS and earnings estimates (analyst consensus)

Data Source: Finnhub (60 requests/minute free tier)
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..mcp.finnhub import FinnhubClient, DailyPrice, Quote, CompanyOverview
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Singleton client instance (lazy initialization)
_market_client: Optional[FinnhubClient] = None


def _get_client() -> FinnhubClient:
    """Get or create the Finnhub client singleton."""
    global _market_client
    if _market_client is None:
        _market_client = FinnhubClient()
    return _market_client


def _set_client(client: FinnhubClient) -> None:
    """Inject a client instance (for testing)."""
    global _market_client
    _market_client = client


class QuoteResult(BaseModel):
    """Result from get_stock_quote tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    price: Optional[float] = Field(None, description="Current/last price")
    change: Optional[float] = Field(None, description="Price change from previous close")
    change_percent: Optional[str] = Field(None, description="Percentage change")
    volume: Optional[int] = Field(None, description="Trading volume")
    latest_trading_day: Optional[str] = Field(None, description="Last trading day")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Day's high")
    low: Optional[float] = Field(None, description="Day's low")
    error: Optional[str] = Field(None, description="Error message if failed")


class DailyPricesResult(BaseModel):
    """Result from get_daily_prices tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    last_refreshed: Optional[str] = Field(None, description="When data was last updated")
    prices: List[Dict[str, Any]] = Field(default_factory=list, description="Daily price data")
    count: int = Field(0, description="Number of data points")
    error: Optional[str] = Field(None, description="Error message if failed")


class CompanyOverviewResult(BaseModel):
    """Result from get_company_overview tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    sector: Optional[str] = Field(None, description="Industry sector")
    market_cap: Optional[float] = Field(None, description="Market capitalization in USD")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    dividend_yield: Optional[float] = Field(None, description="Annual dividend yield")
    eps: Optional[float] = Field(None, description="Earnings per share (TTM)")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high price")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low price")
    error: Optional[str] = Field(None, description="Error message if failed")


class EarningsResult(BaseModel):
    """Result from get_earnings tool."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    quarterly_earnings: List[Dict[str, Any]] = Field(default_factory=list)
    annual_earnings: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = Field(None, description="Error message if failed")


@tool
def get_stock_quote(ticker: str) -> QuoteResult:
    """
    Get the current or most recent stock quote for a ticker.

    Use this for claims about:
    - Current stock price ("AAPL is trading at $180")
    - Today's price movement ("Tesla dropped 5% today")
    - Recent price levels ("Microsoft closed above $400")

    Note: Stock prices are delayed (typically 15-20 minutes).
    For claims about specific historical dates, use get_daily_prices instead.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT")

    Returns:
        QuoteResult with current price, change, volume, and trading day info
    """
    try:
        client = _get_client()
        quote = client.get_quote(symbol=ticker)
        return QuoteResult(success=True, **quote.model_dump())
    except Exception as e:
        logger.error(f"get_stock_quote failed for {ticker}: {e}")
        return QuoteResult(success=False, error=str(e))


@tool
def get_daily_prices(
    ticker: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    full_history: bool = False,
) -> DailyPricesResult:
    """
    Get historical daily price data (OHLCV) for a stock.

    Use this for claims about:
    - Historical prices on specific dates ("AAPL was $150 on Jan 1, 2024")
    - Price ranges over periods ("Tesla's 52-week high was $300")
    - Price comparisons ("Apple doubled since 2020")
    - All-time highs/lows

    The data includes Open, High, Low, Close prices and Volume for each day.

    IMPORTANT: For claims about specific years or date ranges, ALWAYS pass
    from_date and to_date to avoid retrieving too much data. For example,
    for a claim about 2010 prices, use from_date="2010-01-01" to_date="2010-12-31".

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        from_date: Start date in YYYY-MM-DD format (e.g., "2010-01-01").
                   If not set, defaults to 100 trading days ago.
        to_date: End date in YYYY-MM-DD format (e.g., "2010-12-31").
                 If not set, defaults to today.
        full_history: If True and no from_date is set, returns all available
                     history. Use from_date/to_date instead when possible.

    Returns:
        DailyPricesResult with list of daily OHLCV data sorted by date
    """
    try:
        client = _get_client()
        outputsize = "full" if (full_history or from_date) else "compact"
        prices = client.get_daily_prices(
            symbol=ticker,
            from_date=from_date,
            to_date=to_date,
            outputsize=outputsize,
        )
        return DailyPricesResult(
            success=True,
            symbol=prices.symbol,
            last_refreshed=prices.last_refreshed,
            prices=[
                {
                    "date": p.date,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                }
                for p in prices.prices
            ],
            count=len(prices.prices),
        )
    except Exception as e:
        logger.error(f"get_daily_prices failed for {ticker}: {e}")
        return DailyPricesResult(success=False, error=str(e))


@tool
def get_company_overview(ticker: str) -> CompanyOverviewResult:
    """
    Get company fundamentals and valuation metrics from Finnhub.

    Use this for claims about:
    - Market capitalization ("Apple's market cap is $3 trillion")
    - P/E ratio ("Tesla's P/E ratio is 50")
    - Dividend yield ("Microsoft's dividend yield is 0.8%")
    - Trailing twelve month EPS
    - 52-week high/low prices
    - Company sector/industry

    Note: This provides MARKET-DERIVED metrics (based on stock price and
    analyst calculations). For GAAP financial statement data, use SEC tools.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

    Returns:
        CompanyOverviewResult with market cap, P/E, dividend yield, etc.
    """
    try:
        client = _get_client()
        overview = client.get_company_overview(symbol=ticker)
        # Exclude 'description' — CompanyOverview has it but CompanyOverviewResult doesn't
        data = overview.model_dump(exclude={"description"})
        return CompanyOverviewResult(success=True, **data)
    except Exception as e:
        logger.error(f"get_company_overview failed for {ticker}: {e}")
        return CompanyOverviewResult(success=False, error=str(e))


@tool
def get_earnings(ticker: str) -> EarningsResult:
    """
    Get historical earnings data and estimates for a company.

    Use this for claims about:
    - Quarterly EPS ("Apple's Q4 EPS was $1.46")
    - Earnings beats/misses ("Tesla beat earnings estimates")
    - Annual earnings trends
    - Analyst earnings estimates

    This data comes from Finnhub and includes:
    - Reported EPS (actual)
    - Estimated EPS (analyst consensus before earnings)
    - Surprise percentage (how much actual beat/missed estimate)

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

    Returns:
        EarningsResult with quarterly earnings data
    """
    try:
        client = _get_client()
        result = client.get_earnings(symbol=ticker)

        quarterly = result.get("quarterly_earnings", [])

        return EarningsResult(
            success=True,
            symbol=ticker,
            quarterly_earnings=quarterly,
            annual_earnings=[],  # Finnhub doesn't provide annual aggregates
        )
    except Exception as e:
        logger.error(f"get_earnings failed for {ticker}: {e}")
        return EarningsResult(success=False, error=str(e))


# Export all tools for the agent
MARKET_TOOLS = [
    get_stock_quote,
    get_daily_prices,
    get_company_overview,
    get_earnings,
]
