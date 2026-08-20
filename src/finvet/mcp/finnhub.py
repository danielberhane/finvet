"""Finnhub client for market data - direct API integration.

This client calls Finnhub's REST API directly (no MCP overhead).
Provides typed methods for market data:
- get_quote: Current stock quote
- get_daily_prices: Historical OHLCV data (requires a paid Finnhub tier)
- get_company_overview: Market cap, P/E, EPS, 52-week range
- get_earnings: Quarterly EPS actual vs estimates

Rate limit: 60 requests/minute (free tier)
API Docs: https://finnhub.io/docs/api

Note: Historical price data (get_daily_prices) requires a paid Finnhub tier.
On the free tier the endpoint returns 403; that failure is raised rather than
substituted from another source.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubAPIError(Exception):
    """Exception raised for Finnhub API errors."""
    pass


class DailyPrice(BaseModel):
    """Daily OHLCV price data."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")


class DailyPrices(BaseModel):
    """Collection of daily price data."""
    symbol: str = Field(..., description="Stock ticker symbol")
    last_refreshed: str = Field(..., description="Last refresh timestamp")
    prices: List[DailyPrice] = Field(..., description="List of daily prices")


class Quote(BaseModel):
    """Current stock quote."""
    symbol: str = Field(..., description="Stock ticker symbol")
    price: float = Field(..., description="Current price")
    change: Optional[float] = Field(None, description="Price change")
    change_percent: Optional[str] = Field(None, description="Price change percentage")
    volume: Optional[int] = Field(None, description="Trading volume")
    latest_trading_day: Optional[str] = Field(None, description="Latest trading day")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Day's high price")
    low: Optional[float] = Field(None, description="Day's low price")


class CompanyOverview(BaseModel):
    """Company fundamentals and overview data."""
    symbol: str = Field(..., description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    description: Optional[str] = Field(None, description="Business description")
    sector: Optional[str] = Field(None, description="Industry sector")
    market_cap: Optional[float] = Field(None, description="Market capitalization in USD")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    eps: Optional[float] = Field(None, description="Earnings per share (TTM)")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high price")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low price")


# Mock data for testing
MOCK_STOCK_DATA = {
    "AAPL": {"price": 178.50, "name": "Apple Inc.", "sector": "Technology", "market_cap": 2800000000000, "pe_ratio": 28.5, "eps": 6.14, "52_week_high": 199.62, "52_week_low": 164.08, "dividend_yield": 0.51},
    "MSFT": {"price": 413.60, "name": "Microsoft Corporation", "sector": "Technology", "market_cap": 3070000000000, "pe_ratio": 35.91, "eps": 11.52, "52_week_high": 468.35, "52_week_low": 309.45, "dividend_yield": 0.72},
    "GOOGL": {"price": 175.43, "name": "Alphabet Inc.", "sector": "Technology", "market_cap": 2180000000000, "pe_ratio": 24.1, "eps": 7.28, "52_week_high": 191.75, "52_week_low": 130.67, "dividend_yield": 0.0},
    "TSLA": {"price": 248.50, "name": "Tesla Inc.", "sector": "Consumer Cyclical", "market_cap": 790000000000, "pe_ratio": 48.7, "eps": 5.10, "52_week_high": 278.98, "52_week_low": 138.80, "dividend_yield": 0.0},
    "NVDA": {"price": 878.35, "name": "NVIDIA Corporation", "sector": "Technology", "market_cap": 2160000000000, "pe_ratio": 65.2, "eps": 13.47, "52_week_high": 974.00, "52_week_low": 473.20, "dividend_yield": 0.02},
    "META": {"price": 612.77, "name": "Meta Platforms Inc.", "sector": "Technology", "market_cap": 1550000000000, "pe_ratio": 27.8, "eps": 22.04, "52_week_high": 638.40, "52_week_low": 414.50, "dividend_yield": 0.35},
    "AMZN": {"price": 225.94, "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "market_cap": 2350000000000, "pe_ratio": 42.3, "eps": 5.34, "52_week_high": 242.52, "52_week_low": 151.61, "dividend_yield": 0.0},
}


def _safe_float(value, default=None) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None or value == "None" or value == "-" or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class FinnhubClient:
    """
    Direct Finnhub API client for market data.

    Calls Finnhub REST API directly - no MCP overhead.
    Rate limit: 60 requests/minute on free tier.
    """

    def __init__(self, api_key: Optional[str] = None, mock_mode: Optional[bool] = None):
        """
        Initialize Finnhub client.

        Args:
            api_key: Finnhub API key. If not provided, uses settings.
            mock_mode: If True, return mock data instead of calling API.
        """
        self.mock_mode = mock_mode if mock_mode is not None else settings.finnhub_mock_mode
        self.api_key = api_key or settings.finnhub_api_key

        if not self.mock_mode and not self.api_key:
            raise FinnhubAPIError(
                "FINNHUB_API_KEY required. Get free key at https://finnhub.io/register"
            )

        self._client = httpx.Client(timeout=30.0)

        if self.mock_mode:
            logger.info("Finnhub client running in MOCK MODE")
        else:
            logger.info("Finnhub client initialized (direct API)")

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make request to Finnhub API."""
        params = params or {}
        params["token"] = self.api_key
        url = f"{FINNHUB_BASE_URL}/{endpoint}"

        try:
            response = self._client.get(url, params=params)

            if response.status_code == 429:
                raise FinnhubAPIError("Rate limit exceeded (60/min). Wait and retry.")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise FinnhubAPIError(f"Finnhub API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise FinnhubAPIError(f"Request failed: {str(e)}")

    def _get_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Get mock data for a symbol."""
        symbol = symbol.upper()
        if symbol in MOCK_STOCK_DATA:
            return MOCK_STOCK_DATA[symbol]
        return {"price": 100.0, "name": f"{symbol} Inc.", "sector": "Unknown", "market_cap": 50000000000, "pe_ratio": 20.0, "eps": 5.0, "52_week_high": 120.0, "52_week_low": 80.0, "dividend_yield": 0.0}

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current stock quote.

        Finnhub endpoint: GET /quote?symbol=AAPL
        """
        symbol = symbol.upper()

        if self.mock_mode:
            mock = self._get_mock_data(symbol)
            today = datetime.now().strftime("%Y-%m-%d")
            change = round(mock["price"] * 0.012, 2)
            return Quote(
                symbol=symbol, price=mock["price"], change=change,
                change_percent=f"{(change / mock['price']) * 100:.2f}%",
                volume=15000000, latest_trading_day=today,
                previous_close=round(mock["price"] - change, 2),
                open=round(mock["price"] * 0.998, 2),
                high=round(mock["price"] * 1.01, 2),
                low=round(mock["price"] * 0.99, 2),
            )

        data = self._request("quote", {"symbol": symbol})

        if not data or data.get("c") == 0:
            raise FinnhubAPIError(f"No quote data for {symbol}")

        current = data.get("c", 0)
        prev_close = data.get("pc", 0)
        change = round(current - prev_close, 2) if prev_close else 0
        change_pct = round((change / prev_close * 100), 2) if prev_close else 0

        return Quote(
            symbol=symbol,
            price=current,
            change=change,
            change_percent=f"{change_pct:.2f}%",
            volume=None,
            latest_trading_day=datetime.fromtimestamp(data.get("t", time.time())).strftime("%Y-%m-%d"),
            previous_close=prev_close,
            open=data.get("o"),
            high=data.get("h"),
            low=data.get("l"),
        )

    def get_daily_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        outputsize: str = "compact",
    ) -> DailyPrices:
        """
        Get historical daily OHLCV data.

        Finnhub endpoint GET /stock/candle.

        Requires a paid Finnhub tier. The free tier returns HTTP 403, which is
        raised as FinnhubAPIError; there is no alternative historical source.
        """
        symbol = symbol.upper()

        if self.mock_mode:
            mock = self._get_mock_data(symbol)
            today = datetime.now()
            prices = []
            days = 100 if outputsize == "compact" else 365
            for i in range(days):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                factor = 1 + (0.002 * (i % 7 - 3))
                day_price = mock["price"] * factor * (1 - i * 0.0005)
                prices.append(DailyPrice(
                    date=date,
                    open=round(day_price * 0.998, 2),
                    high=round(day_price * 1.01, 2),
                    low=round(day_price * 0.99, 2),
                    close=round(day_price, 2),
                    volume=12000000 + (i * 100000 % 5000000),
                ))
            return DailyPrices(symbol=symbol, last_refreshed=today.strftime("%Y-%m-%d"), prices=prices)

        # Calculate date range
        if to_date:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        else:
            to_dt = datetime.now()

        if from_date:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        else:
            days_back = 100 if outputsize == "compact" else 730
            from_dt = to_dt - timedelta(days=days_back)

        # Finnhub is the only historical price source.
        try:
            data = self._request("stock/candle", {
                "symbol": symbol,
                "resolution": "D",
                "from": int(from_dt.timestamp()),
                "to": int(to_dt.timestamp()),
            })

            if data and data.get("s") == "ok":
                return self._parse_finnhub_candles(symbol, data)

        except FinnhubAPIError as e:
            if "403" in str(e):
                raise FinnhubAPIError(
                    f"Historical candles for {symbol} require a paid Finnhub tier "
                    f"(HTTP 403). No alternative historical price source is configured."
                ) from e
            raise

        raise FinnhubAPIError(
            f"Finnhub returned no historical candle data for {symbol}. "
            f"No alternative historical price source is configured."
        )

    def _parse_finnhub_candles(self, symbol: str, data: Dict[str, Any]) -> DailyPrices:
        """Parse Finnhub candle response into DailyPrices."""
        prices = []
        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        for i in range(len(timestamps)):
            prices.append(DailyPrice(
                date=datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d"),
                open=opens[i] if i < len(opens) else 0,
                high=highs[i] if i < len(highs) else 0,
                low=lows[i] if i < len(lows) else 0,
                close=closes[i] if i < len(closes) else 0,
                volume=int(volumes[i]) if i < len(volumes) else 0,
            ))

        prices.sort(key=lambda p: p.date, reverse=True)

        return DailyPrices(
            symbol=symbol,
            last_refreshed=datetime.now().strftime("%Y-%m-%d"),
            prices=prices,
        )

    def get_company_overview(self, symbol: str) -> CompanyOverview:
        """
        Get company fundamentals and valuation metrics.

        Finnhub endpoints:
        - GET /stock/metric?symbol=AAPL&metric=all (financials)
        - GET /stock/profile2?symbol=AAPL (company info)
        """
        symbol = symbol.upper()

        if self.mock_mode:
            mock = self._get_mock_data(symbol)
            return CompanyOverview(
                symbol=symbol, name=mock["name"],
                description=f"{mock['name']} is a leading company in the {mock['sector']} sector.",
                sector=mock["sector"], market_cap=mock["market_cap"],
                pe_ratio=mock["pe_ratio"], dividend_yield=mock["dividend_yield"],
                eps=mock["eps"], fifty_two_week_high=mock["52_week_high"],
                fifty_two_week_low=mock["52_week_low"],
            )

        # Get metrics
        metrics_data = self._request("stock/metric", {"symbol": symbol, "metric": "all"})
        metrics = metrics_data.get("metric", {}) if metrics_data else {}

        # Get profile
        try:
            profile = self._request("stock/profile2", {"symbol": symbol})
        except Exception as e:
            logger.warning(f"Could not fetch profile for {symbol}: {e}")
            profile = {}

        # Finnhub returns market cap in millions
        market_cap_millions = _safe_float(metrics.get("marketCapitalization"), 0)
        market_cap = market_cap_millions * 1_000_000 if market_cap_millions else None

        return CompanyOverview(
            symbol=symbol,
            name=profile.get("name"),
            description=profile.get("weburl"),
            sector=profile.get("finnhubIndustry"),
            market_cap=market_cap,
            pe_ratio=_safe_float(metrics.get("peBasicExclExtraTTM")),
            dividend_yield=_safe_float(metrics.get("dividendYieldIndicatedAnnual")),
            eps=_safe_float(metrics.get("epsBasicExclExtraItemsTTM")),
            fifty_two_week_high=_safe_float(metrics.get("52WeekHigh")),
            fifty_two_week_low=_safe_float(metrics.get("52WeekLow")),
        )

    def get_earnings(self, symbol: str, limit: int = 8) -> Dict[str, Any]:
        """
        Get company earnings history.

        Finnhub endpoint: GET /stock/earnings?symbol=AAPL
        """
        symbol = symbol.upper()

        if self.mock_mode:
            return {
                "symbol": symbol,
                "quarterly_earnings": [
                    {"period": "2024-09-30", "actual_eps": 1.64, "estimated_eps": 1.60, "surprise": 0.04, "surprise_percent": 2.5},
                    {"period": "2024-06-30", "actual_eps": 1.40, "estimated_eps": 1.35, "surprise": 0.05, "surprise_percent": 3.7},
                ][:limit],
            }

        data = self._request("stock/earnings", {"symbol": symbol})

        if not data:
            return {"symbol": symbol, "quarterly_earnings": []}

        quarterly = []
        for item in data[:limit]:
            quarterly.append({
                "period": item.get("period", ""),
                "actual_eps": item.get("actual"),
                "estimated_eps": item.get("estimate"),
                "surprise": item.get("surprise"),
                "surprise_percent": item.get("surprisePercent"),
            })

        return {"symbol": symbol, "quarterly_earnings": quarterly}

    def get_company_name(self, ticker: str) -> Optional[str]:
        """Return the company name for a ticker, or None if not found."""
        ticker = ticker.upper()
        if self.mock_mode:
            return self._get_mock_data(ticker).get("name")
        try:
            data = self._request("stock/profile2", {"symbol": ticker})
            return data.get("name") or None
        except Exception:
            return None

    def search_ticker(self, query: str) -> Optional[str]:
        """Return the best-matching US common-stock ticker for a company name query."""
        if self.mock_mode:
            return None
        try:
            data = self._request("search", {"q": query, "exchange": "US"})
            for result in data.get("result", []):
                if result.get("type") in ("Common Stock", "EQS"):
                    return result.get("symbol")
            results = data.get("result", [])
            return results[0].get("symbol") if results else None
        except Exception:
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
