"""Tool definitions for domain agents.

This module provides LangChain-compatible tools that wrap MCP servers
and other data sources. Each tool has rich descriptions that enable
the LLM to intelligently select the right tool for verification.

Tool Categories:
- SEC Tools: For GAAP financial statement data from SEC EDGAR filings
- Market Tools: For market data (prices, valuations) from Finnhub
- News Tools: For news search and source verification via Tavily
"""

from .sec_tools import (
    SEC_TOOLS,
    get_company_info,
    get_recent_filings,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
)

from .market_tools import (
    MARKET_TOOLS,
    get_stock_quote,
    get_daily_prices,
    get_company_overview,
    get_earnings,
)

from .news_tools import (
    NEWS_TOOLS,
    search_financial_news,
    verify_news_source,
)

from .tavily_search import TavilyClient

from .filing_search import search_filing_text
from .corroborate_sec import corroborate_with_filing
from .memory_tools import search_past_verifications

# RAG tool for the SEC agent
RAG_TOOLS = [search_filing_text]

__all__ = [
    # Tool collections
    "SEC_TOOLS",
    "MARKET_TOOLS",
    "NEWS_TOOLS",
    "RAG_TOOLS",
    # SEC tools
    "get_company_info",
    "get_recent_filings",
    "get_income_statement",
    "get_balance_sheet",
    "get_cash_flow",
    # Market tools
    "get_stock_quote",
    "get_daily_prices",
    "get_company_overview",
    "get_earnings",
    # News tools
    "search_financial_news",
    "verify_news_source",
    # RAG + A2A tools
    "search_filing_text",
    "corroborate_with_filing",
    # Memory
    "search_past_verifications",
    # Clients
    "TavilyClient",
]
