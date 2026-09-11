"""News search tools for the News verification agent.

These tools provide access to news search through Tavily. The agent uses
these tools to verify claims that require news corroboration, such as
announcements, events, and publicly reported information.

Tool Selection Guide for the LLM:
---------------------------------
1. search_financial_news: Search for news articles about financial claims
2. verify_news_source: Check the credibility of a news source
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .tavily_search import TavilyClient
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Singleton client instance (lazy initialization)
_tavily_client: Optional[TavilyClient] = None

# Credibility tiers for news sources
TIER_1_SOURCES = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "apnews.com",
    "sec.gov",
    "businesswire.com",
    "prnewswire.com",
]

TIER_2_SOURCES = [
    "cnbc.com",
    "marketwatch.com",
    "yahoo.com",
    "barrons.com",
    "seekingalpha.com",
    "thestreet.com",
    "investopedia.com",
]


def _get_client() -> TavilyClient:
    """Get or create the Tavily client singleton."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


def _set_client(client: TavilyClient) -> None:
    """Inject a client instance (for testing)."""
    global _tavily_client
    _tavily_client = client


def _calculate_credibility(url: str) -> float:
    """A domain allowlist wearing a score, and it decides nothing.

    Three tiers by substring match: a wire service or major financial paper,
    a recognised outlet, or anything else. The numbers are ordering, not
    probability -- nobody measured that Reuters is 0.95 accurate. They exist
    so search results sort with the better-known sources first and so the
    model can cite a reason when it weighs two conflicting articles.

    Worth being explicit about what this cannot touch: no numeric verdict
    depends on it. The comparator reads a trusted observation and nothing
    else, so a high score cannot promote a figure lifted from prose, and a
    low one cannot suppress a filed fact. Treat it as presentation.

    Substring matching is deliberately crude and will happily score a URL
    that merely mentions a tier-one domain in its path. That is acceptable
    for ordering; it would not be acceptable if anything downstream trusted
    the number, which is the reason nothing does.
    """
    url_lower = url.lower()

    for source in TIER_1_SOURCES:
        if source in url_lower:
            return 0.95

    for source in TIER_2_SOURCES:
        if source in url_lower:
            return 0.80

    # Default score for unknown sources
    return 0.50


class NewsSearchResult(BaseModel):
    """Result from search_financial_news tool."""
    success: bool = Field(..., description="Whether the search succeeded")
    query: str = Field(..., description="The search query used")
    articles: List[Dict[str, Any]] = Field(default_factory=list, description="Found articles")
    count: int = Field(0, description="Number of articles found")
    error: Optional[str] = Field(None, description="Error message if failed")


class SourceCredibilityResult(BaseModel):
    """Result from verify_news_source tool."""
    url: str = Field(..., description="The URL that was checked")
    domain: str = Field(..., description="Domain of the source")
    credibility_score: float = Field(..., description="Credibility score 0.0-1.0")
    credibility_tier: str = Field(..., description="Tier: HIGH, MEDIUM, or LOW")
    is_primary_source: bool = Field(..., description="Whether this is a primary source")
    notes: List[str] = Field(default_factory=list, description="Notes about the source")


@tool
def search_financial_news(
    query: str,
    company_ticker: Optional[str] = None,
    max_results: int = 10,
    use_credible_sources_only: bool = True,
) -> Dict[str, Any]:
    """
    Search for news articles about a financial claim or event.

    Use this for claims about:
    - Earnings announcements ("Apple announced record revenue")
    - Executive changes ("Tesla CEO Elon Musk...")
    - M&A activity ("Microsoft acquired Activision")
    - Product launches and business events
    - Analyst ratings and recommendations
    - Any claim that requires news corroboration

    The search prioritizes credible financial news sources (Reuters, Bloomberg,
    WSJ, etc.) to ensure reliable information for verification.

    Args:
        query: Search query describing what to find (be specific!)
        company_ticker: Optional ticker to focus search (e.g., "AAPL")
        max_results: Maximum articles to return (default 10)
        use_credible_sources_only: If True, restricts to Tier 1 financial sources

    Returns:
        NewsSearchResult with list of relevant articles, URLs, and snippets
    """
    try:
        client = _get_client()

        # Build search query
        search_query = query
        if company_ticker:
            search_query = f"{company_ticker} {query}"

        # Include credible sources if requested
        include_domains = TIER_1_SOURCES if use_credible_sources_only else None

        results = client.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=include_domains,
        )

        articles = []
        for result in results:
            url = result.get("url", "")
            articles.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("content", "")[:500],  # Limit snippet length
                "published_date": result.get("published_date"),
                "source_domain": url.split("/")[2] if "/" in url else url,
                "credibility_score": _calculate_credibility(url),
            })

        # Sort by credibility
        articles.sort(key=lambda x: x["credibility_score"], reverse=True)

        return NewsSearchResult(
            success=True,
            query=search_query,
            articles=articles,
            count=len(articles),
        ).model_dump()

    except Exception as e:
        logger.error(f"search_financial_news failed: {e}")
        return NewsSearchResult(
            success=False,
            query=query,
            error=str(e),
        ).model_dump()


@tool
def verify_news_source(url: str) -> Dict[str, Any]:
    """
    Check the credibility of a news source URL.

    Use this to assess whether a source is reliable before using it as evidence.
    The tool checks:
    - Whether it's a Tier 1 source (Reuters, Bloomberg, WSJ, SEC, wire services)
    - Whether it's a Tier 2 source (CNBC, MarketWatch, Yahoo Finance)
    - Whether it's a primary source (company press release, SEC filing)

    Credibility scoring:
    - 0.90-1.00: Tier 1 (highly credible, use as primary evidence)
    - 0.70-0.89: Tier 2 (credible, good for corroboration)
    - 0.50-0.69: Tier 3 (use with caution, seek corroboration)
    - Below 0.50: Low credibility (not recommended for verification)

    Args:
        url: The URL to check

    Returns:
        SourceCredibilityResult with credibility score and tier
    """
    try:
        # Extract domain from URL
        if "://" in url:
            domain = url.split("/")[2]
        else:
            domain = url.split("/")[0]

        domain = domain.lower().replace("www.", "")

        credibility_score = _calculate_credibility(url)

        # Determine tier
        if credibility_score >= 0.90:
            tier = "HIGH"
        elif credibility_score >= 0.70:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # Check if primary source
        is_primary = any(s in domain for s in [
            "sec.gov",
            "businesswire.com",
            "prnewswire.com",
            "globenewswire.com",
        ])

        notes = []
        if is_primary:
            notes.append("This is a primary source (company press release or SEC filing)")
        if tier == "HIGH":
            notes.append("Tier 1 financial news source - highly credible")
        elif tier == "MEDIUM":
            notes.append("Tier 2 financial news source - credible for corroboration")
        else:
            notes.append("Unknown or low-tier source - seek additional corroboration")

        return SourceCredibilityResult(
            url=url,
            domain=domain,
            credibility_score=credibility_score,
            credibility_tier=tier,
            is_primary_source=is_primary,
            notes=notes,
        ).model_dump()

    except Exception as e:
        logger.error(f"verify_news_source failed for {url}: {e}")
        return SourceCredibilityResult(
            url=url,
            domain="unknown",
            credibility_score=0.0,
            credibility_tier="LOW",
            is_primary_source=False,
            notes=[f"Error checking source: {str(e)}"],
        ).model_dump()


# Export all tools for the agent
NEWS_TOOLS = [
    search_financial_news,
    verify_news_source,
]
