"""Tavily web search client wrapper."""

from typing import Dict, List, Optional
from tavily import TavilyClient as TavilySDK
from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TavilyClient:
    """Wrapper around Tavily web search API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key. If not provided, uses settings.tavily_api_key
        """
        self.api_key = api_key or settings.tavily_api_key
        self.client = TavilySDK(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_range: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced"
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            time_range: Optional time range filter

        Returns:
            List of search result dictionaries
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            results = response.get("results", [])
            logger.info(f"Tavily search returned {len(results)} results for: {query}")

            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []
