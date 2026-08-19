"""Agent-to-Agent corroboration tool.

Allows the SEC agent to ask the News agent to corroborate findings
from SEC filing text, enabling cross-source verification.
"""

from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CorroborationResult(BaseModel):
    """Result from corroborate_with_news tool."""
    success: bool = Field(..., description="Whether corroboration completed")
    news_verdict: str = Field(
        "NOT_ENOUGH_INFO",
        description="News agent verdict: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO",
    )
    news_confidence: float = Field(0.0, description="News agent confidence 0.0-1.0")
    news_reasoning: str = Field("", description="News agent reasoning")
    sources_checked: int = Field(0, description="Number of news sources checked")
    error: Optional[str] = Field(None, description="Error message if failed")


@tool
def corroborate_with_news(
    finding: str,
    query: str,
    ticker: str,
) -> CorroborationResult:
    """Ask the News agent to corroborate a finding from SEC filing text.

    Use this AFTER finding something significant in a filing via
    search_filing_text — for example, a risk disclosure, an M&A
    announcement, a guidance change, or a material event — to check
    whether news sources confirm it.

    This provides cross-source verification: the SEC filing is the
    primary source, and news coverage acts as independent confirmation.

    Only use when the finding is significant and corroboration adds value.
    Do NOT use for routine financial metrics.

    Args:
        finding: The specific finding from the SEC filing to corroborate.
                 (e.g., "Apple disclosed supply chain concentration risk in China")
        query: A search query to find relevant news articles.
               (e.g., "Apple supply chain China risk 2024")
        ticker: Company ticker symbol (e.g., "AAPL").

    Returns:
        CorroborationResult with the News agent's verdict and reasoning.
    """
    try:
        # Import here to avoid circular imports
        from ..agents.news_agent.react_agent import NewsAgent

        logger.info(f"A2A corroboration: SEC → News for '{finding[:80]}...'")

        # Build a minimal state for the News agent
        corroboration_state = {
            "claim_raw": f"Verify this SEC filing disclosure: {finding}",
            "request_id": "a2a-corroboration",
        }

        # Run the News agent with fewer iterations (focused search)
        news_agent = NewsAgent(max_iterations=3)
        evidence = news_agent.execute(corroboration_state)

        logger.info(
            f"A2A corroboration result: {evidence['verdict']} "
            f"(confidence: {evidence['confidence']:.2f})"
        )

        return CorroborationResult(
            success=True,
            news_verdict=evidence.get("verdict", "NOT_ENOUGH_INFO"),
            news_confidence=evidence.get("confidence", 0.0),
            news_reasoning=evidence.get("reasoning", ""),
            sources_checked=len(evidence.get("tools_called", [])),
        )

    except Exception as e:
        logger.error(f"corroborate_with_news failed: {e}")
        return CorroborationResult(success=False, error=str(e))
