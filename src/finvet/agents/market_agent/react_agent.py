"""Market ReAct Agent for verifying claims about market data.

This agent verifies claims about stock prices, market capitalization,
P/E ratios, and other market-derived metrics using Finnhub data.
"""

from pathlib import Path

from ..base import BaseVerificationAgent
from ...config.constants import AGENT_MAX_ITERATIONS
from ...tools.market_tools import MARKET_TOOLS
from ...tools.memory_tools import search_past_verifications

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "market_system.txt"
MARKET_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


class MarketAgent(BaseVerificationAgent):
    """
    Market verification agent using ReAct pattern.

    This agent queries Finnhub to verify claims about stock prices,
    market cap, valuations, and other market-derived metrics.
    """

    def __init__(self, max_iterations: int = AGENT_MAX_ITERATIONS):
        """Initialize the Market Agent with market tools."""
        super().__init__(
            agent_type="market",
            tools=MARKET_TOOLS + [search_past_verifications],
            system_prompt=MARKET_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    def _get_source_description(self) -> str:
        """Return description of Finnhub data source."""
        return "Finnhub Market Data"
