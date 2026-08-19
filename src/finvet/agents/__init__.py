"""Domain verification agents using ReAct pattern.

Each agent uses DeepSeek as the LLM brain and has access to domain-specific
tools for retrieving evidence. Agents use the ReAct (Reasoning + Acting)
pattern to intelligently select tools and verify claims.

Agent Types:
- SECAgent: For GAAP financial claims (revenue, earnings, etc.)
- MarketAgent: For market data claims (prices, valuations)
- NewsAgent: For news-based claims (announcements, events)
"""

from .base import BaseVerificationAgent
from .sec_agent.react_agent import SECAgent
from .market_agent.react_agent import MarketAgent
from .news_agent.react_agent import NewsAgent

__all__ = [
    "BaseVerificationAgent",
    "SECAgent",
    "MarketAgent",
    "NewsAgent",
]
