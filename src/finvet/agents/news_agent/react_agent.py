"""News ReAct Agent for verifying claims through news sources.

This agent verifies claims about announcements, events, and publicly
reported information using news search via Tavily.
"""

from pathlib import Path

from ..base import BaseVerificationAgent
from ...config.constants import AGENT_MAX_ITERATIONS
from ...tools.corroborate_sec import corroborate_with_filing
from ...tools.macro_tools import get_macro_indicator
from ...tools.news_tools import NEWS_TOOLS
from ...tools.memory_tools import search_past_verifications

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "news_system.txt"
NEWS_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


class NewsAgent(BaseVerificationAgent):
    """
    News verification agent using ReAct pattern.

    This agent searches news sources via Tavily to verify claims about
    announcements, events, and publicly reported information.
    """

    # Without this the class inherits the empty set from BaseVerificationAgent
    # and the A2A result is never captured as provenance — the evidence would
    # reach a verdict and then vanish before the audit trail.
    _provenance_tool_names = {"corroborate_with_filing"}

    def __init__(self, max_iterations: int = AGENT_MAX_ITERATIONS):
        """Initialize the News Agent with news + A2A tools."""
        super().__init__(
            agent_type="news",
            tools=NEWS_TOOLS + [
                get_macro_indicator,
                search_past_verifications,
                corroborate_with_filing,
            ],
            system_prompt=NEWS_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    def _get_source_description(self) -> str:
        """Return description of news data sources."""
        return "Financial News (Tavily search of Reuters, Bloomberg, WSJ, etc.)"
