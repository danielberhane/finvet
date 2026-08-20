"""SEC ReAct Agent for verifying claims against SEC EDGAR filings.

This agent is the authoritative source for GAAP accounting metrics.
It uses tools to query SEC EDGAR for financial statement data and
applies reasoning to verify claims about revenue, earnings, and
other financial metrics.
"""

from pathlib import Path

from ..base import BaseVerificationAgent
from ...config.constants import AGENT_MAX_ITERATIONS
from ...tools.sec_tools import SEC_TOOLS
from ...tools.filing_search import search_filing_text
from ...tools.corroborate import corroborate_with_news
from ...tools.memory_tools import search_past_verifications

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sec_system.txt"
SEC_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


class SECAgent(BaseVerificationAgent):
    """
    SEC verification agent using ReAct pattern.

    This agent queries SEC EDGAR to verify claims about company financials
    using GAAP-compliant data from 10-K and 10-Q filings, and can search
    filing narrative text via RAG and cross-verify with the News agent.
    """

    # Track full results from RAG and A2A tools for provenance auditing
    _provenance_tool_names = {"search_filing_text", "corroborate_with_news"}

    def __init__(self, max_iterations: int = AGENT_MAX_ITERATIONS):
        """Initialize the SEC Agent with SEC + RAG + A2A tools."""
        super().__init__(
            agent_type="sec",
            tools=SEC_TOOLS + [search_filing_text, corroborate_with_news, search_past_verifications],
            system_prompt=SEC_SYSTEM_PROMPT,
            max_iterations=max_iterations,
        )

    def _get_source_description(self) -> str:
        """Return description of SEC EDGAR data source."""
        return "SEC EDGAR (XBRL GAAP financial statements)"
