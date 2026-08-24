"""Filing text search tool for the SEC agent.

Uses the RAG service to search SEC filing narrative text (MD&A, risk factors,
footnotes, segment breakdowns, non-GAAP metrics) that isn't available through
structured XBRL data.
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..rag.service import get_rag_service
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FilingSearchResult(BaseModel):
    """Result from search_filing_text tool."""
    success: bool = Field(..., description="Whether the search succeeded")
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Matching text chunks with section, filing_type, period_end, score",
    )
    total_found: int = Field(0, description="Number of matching chunks returned")
    error: Optional[str] = Field(None, description="Error message if failed")


@tool
def search_filing_text(
    query: str,
    ticker: str,
    section: str = "",
    filing_type: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search the narrative text of SEC filings (10-K, 10-Q) for information
    NOT available in structured XBRL financial statements.

    Use this tool to find information in filing prose such as:
    - Segment revenue breakdowns (e.g., "iPhone revenue", "Services segment")
    - Risk factor disclosures (e.g., "supply chain risks", "regulatory risks")
    - MD&A commentary (management discussion about performance drivers)
    - Non-GAAP metrics and reconciliations
    - Legal proceedings and contingencies
    - Guidance and forward-looking statements
    - Footnotes with accounting policy details

    Do NOT use this tool for standard GAAP line items (total revenue, net income,
    EPS, total assets) — use get_income_statement, get_balance_sheet, or
    get_cash_flow instead.

    Args:
        query: Natural language search query describing what to find.
        ticker: Company ticker symbol (e.g., "AAPL", "TSLA").
        section: Optional section filter. Values: "risk_factors", "mda",
                 "business", "financial_statements_and_notes", "market_risk",
                 "cybersecurity", "controls_and_procedures", "executive_compensation",
                 "legal_proceedings".
        filing_type: Optional filing type filter ("10-K" or "10-Q").
        top_k: Number of results to return (default 5).

    Returns:
        FilingSearchResult fields as a dict: success, chunks, total_found, error.
    """
    # Returned as a dict, not the model: LangChain stringifies a tool's return
    # value, and a BaseModel's repr ("success=True chunks=[...]") is neither
    # JSON nor a Python literal, so _parse_provenance cannot recover it and the
    # retrieved chunks never reach data_sources or the audit trail.
    try:
        rag = get_rag_service()
        if not rag.available:
            return FilingSearchResult(
                success=False,
                error="RAG service not available (OpenAI API key not configured)",
            ).model_dump()

        results = rag.search(
            query=query,
            ticker=ticker,
            section=section or None,
            filing_type=filing_type or None,
            top_k=top_k,
        )

        logger.info(f"Filing search for '{query}' ({ticker}): {len(results)} results")

        return FilingSearchResult(
            success=True,
            chunks=results,
            total_found=len(results),
        ).model_dump()

    except Exception as e:
        logger.error(f"search_filing_text failed: {e}")
        return FilingSearchResult(success=False, error=str(e)).model_dump()
