"""Episodic memory tool for agent ReAct loop."""

from typing import Optional

from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from ..config.constants import MEMORY_CONTEXT_THRESHOLD
from ..utils.logging import get_logger

logger = get_logger(__name__)

NAMESPACE = ("claims",)
_store: Optional[BaseStore] = None


def _get_store() -> Optional[BaseStore]:
    return _store


def _set_store(store: BaseStore) -> None:
    global _store
    _store = store


@tool
def search_past_verifications(
    query: str,
    ticker: Optional[str] = None,
) -> str:
    """Search for similar past claim verifications.

    Use this to check if a similar claim has been verified before.
    Returns past verdicts, confidence scores, tools that worked,
    and key findings.

    Args:
        query: What you're looking for (e.g., "Apple revenue FY2024")
        ticker: Optional stock ticker to narrow results (e.g., "AAPL")
    """
    store = _get_store()
    if store is None:
        return "Memory search unavailable."
    try:
        filter_dict = {}
        if ticker:
            filter_dict["ticker"] = ticker.upper()
        results = store.search(
            NAMESPACE,
            query=query,
            filter=filter_dict if filter_dict else None,
            limit=3,
        )
        if not results:
            return "No similar past verifications found."
        episodes = []
        for item in results:
            if item.score < MEMORY_CONTEXT_THRESHOLD:
                continue
            val = item.value
            episode = (
                f"Past verification (similarity: {item.score:.2f}):\n"
                f"  Claim: {val['claim_text']}\n"
                f"  Verdict: {val['verdict']} (confidence: {val['confidence']:.2f})\n"
            )
            if val.get("retrieved_value"):
                episode += f"  Retrieved value: {val['retrieved_value']:,.0f}\n"
            if val.get("tools_called"):
                episode += f"  Tools used: {', '.join(val['tools_called'])}\n"
            if val.get("key_finding"):
                episode += f"  Key finding: {val['key_finding']}\n"
            episodes.append(episode)
        if not episodes:
            return "No similar past verifications found above confidence threshold."
        return (
            f"Found {len(episodes)} relevant past verification(s):\n\n"
            + "\n".join(episodes)
            + "\nNote: Use these as reference only. Verify independently with current data."
        )
    except Exception as e:
        logger.warning(f"Past verification search failed: {e}")
        return "Memory search unavailable."
