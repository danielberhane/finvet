"""Provenance tools must return something _parse_provenance can recover.

The regression this pins: both tools returned a pydantic BaseModel. LangChain
stringifies a tool's return value, and a BaseModel's repr is
"success=True chunks=[...]" — neither JSON nor a Python literal. So
_parse_provenance fell through to {"raw": ...}, run_sec_agent's
`prov_result.get("success")` was None, and every retrieved chunk was dropped
before reaching data_sources, the RAG badge, or the audit trail. The agent used
RAG correctly and the system could not prove it.
"""

from typing import Any, Dict

from finvet.agents.base import BaseVerificationAgent
from finvet.tools.corroborate import CorroborationResult, corroborate_with_news
from finvet.tools.filing_search import FilingSearchResult, search_filing_text


def _roundtrip(payload) -> Dict[str, Any]:
    """Serialize the way LangChain does, then parse the way base.py does."""
    return BaseVerificationAgent._parse_provenance(str(payload))


class TestProvenanceRoundTrip:

    def test_filing_search_result_survives_as_dict(self):
        payload = FilingSearchResult(
            success=True,
            chunks=[{"chunk_text": "Services net sales were $109,158 million"}],
            total_found=1,
        ).model_dump()
        parsed = _roundtrip(payload)
        assert parsed.get("success") is True
        assert parsed["chunks"][0]["chunk_text"].startswith("Services net sales")

    def test_corroboration_result_survives_as_dict(self):
        payload = CorroborationResult(
            success=True, news_verdict="SUPPORTS", news_confidence=0.9,
            news_reasoning="two outlets confirm", sources_checked=2,
        ).model_dump()
        parsed = _roundtrip(payload)
        assert parsed.get("success") is True
        assert parsed["news_verdict"] == "SUPPORTS"

    def test_bare_model_does_not_survive(self):
        """The shape of the bug, pinned: returning the model loses everything."""
        payload = FilingSearchResult(success=True, chunks=[{"chunk_text": "x"}],
                                     total_found=1)
        parsed = _roundtrip(payload)
        assert parsed.get("success") is None
        assert list(parsed) == ["raw"]


class TestToolsReturnDicts:
    """run_sec_agent requires isinstance(result, dict) and result['success'],
    so the declared return type is load-bearing, not cosmetic."""

    def test_filing_search_returns_dict_on_error(self, monkeypatch):
        monkeypatch.setattr("finvet.tools.filing_search.get_rag_service",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        out = search_filing_text.invoke(
            {"query": "q", "ticker": "AAPL"})
        assert isinstance(out, dict)
        assert out["success"] is False
        assert "boom" in out["error"]

    def test_corroborate_returns_dict_on_error(self, monkeypatch):
        monkeypatch.setattr(
            "finvet.agents.news_agent.react_agent.NewsAgent",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("news down")))
        out = corroborate_with_news.invoke(
            {"finding": "f", "query": "q", "ticker": "AAPL"})
        assert isinstance(out, dict)
        assert out["success"] is False
