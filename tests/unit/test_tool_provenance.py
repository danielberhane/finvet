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
from finvet.tools.corroborate_sec import corroborate_with_filing
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
            "finvet.graph.nodes.domain_agents.run_sec_agent_scoped",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sec down")))
        out = corroborate_with_filing.invoke(
            {"finding": "f", "ticker": "AAPL"})
        assert isinstance(out, dict)
        assert out["success"] is False


class TestEveryToolReturnsADict:
    """Task 2 extended the dict-return rule from two tools to all of them.

    Nothing structural stops someone reverting a tool to `return XResult(...)`.
    That change is invisible in review and silent at runtime: LangChain
    stringifies the model, _parse_provenance recovers only {"raw": ...}, the
    payload has no `success` field, and the call becomes untrusted -- so every
    numeric claim served by that tool quietly degrades to NOT_ENOUGH_INFO.
    This pins the rule against the source rather than against a fixture.
    """

    TOOL_MODULES = ["sec_tools", "market_tools", "macro_tools", "news_tools",
                    "filing_search", "corroborate_sec"]

    def test_no_tool_returns_a_bare_result_model(self):
        import inspect
        import re

        import finvet.tools as tools_pkg

        offenders = []
        for name in self.TOOL_MODULES:
            module = __import__(f"finvet.tools.{name}", fromlist=["*"])
            source = inspect.getsource(module)
            for match in re.finditer(r"return ([A-Z][A-Za-z]*Result)\(", source):
                tail = source[match.end():]
                depth, i = 1, 0
                while depth:
                    if tail[i] == "(":
                        depth += 1
                    elif tail[i] == ")":
                        depth -= 1
                    i += 1
                if not tail[i:i + 13] == ".model_dump()":
                    offenders.append(f"{name}: {match.group(1)}")
        assert not offenders, (
            "these tool returns are bare models and will lose their provenance: "
            + ", ".join(offenders))
        assert tools_pkg  # the package imports cleanly

    def test_record_classifies_a_real_failed_tool_result(self, monkeypatch):
        """A caught failure is a transport success and an application failure."""
        from finvet.models.evidence import tool_record_from_result
        from finvet.tools.sec_tools import get_income_statement

        monkeypatch.setattr(
            "finvet.tools.sec_tools._get_client",
            lambda: (_ for _ in ()).throw(RuntimeError("MCP down")))
        result = get_income_statement.invoke(
            {"cik": "1", "accession_number": "a", "period": "annual"})

        record = tool_record_from_result("get_income_statement", {}, result)
        assert record.transport_success is True
        assert record.application_success is False
        assert record.trusted_success is False
        assert record.call_succeeded is False

    def test_record_classifies_a_real_successful_tool_result(self, monkeypatch):
        from finvet.mcp.sec_edgar import FinancialItem
        from finvet.models.evidence import tool_record_from_result
        from finvet.tools import sec_tools

        class _Client:
            def get_financials(self, **kwargs):
                return [FinancialItem(line_item="Revenues", concept="Revenues",
                                      value=391_035_000_000.0, units="USD",
                                      period="annual", period_end="2024-09-28")]

        monkeypatch.setattr(sec_tools, "_get_client", lambda: _Client())
        result = sec_tools.get_income_statement.invoke(
            {"cik": "1", "accession_number": "a", "period": "annual"})

        record = tool_record_from_result("get_income_statement", {}, result)
        assert record.trusted_success is True
        assert record.payload["items"][0]["value"] == 391_035_000_000.0

    def test_unknown_payload_shape_is_untrusted_but_not_failed(self):
        """A tool returning a plain string has not failed, but is not evidence."""
        from finvet.models.evidence import tool_record_from_result

        record = tool_record_from_result("search_past_verifications", {},
                                         "No similar past verifications found.")
        assert record.call_succeeded is True     # nothing went wrong
        assert record.trusted_success is False   # and nothing is verifiable
