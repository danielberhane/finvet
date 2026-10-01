"""The RAGAS runner grades the model's words against the filing text, nothing else.

`scripts/eval_rag_ragas.py` hands RAGAS two things taken from FinVet's
response: the model's reasoning and the retrieved passages. Both arrive mixed
with text FinVet's own code adds -- a summary block appended to the reasoning,
and a delimiter wrapped around each passage. Graded as-is, the summary would
count as the model's words and the delimiter as filing text.

These tests build the response with FinVet's real producers -- the
`search_filing_text` tool over a fake index, and `response_generator` -- and
check the runner's helpers recover exactly the model's reasoning and the
stored passage text. If either producer changes its format, these fail
before a RAGAS run can silently grade the wrong text.
"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT = Path("scripts/eval_rag_ragas.py")
REASONING = ("The 10-K states that many components come from a single supplier, "
             "which supports the claim.")
PASSAGE = "Many components we use are available from only one or a few suppliers."


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("eval_rag_ragas", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retrieved_chunks():
    """Chunks exactly as search_filing_text returns them, from a fake index."""
    from finvet.rag.types import RAGSearchResult
    from finvet.tools.filing_search import search_filing_text

    hit = RAGSearchResult(
        evidence_id="a" * 64, ticker="AAPL", cik="0000320193", filing_type="10-K",
        filing_date="2025-10-31", period_end="2025-09-27", part=None, item_number="1A",
        section="risk_factors", section_title="Risk Factors", chunk_index=12,
        chunk_text=PASSAGE, content_sha256="b" * 64, vector_similarity=0.71,
        vector_rank=1, keyword_score=0.0, keyword_rank=None, rrf_score=0.0164)
    rag = MagicMock()
    rag.available = True
    rag.search.return_value = [hit]
    with patch("finvet.tools.filing_search.get_rag_service", return_value=rag), \
         patch("finvet.tools.filing_search._current_period_target",
               return_value=(None, None)):
        result = search_filing_text.invoke(
            {"query": "supplier concentration", "ticker": "AAPL"})
    return result["chunks"]


def _response(agent_evidence, chunks):
    """The response FinVet's /verify returns, built by its response generator."""
    from finvet.graph.nodes.response_generator import response_generator
    from finvet.models.claim import ParsedClaim

    out = response_generator({
        "request_id": "req_000000000042",
        "claim_raw": "Apple's 10-K says it depends on few suppliers",
        "parsed_claim": ParsedClaim(claim_type="sec", ticker="AAPL", metric=None,
                                    operator=None, value=None),
        "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS", "confidence": 0.9,
                           "tools_called": ["search_filing_text"], **agent_evidence},
        "rag_chunks_retrieved": chunks,
        "final_verdict": "SUPPORTS", "final_confidence": 0.9,
        "execution_start_time": "2026-10-01T00:00:00",
    })
    return out["final_response"]


class TestTheModelsReasoningIsWhatGetsGraded:

    def test_the_appended_summary_is_removed(self, runner):
        response = _response({"reasoning": REASONING,
                              "source_description": "SEC EDGAR (XBRL GAAP financial statements)",
                              "retrieved_value": 391_035_000_000,
                              "magnitude_difference_percent": 0.0},
                             _retrieved_chunks())
        assert "- Data source:" in response["explanation"], "precondition: the block is there"
        assert runner.model_reasoning(response["explanation"]) == REASONING

    def test_reasoning_without_a_summary_is_left_whole(self, runner):
        response = _response({"reasoning": REASONING}, _retrieved_chunks())
        assert runner.model_reasoning(response["explanation"]) == REASONING

    def test_a_reasoning_paragraph_break_is_not_mistaken_for_the_summary(self, runner):
        two_paragraphs = REASONING + "\n\nThe passage names no specific supplier."
        response = _response({"reasoning": two_paragraphs,
                              "source_description": "SEC EDGAR (XBRL GAAP financial statements)"},
                             _retrieved_chunks())
        assert runner.model_reasoning(response["explanation"]) == two_paragraphs

    def test_no_reasoning_yields_nothing_to_grade(self, runner):
        response = _response({}, _retrieved_chunks())
        assert runner.model_reasoning(response["explanation"]) == ""

    def test_code_written_reasoning_is_detected(self, runner):
        from finvet.agents.base import deterministic_reasoning
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import TrustedObservation

        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                            operator="eq", value=391e9)
        obs = TrustedObservation(
            tool="get_income_statement", metric="revenue", value=391_035_000_000,
            units="USD", period_end="2025-09-27", concept="Revenues")
        assert runner.is_code_written(deterministic_reasoning(claim, obs, "SUPPORTS"))
        assert not runner.is_code_written(REASONING)


class TestThePassagesAreTheStoredFilingText:

    def test_the_delimiter_is_removed(self, runner):
        chunks = _retrieved_chunks()
        assert chunks[0]["chunk_text"].startswith("<filing_excerpt>"), "precondition: wrapped"
        response = _response({"reasoning": REASONING}, chunks)
        assert runner.passage_texts(response) == [PASSAGE]

    def test_passage_identities_carry_the_hash_not_the_text(self, runner):
        response = _response({"reasoning": REASONING}, _retrieved_chunks())
        ids = runner.passage_identities(response)
        assert ids == [{"content_sha256": "b" * 64, "section": "risk_factors",
                        "filing_type": "10-K", "period_end": "2025-09-27"}]

    def test_no_retrieval_yields_no_passages(self, runner):
        response = _response({"reasoning": REASONING}, [])
        assert runner.passage_texts(response) == []


class TestTheRunRefusesWhatWouldSpoilIt:

    def test_an_altered_claim_file_is_refused(self, runner, tmp_path):
        golden = tmp_path / "golden_c.jsonl"
        golden.write_text("\n".join(
            '{"id": %d, "claim": "c%d", "expected": {"verdict": "SUPPORTS"}}' % (i, i)
            for i in runner.CLAIM_IDS))
        extra = tmp_path / "extra.jsonl"
        extra.write_text('{"id": 2001, "claim": "edited after the freeze", "kind": "supported"}\n')
        with pytest.raises(SystemExit, match="frozen hash"):
            runner.load_claims(tmp_path, extra)

    def test_the_private_record_must_live_under_notes(self, runner):
        with pytest.raises(SystemExit, match="notes/"):
            runner.private_path("docs/eval/runs/record.json")
        assert runner.private_path("notes/rag-ragas-record.json").name == "rag-ragas-record.json"


class TestThePublicArtifactCarriesNoText:

    def _record(self):
        passes = [{"faithfulness": 1.0, "answer_relevancy": 0.9, "context_precision": 1.0,
                   "reasons": {"faithfulness": "restates the held-out claim"}}] * 2
        return {"id": 2001, "kind": "supported", "origin": "rag_ragas_claims_v1",
                "claim": "HELD-OUT CLAIM TEXT", "expected_verdict": "SUPPORTS", "http": 200,
                "verdict": "SUPPORTS", "confidence": 0.9, "reasoning": "MODEL REASONING TEXT",
                "passages": ["FILING PASSAGE TEXT"], "passage_identities": [{"content_sha256": "b" * 64}],
                "judge_passes": passes, "faithfulness_spread": 0.0, "unstable": False}

    def test_no_claim_reasoning_passage_or_judge_text_is_published(self, runner):
        published = json.dumps(runner.public_view(self._record()))
        for secret in ("HELD-OUT CLAIM TEXT", "MODEL REASONING TEXT", "FILING PASSAGE TEXT",
                       "restates the held-out claim"):
            assert secret not in published

    def test_the_summary_reports_each_kind_and_the_bar(self, runner):
        rec = self._record()
        other = {**rec, "id": 2029, "kind": "not_in_filing", "expected_verdict": "NOT_ENOUGH_INFO",
                 "verdict": "NOT_ENOUGH_INFO", "excluded": "no passages retrieved"}
        other.pop("judge_passes")
        summary = runner.summarise([rec, other])
        assert summary["all"]["claims"] == 2 and summary["all"]["scored"] == 1
        assert summary["by_kind"]["not_in_filing"]["excluded"] == 1
        assert summary["by_kind"]["supported"]["faithfulness_mean"] == 1.0
        assert summary["bar"]["faithfulness_mean"] == 0.85
