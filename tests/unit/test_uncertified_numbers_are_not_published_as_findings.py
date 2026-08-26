"""Two audit records that said things the system had not established.

Verified live on `Apple was fined 500 million euros by the European Commission`:

    news LLM        SUPPORTS
    news final      NOT_ENOUGH_INFO  conf 0.5   (override_applied)
    SEC delegate    8 filing passages read
    SEC final       NOT_ENOUGH_INFO  conf 0.5
    a2a status      NO_MATCHING_DISCLOSURE
    retrieved_value 500000000.0

Both agents failed closed correctly: `fine_amount` has no XBRL concept and
filing prose may never certify a number, so `compare_and_override` refused to
decide. That guard is right and is not touched here. What was wrong is
everything the record said afterwards.

**A -- the number.** `compare_and_override` sets
`verdict_output.retrieved_value` from the trusted observation at base.py:676,
under the comment "What the response shows must be what Python compared, not
what the model reported finding." The fail-closed branch returns *before* that
line, so the field kept the verdict LLM's own reading of the prose, and the
evidence dict published it. Downstream that field is indistinguishable from a
figure lifted out of an XBRL fact -- same name, same type, no marker. The
invariant was enforced on the success path and skipped on the failure path.

**B -- the status.** NO_MATCHING_DISCLOSURE asserts that the issuer's filing
does not mention this. Eight passages were read and the amount was found. The
cause is that `classify_status` maps every non-decisive verdict to that status,
conflating "the agent could not certify a value" with "the document is silent".
Only the second is a claim about the document.

FOUND_UNCERTIFIED is keyed on retrieved chunks, never on the model-populated
`retrieved_value`: deciding the status from a model's number would put model
output back in charge one layer up, which is defect A wearing a different hat.

Neither fix makes CORROBORATES or CONTRADICTS reachable. That needs an
event-bound narrative observation and is deliberately out of scope.
"""

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.models.a2a import (
    A2A_FAILED,
    A2A_FOUND_UNCERTIFIED,
    A2A_NO_MATCHING_DISCLOSURE,
    A2A_PENDING_CLASSIFICATION,
    A2A_SOURCE_UNAVAILABLE,
    reclassify_corroboration,
    summarize_filing_search,
)

MODEL_READING = 500_000_000.0


class _Agent(BaseVerificationAgent):
    def _get_source_description(self):
        return "SEC EDGAR"

    def _get_system_prompt(self):
        return "test"


class _Claim:
    claim_type = "news"
    metric = "fine_amount"
    operator = "eq"
    value = MODEL_READING
    range_min = None
    range_max = None
    period = None


def _run_fail_closed(monkeypatch):
    """The real `execute`, with no trusted observation available.

    Driven through the producer rather than asserting on a hand-built dict:
    the defect was in what the evidence builder emitted (CLAUDE.md gotcha 6).
    """
    agent = _Agent.__new__(_Agent)
    agent.agent_type = "sec"
    agent.max_iterations = 5
    monkeypatch.setattr(agent, "_build_context", lambda s: "ctx", raising=False)
    monkeypatch.setattr(
        agent, "react_agent",
        type("R", (), {"invoke": staticmethod(lambda *a, **k: {"messages": []})})(),
        raising=False)
    monkeypatch.setattr(agent, "_extract_tool_info",
                        lambda m: ([], [], [], []), raising=False)
    # What the verdict LLM claimed to read out of the filing prose.
    monkeypatch.setattr(
        agent, "_extract_verdict",
        lambda m, s: VerdictOutput(verdict="SUPPORTS", confidence=0.9,
                                   reasoning="found in the filing",
                                   retrieved_value=MODEL_READING),
        raising=False)
    # No structured source exists for fine_amount, so this is always None.
    monkeypatch.setattr("finvet.agents.base.resolve_trusted_observation",
                        lambda *a, **k: None)

    return agent.execute({"parsed_claim": _Claim(), "canonical_period": None,
                          "request_id": "req_1"})


class TestAModelReadingIsNotPublishedAsARetrievedValue:

    def test_the_verdict_still_fails_closed(self, monkeypatch):
        """The guard under test is not being weakened."""
        evidence = _run_fail_closed(monkeypatch)

        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["trusted_observation"] is None

    def test_retrieved_value_is_not_the_models_number(self, monkeypatch):
        evidence = _run_fail_closed(monkeypatch)

        assert evidence["retrieved_value"] is None, (
            "the verdict LLM's reading of prose was published in the same "
            "field that elsewhere holds a certified XBRL fact")

    def test_no_magnitude_difference_is_computed_from_it(self, monkeypatch):
        """A percentage against an uncertified number reads as measurement."""
        evidence = _run_fail_closed(monkeypatch)

        assert evidence.get("magnitude_difference_percent") is None


class TestBAFilingThatWasReadIsNotReportedAsSilent:

    def _result(self, chunks, success=True):
        return {
            "success": True,
            "status": A2A_PENDING_CLASSIFICATION,
            "verdict": "NOT_ENOUGH_INFO",
            "provenance": [{
                "tool": "search_filing_text",
                "result": {"success": success, "chunks": chunks},
            }],
        }

    def test_chunks_retrieved_means_found_uncertified(self):
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result([{"chunk_id": "c1"},
                                             {"chunk_id": "c2"}]))

        assert out["status"] == A2A_FOUND_UNCERTIFIED, (
            "eight passages were read and the record said the filing "
            "does not mention this")

    def test_a_genuinely_empty_search_still_reports_no_disclosure(self):
        """The status is not being retired -- an empty result still earns it."""
        out = reclassify_corroboration("NOT_ENOUGH_INFO", self._result([]))

        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    def test_a_search_that_errored_is_unchanged(self):
        out = reclassify_corroboration(
            "NOT_ENOUGH_INFO", self._result([], success=False))

        assert out["status"] == A2A_FAILED

    def test_an_unreachable_source_is_unchanged(self):
        result = self._result([], success=False)
        result["provenance"][0]["result"]["error"] = "SEC MCP unavailable"

        out = reclassify_corroboration("NOT_ENOUGH_INFO", result)

        assert out["status"] == A2A_SOURCE_UNAVAILABLE

    def test_the_status_does_not_depend_on_the_models_number(self):
        """Keying off retrieved_value would reintroduce defect A one layer up."""
        result = self._result([])
        result["retrieved_value"] = MODEL_READING

        out = reclassify_corroboration("NOT_ENOUGH_INFO", result)

        assert out["status"] == A2A_NO_MATCHING_DISCLOSURE

    def test_summarize_reports_whether_anything_came_back(self):
        summary = summarize_filing_search([{
            "tool": "search_filing_text",
            "result": {"success": True, "chunks": [{"chunk_id": "c1"}]}}])

        assert summary["searched"] is True
        assert summary["retrieved"] is True


class TestNeitherFixMakesADisagreementReachable:
    """Guards the scope boundary. FOUND_UNCERTIFIED is an honest label for a
    non-decisive outcome, not a route to one."""

    def test_found_uncertified_is_not_a_decisive_status(self):
        from finvet.models.a2a import A2A_CONTRADICTS, A2A_CORROBORATES

        assert A2A_FOUND_UNCERTIFIED not in (A2A_CONTRADICTS, A2A_CORROBORATES)

    def test_it_does_not_trigger_human_review(self):
        """Only CONTRADICTS raises source_disagreement. A filing that was read
        but could not certify a figure is not a conflict between sources."""
        import inspect

        from finvet.graph.nodes import output_guardrails

        source = inspect.getsource(output_guardrails)
        assert "A2A_FOUND_UNCERTIFIED" not in source
