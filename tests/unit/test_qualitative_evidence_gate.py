"""A decisive verdict on a claim with no number must rest on something retrieved.

The defect, observed live and reproducibly (4/4 runs):

    "Apple's annual report discusses its plans to open a theme park in Ohio"
    -> REFUTES, confidence 0.95

The News agent called `search_financial_news` four times, every call succeeded,
every call returned zero articles, and the model read that silence as proof the
claim was false. Nothing in the pipeline objected. `_apply_override` is the one
component whose job is to overrule the model, but its only guard is numeric --
`claimed_val is not None and observation is None` -- and this claim names no
value, so the guard did not apply and the model's verdict was released
untouched at 0.95.

That is absence of evidence presented as evidence of absence, which D9 forbids
on the delegation path and `a2a.summarize_filing_search` already enforces there:
"the filing does not mention this" may only be said if a document was actually
read. The same sentence is not permissible just because it arrives from the
direct route instead of a nested one.

Note what the fix must *not* do. A qualitative claim is not a broken one --
where retrieval genuinely returned articles or filing chunks, the model has
real text to reason from and a decisive verdict stands. The gate asks only
whether anything was retrieved, never whether the model read it correctly.

`search_past_verifications` is deliberately not a retrieval tool. A prior
verification is this system's own earlier output; treating it as evidence about
the world would let a verdict bootstrap itself.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.models.evidence import qualitative_evidence_gap


class _Agent(BaseVerificationAgent):
    """Concrete stub. The base class is abstract; only `_apply_override` is
    under test and it touches neither of these."""

    def _get_source_description(self):
        return "test"

    def _get_system_prompt(self):
        return "test"


class _Qualitative:
    """A claim naming no value: nothing for the numeric guard to compare."""
    metric = "revenue"
    value = None
    operator = None
    claim_type = "news"


class _Numeric:
    metric = "revenue"
    value = 391_035_000_000.0
    operator = "eq"
    claim_type = "sec"


def _records(*calls):
    """Build tool records the way the agent does, from real messages.

    The records come out of the production extractor rather than being
    hand-assembled, so this asserts the shape the system emits.
    """
    messages = []
    for i, (name, result) in enumerate(calls):
        call_id = f"call_{i}"
        messages.append(
            AIMessage(content="", tool_calls=[
                {"name": name, "args": {"query": "theme park"}, "id": call_id}]))
        messages.append(
            ToolMessage(content=str(result), tool_call_id=call_id, name=name))
    _, _, _, records = BaseVerificationAgent._extract_tool_info(
        BaseVerificationAgent, messages)
    return records


def _news(count, success=True, error=None):
    return {"success": success, "query": "theme park",
            "articles": [{"title": f"a{i}"} for i in range(count)],
            "count": count, "error": error}


def _filing(count, success=True, reason=None, error=None):
    return {"success": success,
            "chunks": [{"chunk_text": f"c{i}"} for i in range(count)],
            "total_found": count, "reason": reason, "error": error}


class TestTheGapIsClassified:
    """Why a claim has nothing behind it, distinguished rather than collapsed.

    A reader deciding whether to re-run needs to know if the source was silent
    or simply unreachable; those have opposite remedies.
    """

    def test_a_search_that_returned_nothing_is_silence(self):
        gap = qualitative_evidence_gap(_records(
            ("search_financial_news", _news(0))))
        assert gap == "no_supporting_evidence"

    def test_no_retrieval_at_all_is_its_own_gap(self):
        assert qualitative_evidence_gap(_records()) == "no_retrieval_attempted"

    def test_memory_alone_is_not_retrieval(self):
        """A prior verification is this system's own output. Counting it would
        let a verdict cite itself."""
        gap = qualitative_evidence_gap(_records(
            ("search_past_verifications", {"success": True,
                                           "matches": [{"verdict": "REFUTES"}]})))
        assert gap == "no_retrieval_attempted"

    def test_a_failed_search_is_not_silence(self):
        """A failure to reach the source is not the source failing to speak."""
        gap = qualitative_evidence_gap(_records(
            ("search_financial_news", _news(0, success=False, error="timeout"))))
        assert gap == "retrieval_unavailable"

    def test_an_unindexed_issuer_is_not_silence_either(self):
        """There was no filing to be silent."""
        gap = qualitative_evidence_gap(_records(
            ("search_filing_text", _filing(0, reason="no_corpus"))))
        assert gap == "no_indexed_source"

    def test_retrieved_articles_close_the_gap(self):
        assert qualitative_evidence_gap(_records(
            ("search_financial_news", _news(2)))) is None

    def test_retrieved_filing_chunks_close_the_gap(self):
        assert qualitative_evidence_gap(_records(
            ("search_filing_text", _filing(3)))) is None

    def test_one_productive_search_among_many_empty_ones_is_enough(self):
        """The live failure called the tool four times. One real hit is
        evidence; four empty calls are not four times as much nothing."""
        assert qualitative_evidence_gap(_records(
            ("search_financial_news", _news(0)),
            ("search_financial_news", _news(0)),
            ("search_financial_news", _news(1)),
        )) is None


class TestADecisiveVerdictRequiresRetrievedEvidence:
    """Driven through `_apply_override`, the component that overrules the
    model. Testing a wrapper would pass whether or not this was fixed."""

    def _override(self, verdict, records, claim=_Qualitative, confidence=0.95):
        output = VerdictOutput(verdict=verdict, confidence=confidence,
                               reasoning="r")
        agent = _Agent.__new__(_Agent)
        agent.agent_type = "news"
        return agent._apply_override(
            output, {"parsed_claim": claim()}, None,
            evidence_gap=qualitative_evidence_gap(records))

    @pytest.mark.parametrize("verdict", ["REFUTES", "SUPPORTS"])
    def test_silence_cannot_decide_a_claim_either_way(self, verdict):
        """The pinned defect. Absence supports nothing and refutes nothing."""
        final, confidence, _ = self._override(
            verdict, _records(("search_financial_news", _news(0))))

        assert final == "NOT_ENOUGH_INFO", (
            f"a {verdict} at 0.95 was released on zero retrieved articles")
        assert confidence <= 0.5

    def test_the_theme_park_claim_specifically(self):
        """4/4 live runs: four successful searches, zero articles, REFUTES
        0.95. The exact shape that reached a user."""
        final, confidence, _ = self._override("REFUTES", _records(
            ("search_financial_news", _news(0)),
            ("search_financial_news", _news(0)),
            ("search_past_verifications", {"success": True, "matches": []}),
            ("search_financial_news", _news(0)),
        ))

        assert final == "NOT_ENOUGH_INFO"
        assert confidence <= 0.5

    def test_a_verdict_backed_by_articles_stands(self):
        """The count gate asks whether anything was retrieved, never whether
        the model read it correctly. Real articles, verdict untouched.

        Asserted with SUPPORTS rather than REFUTES: this test first pinned a
        REFUTES here, written while the live defect was still believed to be
        an empty result set. It is not -- the search returns ten irrelevant
        articles -- so a qualitative REFUTES is now declined by the separate
        rule in TestNonCorroborationIsNotContradiction whatever the count.
        Keeping the old assertion would have tested the count gate through a
        path that no longer reaches it."""
        final, confidence, _ = self._override(
            "SUPPORTS", _records(("search_financial_news", _news(3))))

        assert final == "SUPPORTS"
        assert confidence == 0.95

    def test_a_verdict_backed_by_filing_chunks_stands(self):
        final, _, _ = self._override(
            "SUPPORTS", _records(("search_filing_text", _filing(2))))

        assert final == "SUPPORTS"

    def test_an_unavailable_source_also_fails_closed(self):
        final, _, _ = self._override(
            "REFUTES", _records(
                ("search_financial_news", _news(0, success=False, error="503"))))

        assert final == "NOT_ENOUGH_INFO"

    def test_an_already_undecided_verdict_is_left_alone(self):
        """Nothing to overrule; the gate must not manufacture a change."""
        final, _, _ = self._override(
            "NOT_ENOUGH_INFO", _records(("search_financial_news", _news(0))),
            confidence=0.4)

        assert final == "NOT_ENOUGH_INFO"


class TestNonCorroborationIsNotContradiction:
    """The second half of the defect, and the one the count gate cannot reach.

    `search_financial_news("Apple theme park Ohio")` returns **ten** articles:
    "Move Over, Kids, It's Grown-Up Time", "Ohio Rich in New Travel Experiences
    for 2014", "apple hospitality reit, inc." Retrieval succeeded and the count
    gate stands aside, correctly -- something was retrieved. The model then
    reads ten irrelevant articles, finds nothing about a theme park, and
    returns REFUTES at 0.95.

    So the failure is not an empty result set. It is that "I searched and could
    not confirm this" is being reported as "this is false". No count can
    distinguish an article that contradicts a claim from one that merely fails
    to mention it, and nothing deterministic can read relevance out of prose.

    Release A therefore declines the verdict rather than guessing at it: a
    claim naming no value is not refuted on retrieval evidence alone. Refuting
    a qualitative claim needs a source that contradicts it, and the only such
    signal in the pipeline is an A2A CONTRADICTS, which escalates to a person
    anyway. This is the same rule D9 already applies to filing silence, which
    does not become weaker because it arrived from the news route.

    SUPPORTS is treated differently on purpose. Confirming a claim requires
    having found text that asserts it, which retrieval did supply; refuting one
    on absence is the classic fallacy and the one observed in production.
    """

    def _override(self, verdict, records, confidence=0.95):
        output = VerdictOutput(verdict=verdict, confidence=confidence,
                               reasoning="r")
        agent = _Agent.__new__(_Agent)
        agent.agent_type = "news"
        return agent._apply_override(
            output, {"parsed_claim": _Qualitative()}, None,
            evidence_gap=qualitative_evidence_gap(records))

    def test_ten_irrelevant_articles_do_not_refute(self):
        """The live shape, exactly: retrieval succeeded, count is 10, and the
        model still may not call the claim false."""
        final, confidence, _ = self._override(
            "REFUTES", _records(("search_financial_news", _news(10))))

        assert final == "NOT_ENOUGH_INFO", (
            "non-corroboration was released as contradiction")
        assert confidence <= 0.5

    def test_a_qualitative_claim_can_still_be_supported(self):
        """The narrowing is asymmetric and must stay that way, or the news
        route stops producing any verdict at all."""
        final, confidence, _ = self._override(
            "SUPPORTS", _records(("search_financial_news", _news(10))))

        assert final == "SUPPORTS"
        assert confidence == 0.95

    def test_filing_text_cannot_refute_a_qualitative_claim_either(self):
        """Same rule, other route. D9 already says filing silence is not
        contradiction; retrieved chunks that fail to mention the claim are
        that same silence."""
        final, _, _ = self._override(
            "REFUTES", _records(("search_filing_text", _filing(4))))

        assert final == "NOT_ENOUGH_INFO"


class TestTheDeclineIsDeclaredRatherThanQueued:
    """A declined qualitative verdict terminates with a reason.

    Downgrading to NOT_ENOUGH_INFO at low confidence is not enough on its own:
    low confidence trips `low_confidence` in output_guardrails, and the claim
    goes to a human as PENDING. That is the mistake the publication report
    already names in its own words -- "the same mistake as escalating filing
    silence ... it fills a person's queue with items they cannot act on" --
    because a reviewer opening this claim sees exactly the nothing FinVet saw.

    So the decline carries a machine-readable limitation, which suppresses the
    escalation the same way `unsupported_q4_derivation` does, and the caller is
    told why rather than being handed a queue position.
    """

    def _reason(self, claimed_value, verdict, gap):
        from finvet.models.evidence import qualitative_decline_reason

        return qualitative_decline_reason(claimed_value, verdict, gap)

    def test_non_corroboration_has_its_own_reason(self):
        assert self._reason(None, "REFUTES", None) == \
            "non_corroboration_is_not_contradiction"

    def test_an_empty_search_reports_the_gap_itself(self):
        assert self._reason(None, "REFUTES", "no_supporting_evidence") == \
            "no_supporting_evidence"

    def test_a_standing_verdict_has_no_reason(self):
        assert self._reason(None, "SUPPORTS", None) is None

    def test_a_numeric_claim_has_no_reason(self):
        assert self._reason(391_035_000_000.0, "REFUTES", None) is None

    def test_the_limitation_suppresses_the_escalation(self):
        """Driven through the real guardrail node, which is what decides."""
        import importlib

        node = importlib.import_module("finvet.graph.nodes.output_guardrails")
        result = node.output_guardrails({
            "request_id": "req_gate",
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.3,
            "agent_evidence": {
                "verdict": "NOT_ENOUGH_INFO",
                "confidence": 0.3,
                "limitation": "non_corroboration_is_not_contradiction",
                "reasoning": "declined",
            },
        })

        assert not result.get("hitl_required"), (
            "a declared decline was queued for a reviewer who sees the same "
            "nothing the system saw")
        assert "low_confidence" not in (result.get("hitl_triggers") or [])


class TestTheNumericPathIsUnaffected:
    """A claim naming a value is governed by the trusted observation, not by
    retrieval counts. These two guards must not start overlapping."""

    def _override(self, verdict, records):
        from finvet.models.evidence import TrustedObservation

        observation = TrustedObservation(
            tool="get_income_statement", metric="revenue",
            concept="Revenues", value=391_035_000_000.0,
            units="USD", period_end="2024-09-28")
        output = VerdictOutput(verdict=verdict, confidence=0.95, reasoning="r")
        agent = _Agent.__new__(_Agent)
        agent.agent_type = "sec"
        return agent._apply_override(
            output, {"parsed_claim": _Numeric()}, observation,
            evidence_gap=qualitative_evidence_gap(records))

    def test_a_numeric_claim_is_not_gated_on_retrieval_counts(self):
        """XBRL facts do not arrive through a retrieval tool, so an empty
        filing search must not veto a verdict the observation settled."""
        final, _, _ = self._override("SUPPORTS", _records())

        assert final == "SUPPORTS"
