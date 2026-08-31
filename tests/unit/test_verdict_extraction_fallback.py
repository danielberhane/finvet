"""A filed fact is not discarded because the verdict LLM misbehaved.

`Apple's shareholders equity was less than $100 billion in fiscal year 2024`
returned PENDING in 4 of 14 runs. The retrieval worked every time: XBRL
returned StockholdersEquity = $56.95B. What failed was the *second* LLM call,
the one that extracts a structured verdict, which ran to its token limit.

`execute` called `_extract_verdict` before `resolve_trusted_observation` and
returned `_error_evidence` from the exception handler, so the trusted
observation was never resolved and Python never compared 56.95B < 100B. The
claim went to a human for want of a comparison the system could have done
without any model at all.

That is the architecture inverted. The deterministic layer exists to overrule
the model; here a model failure discarded the deterministic layer's only input.
The comparison needs nothing from the LLM: it needs the claimed value, the
operator, and a trusted observation, all of which were already in hand.

So the observation is resolved *first*, and a verdict-extraction failure falls
back to the pure comparator. The fallback is not a rescue of the model's
opinion -- there is no opinion to rescue. It is Python answering the question
on the evidence, and saying so in its own words.

Fails closed where it must: no observation, or no claimed value, and the run
still returns NOT_ENOUGH_INFO.
"""


from finvet.agents.base import BaseVerificationAgent
from finvet.models.evidence import TrustedObservation


class _Agent(BaseVerificationAgent):
    def _get_source_description(self):
        return "SEC EDGAR"

    def _get_system_prompt(self):
        return "test"


class _Claim:
    claim_type = "sec"
    metric = "shareholders_equity"
    operator = "lt"
    value = 100_000_000_000.0
    period = "fiscal year 2024"


def _observation(value=56_950_000_000.0):
    return TrustedObservation(
        tool="get_income_statement", metric="shareholders_equity",
        concept="StockholdersEquity", value=value, units="USD",
        period_end="2024-09-28")


class _Period:
    """A resolved period. Without one the temporal guard nulls the
    observation, which is correct production behaviour and not what these
    tests are about.

    `period_type` is required on the real CanonicalPeriod and was omitted here;
    once a placeholder period stopped being used as a bound, a stub without it
    was correctly read as a placeholder. The stub was incomplete, not the gate.
    """
    period_type = "annual"
    start_date = "2023-10-01"
    end_date = "2024-09-28"
    fiscal_quarter = None


def _state():
    return {"parsed_claim": _Claim(), "request_id": "r",
            "canonical_period": _Period()}


def _agent():
    agent = _Agent.__new__(_Agent)
    agent.agent_type = "sec"
    return agent


class TestTheComparatorNeedsNoModel:
    """`compare_observation` is the whole numeric decision, extracted so both
    the override and the fallback run identical logic."""

    def _compare(self, claim, observation):
        from finvet.agents.base import compare_observation

        return compare_observation(claim, observation)

    def test_the_live_failure_is_decided_deterministically(self):
        """56.95B < 100B. No LLM required, and none consulted."""
        verdict, confidence, diff = self._compare(_Claim(), _observation())

        assert verdict == "SUPPORTS"
        assert confidence >= 0.90
        assert diff is not None

    def test_a_false_inequality_is_refuted(self):
        claim = _Claim()
        claim.value = 10_000_000_000.0     # 56.95B < 10B is false
        verdict, _, _ = self._compare(claim, _observation())

        assert verdict == "REFUTES"

    def test_no_observation_fails_closed(self):
        verdict, confidence, _ = self._compare(_Claim(), None)

        assert verdict == "NOT_ENOUGH_INFO"
        assert confidence <= 0.5

    def test_no_claimed_value_fails_closed(self):
        """Nothing to compare against; the comparator has no opinion."""
        claim = _Claim()
        claim.value = None
        verdict, _, _ = self._compare(claim, _observation())

        assert verdict == "NOT_ENOUGH_INFO"

    def test_an_uninterpretable_operator_fails_closed(self):
        claim = _Claim()
        claim.operator = "roughly_around"
        verdict, _, _ = self._compare(claim, _observation())

        assert verdict == "NOT_ENOUGH_INFO"


class TestAVerdictFailureDoesNotDiscardTheFact:
    """Driven through `execute`, which is where the ordering bug lived."""

    def _run(self, monkeypatch, *, observation, verdict_raises=True):
        agent = _agent()
        agent.max_iterations = 5

        monkeypatch.setattr(agent, "_build_context", lambda state: "ctx",
                            raising=False)
        monkeypatch.setattr(
            agent, "react_agent",
            type("R", (), {"invoke": staticmethod(lambda *a, **k: {"messages": []})})(),
            raising=False)
        monkeypatch.setattr(
            agent, "_extract_tool_info",
            lambda messages: ([], [], [], []), raising=False)
        monkeypatch.setattr(
            "finvet.agents.base.resolve_trusted_observation",
            lambda *a, **k: observation)

        def _boom(messages, state):
            raise RuntimeError(
                "maximum context length 8192 tokens exceeded")

        if verdict_raises:
            monkeypatch.setattr(agent, "_extract_verdict", _boom, raising=False)

        return agent.execute(_state())

    def test_the_deterministic_verdict_is_produced_anyway(self, monkeypatch):
        """The regression, pinned. This returned NOT_ENOUGH_INFO."""
        evidence = self._run(monkeypatch, observation=_observation())

        assert evidence["verdict"] == "SUPPORTS", (
            "a filed fact was discarded because the verdict LLM failed")
        assert evidence["retrieved_value"] == 56_950_000_000.0

    def test_the_observation_reaches_the_evidence(self, monkeypatch):
        evidence = self._run(monkeypatch, observation=_observation())
        observation = evidence["trusted_observation"]

        assert observation["concept"] == "StockholdersEquity"
        assert observation["period_end"] == "2024-09-28"

    def test_the_run_is_not_reported_as_a_clean_model_verdict(self,
                                                              monkeypatch):
        """A reader must be able to tell Python decided this, and that the
        model contributed nothing."""
        evidence = self._run(monkeypatch, observation=_observation())

        assert evidence["llm_original_verdict"] is None
        assert evidence["execution_status"] == "completed"
        assert evidence.get("verdict_source") == "deterministic_fallback"

    def test_the_explanation_is_pythons_own(self, monkeypatch):
        """Not a templated sentence that reads like model output."""
        evidence = self._run(monkeypatch, observation=_observation())
        reasoning = evidence["reasoning"]

        assert "56,950,000,000" in reasoning
        assert "100,000,000,000" in reasoning
        assert "StockholdersEquity" in reasoning

    def test_without_an_observation_it_still_fails_closed(self, monkeypatch):
        """The fallback rescues a comparison, never a verdict. With nothing to
        compare, the run is still an error."""
        evidence = self._run(monkeypatch, observation=None)

        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["execution_status"] == "failed"

    def test_a_healthy_run_is_untouched(self, monkeypatch):
        """The control: when verdict extraction works, nothing changes."""
        agent = _agent()
        agent.max_iterations = 5
        monkeypatch.setattr(agent, "_build_context", lambda state: "ctx",
                            raising=False)
        monkeypatch.setattr(
            agent, "react_agent",
            type("R", (), {"invoke": staticmethod(lambda *a, **k: {"messages": []})})(),
            raising=False)
        monkeypatch.setattr(agent, "_extract_tool_info",
                            lambda messages: ([], [], [], []), raising=False)
        monkeypatch.setattr("finvet.agents.base.resolve_trusted_observation",
                            lambda *a, **k: _observation())

        from finvet.agents.base import VerdictOutput
        monkeypatch.setattr(
            agent, "_extract_verdict",
            lambda messages, state: VerdictOutput(
                verdict="SUPPORTS", confidence=0.8, reasoning="model said so"),
            raising=False)

        evidence = agent.execute(_state())

        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["llm_original_verdict"] == "SUPPORTS"
        assert evidence.get("verdict_source") != "deterministic_fallback"
