"""The override is the thesis, and nothing drove it end to end.

`_apply_override` had thorough arithmetic tests, and every downstream reader of
`override_applied` was tested against a hand-built dict. Between them sat the
path that actually matters -- a model returning one verdict, `execute`
publishing another -- and no test crossed it. `test_base_agent` asserted the
flag was *False*; the True cases were all fixtures the tests wrote themselves.
That is the failure mode CLAUDE.md's sixth gotcha describes, applied to the one
mechanism the README leads with.

The scenario is the real one: Apple's FY2024 10-K carries $294.9B for the
Products segment beside $391.0B consolidated. A model reading prose picks the
segment figure and confidently refutes a true claim. Here the trusted
observation carries the segment number so the comparison genuinely disagrees
with the model, which is what makes the override observable.
"""

import pytest

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.models.evidence import TrustedObservation


class _Agent(BaseVerificationAgent):
    def _get_source_description(self):
        return "SEC EDGAR"

    def _get_system_prompt(self):
        return "test"


class _Claim:
    claim_type = "sec"
    metric = "revenue"
    operator = "eq"
    value = 391_000_000_000.0
    period = "fiscal year 2024"


class _Period:
    period_type = "annual"
    start_date = "2023-10-01"
    end_date = "2024-09-28"
    fiscal_quarter = None


def _observation(value):
    return TrustedObservation(
        tool="get_income_statement", metric="revenue",
        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        value=value, units="USD", period_end="2024-09-28",
        source_id="0000320193-24-000123")


def _run(monkeypatch, *, model_verdict, model_confidence, observed):
    """Drive `execute`, stubbing only the two model calls."""
    agent = _Agent.__new__(_Agent)
    agent.agent_type = "sec"
    agent.max_iterations = 5

    monkeypatch.setattr(agent, "_build_context", lambda state: "ctx",
                        raising=False)
    monkeypatch.setattr(
        agent, "react_agent",
        type("R", (), {"invoke": staticmethod(
            lambda *a, **k: {"messages": []})})(),
        raising=False)
    monkeypatch.setattr(agent, "_extract_tool_info",
                        lambda messages: (["get_income_statement"], [], [], []),
                        raising=False)
    monkeypatch.setattr("finvet.agents.base.resolve_trusted_observation",
                        lambda *a, **k: _observation(observed))
    monkeypatch.setattr(
        agent, "_extract_verdict",
        lambda messages, state: VerdictOutput(
            verdict=model_verdict, confidence=model_confidence,
            reasoning="the filing says so",
            # The model reports the number it read, which is the segment
            # figure. The published value must not be this one.
            retrieved_value=observed if model_verdict != "SUPPORTS" else 391_000_000_000.0,
            source_description="10-K"),
        raising=False)

    return agent.execute({"parsed_claim": _Claim(), "request_id": "r",
                          "canonical_period": _Period()})


class TestPythonOverrulesTheModel:

    def test_a_confident_wrong_model_verdict_is_replaced(self, monkeypatch):
        """The model reads the segment figure and says SUPPORTS. It is not
        the consolidated number, so the comparison refutes."""
        evidence = _run(monkeypatch, model_verdict="SUPPORTS",
                        model_confidence=0.99, observed=294_900_000_000.0)

        assert evidence["verdict"] == "REFUTES"
        assert evidence["override_applied"] is True
        assert evidence["llm_original_verdict"] == "SUPPORTS"

    def test_both_verdicts_are_recorded_not_just_the_winner(self, monkeypatch):
        """The README's claim: when they disagree, both are kept."""
        evidence = _run(monkeypatch, model_verdict="SUPPORTS",
                        model_confidence=0.99, observed=294_900_000_000.0)

        assert evidence["llm_original_verdict"] != evidence["verdict"]

    def test_the_published_number_is_the_one_python_compared(self, monkeypatch):
        """Not the value the model said it found."""
        evidence = _run(monkeypatch, model_verdict="SUPPORTS",
                        model_confidence=0.99, observed=294_900_000_000.0)

        assert evidence["retrieved_value"] == 294_900_000_000.0
        assert evidence["trusted_observation"]["value"] == 294_900_000_000.0
        assert evidence["magnitude_difference_percent"] > 20

    def test_the_corrected_verdict_does_not_inherit_the_model_confidence(
            self, monkeypatch):
        """0.99 described the verdict that was just disproven. The comparison
        decided the answer, so the comparison states the confidence."""
        evidence = _run(monkeypatch, model_verdict="SUPPORTS",
                        model_confidence=0.99, observed=294_900_000_000.0)

        assert evidence["confidence"] == 0.90

    def test_a_borderline_override_is_held_at_the_lower_level(self, monkeypatch):
        """1.19% against a 1.5% tolerance: inside, but not comfortably. The
        comparator's own 0.85 must survive a confidently wrong model."""
        evidence = _run(monkeypatch, model_verdict="REFUTES",
                        model_confidence=0.99, observed=395_650_000_000.0)

        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["override_applied"] is True
        assert evidence["confidence"] == 0.85


class TestAgreementIsUnchanged:

    @pytest.mark.parametrize("model_confidence,expected", [(0.97, 0.97),
                                                           (0.60, 0.90)])
    def test_the_model_keeps_its_confidence_when_it_was_right(
            self, monkeypatch, model_confidence, expected):
        """No override, so the higher of the two stands as before."""
        evidence = _run(monkeypatch, model_verdict="SUPPORTS",
                        model_confidence=model_confidence,
                        observed=391_035_000_000.0)

        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["override_applied"] is False
        assert evidence["confidence"] == expected
