"""A period the resolver could not determine is not a period to compare against.

When a claim names no period, `period_resolver` returns `period_type="current"`
with `start_date == end_date == today` — a placeholder meaning "we did not
know", not a reporting window. `sec_tools.py:48-51` already guards against
using it, and says why:

    "current" and "event_relative" carry today's date as a placeholder --
    targeting XBRL with it would match nothing and flag every value unverified.

`agents/base.py` passed exactly those dates into `resolve_trusted_observation`
with no such gate, so the warning came true one layer over. Measured against a
real Apple FY2024 fact:

    bounds = 2026-08-26 .. 2026-08-26   ->  trusted observation: None
    no period constraint                ->  ('Revenues', 391,035,000,000)

Every numeric SEC claim naming no period was therefore forced to
NOT_ENOUGH_INFO by a date that means "unknown".

The fix is not simply to drop the bounds, because the same placeholder is also
the fallback for a period that was named but could not be parsed — and there
`canonical_period is not None` made `temporal_status` report "resolved" when
nothing had been. Dropping the bounds alone would let such a claim be settled
by any period's figure while still claiming its period was resolved. So the
placeholder decides both:

    claim names no period      -> no bounds, "not_period_bound", fact resolves
    claim names a real period  -> that window, "resolved", unchanged
    claim names an unparseable
      period                   -> no bounds, "unresolved_period", fails closed
"""


import pytest

from finvet.agents.base import BaseVerificationAgent, VerdictOutput
from finvet.models.claim import CanonicalPeriod
from finvet.models.evidence import TrustedObservation

TODAY = "2026-08-26"

OBSERVATION = {"tool": "get_income_statement", "metric": "revenue",
               "concept": "Revenues", "value": 391_035_000_000.0,
               "units": "USD", "period_end": "2024-09-28"}


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
    range_min = None
    range_max = None

    def __init__(self, period):
        self.period = period


def _placeholder():
    """What the resolver returns when it could not determine a period."""
    return CanonicalPeriod(
        period_type="current", start_date=TODAY, end_date=TODAY,
        is_assumption=True,
        assumptions=["No specific period provided, using current date"])


def _run(monkeypatch, claim_period, canonical):
    """Drive the real `execute`, capturing what reached the resolver."""
    seen = {}

    def _capture(parsed_claim, records, expected_period_end=None,
                 expected_period_start=None, **_kwargs):
        seen["end"] = expected_period_end
        seen["start"] = expected_period_start
        return TrustedObservation(**OBSERVATION)

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
    monkeypatch.setattr(agent, "_extract_verdict",
                        lambda m, s: VerdictOutput(verdict="SUPPORTS",
                                                   confidence=0.9,
                                                   reasoning="r"),
                        raising=False)
    monkeypatch.setattr("finvet.agents.base.resolve_trusted_observation",
                        _capture)

    evidence = agent.execute({"parsed_claim": _Claim(claim_period),
                              "canonical_period": canonical,
                              "request_id": "r"})
    return evidence, seen


class TestAPlaceholderNeverBecomesAPeriodBound:

    def test_a_claim_naming_no_period_is_not_bounded_by_today(self, monkeypatch):
        """The proven defect. These bounds rejected a real filed fact."""
        _, seen = _run(monkeypatch, None, _placeholder())

        assert seen["end"] is None, (
            f"today's placeholder ({TODAY}) was used as a period bound")
        assert seen["start"] is None

    def test_such_a_claim_still_reaches_a_verdict(self, monkeypatch):
        evidence, _ = _run(monkeypatch, None, _placeholder())

        assert evidence["verdict"] == "SUPPORTS"
        assert evidence["trusted_observation"] is not None

    def test_it_is_recorded_as_not_period_bound(self, monkeypatch):
        evidence, _ = _run(monkeypatch, None, _placeholder())

        assert evidence["temporal_status"] == "not_period_bound"

    def test_an_event_relative_placeholder_is_treated_the_same(self, monkeypatch):
        placeholder = CanonicalPeriod(
            period_type="event_relative", start_date=TODAY, end_date=TODAY,
            is_assumption=True)
        _, seen = _run(monkeypatch, None, placeholder)

        assert seen["end"] is None


class TestAnUnparseablePeriodStillFailsClosed:
    """The same placeholder is the fallback for a period that *was* named. It
    must not become licence to compare against any period's figure."""

    def test_it_is_not_reported_as_resolved(self, monkeypatch):
        evidence, _ = _run(monkeypatch, "sometime last year", _placeholder())

        assert evidence["temporal_status"] == "unresolved_period", (
            "a period that fell back to a placeholder was called resolved")

    def test_the_verdict_declines(self, monkeypatch):
        evidence, _ = _run(monkeypatch, "sometime last year", _placeholder())

        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["trusted_observation"] is None


class TestARealPeriodIsUnaffected:

    def test_its_window_is_still_applied(self, monkeypatch):
        real = CanonicalPeriod(period_type="annual",
                               start_date="2023-10-01", end_date="2024-09-28")
        _, seen = _run(monkeypatch, "fiscal year 2024", real)

        assert seen["end"] == "2024-09-28"
        assert seen["start"] == "2023-10-01"

    def test_it_is_still_reported_as_resolved(self, monkeypatch):
        real = CanonicalPeriod(period_type="annual",
                               start_date="2023-10-01", end_date="2024-09-28")
        evidence, _ = _run(monkeypatch, "fiscal year 2024", real)

        assert evidence["temporal_status"] == "resolved"

    @pytest.mark.parametrize("period_type,start,end", [
        ("quarterly", "2024-07-01", "2024-09-30"),
        ("half_year", "2024-01-01", "2024-06-30"),
        ("date", "2024-03-15", "2024-03-15"),
    ])
    def test_every_datable_type_keeps_its_bounds(self, monkeypatch,
                                                 period_type, start, end):
        real = CanonicalPeriod(period_type=period_type,
                               start_date=start, end_date=end)
        _, seen = _run(monkeypatch, "some period", real)

        assert seen["end"] == end


class TestTheGateMatchesTheOneAlreadyInTheCodebase:
    """`sec_tools` drew this line first; there should not be two answers to
    the same question."""

    def test_the_placeholder_types_are_the_ones_sec_tools_excludes(self):
        from finvet.tools.sec_tools import _DATABLE_PERIOD_TYPES

        assert "current" not in _DATABLE_PERIOD_TYPES
        assert "event_relative" not in _DATABLE_PERIOD_TYPES

    def test_every_type_the_resolver_emits_is_accounted_for(self):
        """A type the resolver can produce that is in neither set would be
        silently treated as a placeholder."""
        from finvet.tools.sec_tools import _DATABLE_PERIOD_TYPES

        emitted = {"date", "half_year", "quarterly", "annual",
                   "current", "event_relative"}
        placeholders = {"current", "event_relative"}

        assert emitted - _DATABLE_PERIOD_TYPES == placeholders
