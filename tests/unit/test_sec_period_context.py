"""Tests for threading the resolved period into the SEC tools.

`period_resolver` already computes the exact date range a claim refers to. Before
this, that never reached retrieval: the SEC tools called get_financials without a
period, so a filing's comparative years came back interchangeably.

The period is injected out-of-band rather than handed to the model as a tool
argument. It was resolved deterministically upstream, so letting the LLM
re-transcribe it into a call would add an error channel for no benefit — the same
reasoning behind the Python verdict override.
"""

from unittest.mock import MagicMock

import pytest

from finvet.models.claim import CanonicalPeriod, ParsedClaim
from finvet.tools import sec_tools
from finvet.tools.sec_tools import (
    get_balance_sheet,
    get_cash_flow,
    get_income_statement,
    period_target_for,
    use_period_target,
)


def _sec_claim():
    return ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                       operator="eq", value=391e9, period="fiscal 2024")


def _period(period_type, start="2024-01-01", end="2024-12-31"):
    return CanonicalPeriod(
        period_type=period_type, start_date=start, end_date=end,
        is_assumption=False, assumptions=[], original_mention="x",
    )


class TestPeriodTargetFor:

    @pytest.mark.parametrize("period_type", ["annual", "quarterly", "half_year", "date"])
    def test_datable_periods_yield_a_target(self, period_type):
        assert period_target_for(_period(period_type)) == ("2024-12-31", period_type)

    @pytest.mark.parametrize("period_type", ["current", "event_relative"])
    def test_undatable_periods_yield_nothing(self, period_type):
        """These carry today's date as a placeholder. Targeting XBRL with it
        would match no fact and flag every value unverified."""
        assert period_target_for(_period(period_type)) is None

    def test_missing_period_yields_nothing(self):
        assert period_target_for(None) is None

    def test_period_without_an_end_date_yields_nothing(self):
        assert period_target_for(_period("annual", end="")) is None


class TestUsePeriodTarget:

    def test_context_is_empty_by_default(self):
        assert sec_tools._current_period_target() == (None, None)

    def test_context_applies_inside_the_block(self):
        with use_period_target("2024-06-30", "quarterly"):
            assert sec_tools._current_period_target() == ("2024-06-30", "quarterly")

    def test_context_is_cleared_on_exit(self):
        with use_period_target("2024-06-30", "quarterly"):
            pass
        assert sec_tools._current_period_target() == (None, None)

    def test_context_is_cleared_even_when_the_body_raises(self):
        with pytest.raises(RuntimeError):
            with use_period_target("2024-06-30", "quarterly"):
                raise RuntimeError("agent blew up")
        assert sec_tools._current_period_target() == (None, None)

    def test_none_target_is_a_no_op(self):
        with use_period_target(None, None):
            assert sec_tools._current_period_target() == (None, None)


class TestToolsForwardTheTarget:

    @pytest.fixture
    def client(self):
        fake = MagicMock()
        fake.get_financials.return_value = []
        sec_tools._set_client(fake)
        yield fake
        sec_tools._set_client(None)

    def test_income_statement_forwards_period_end_and_kind(self, client):
        with use_period_target("2024-06-30", "quarterly"):
            get_income_statement.invoke({"cik": "1", "accession_number": "a"})
        kwargs = client.get_financials.call_args.kwargs
        assert kwargs["period_end"] == "2024-06-30"
        assert kwargs["period"] == "quarterly"

    def test_balance_sheet_forwards_the_target(self, client):
        with use_period_target("2024-09-28", "annual"):
            get_balance_sheet.invoke({"cik": "1", "accession_number": "a"})
        kwargs = client.get_financials.call_args.kwargs
        assert kwargs["period_end"] == "2024-09-28"

    def test_cash_flow_forwards_the_target(self, client):
        with use_period_target("2024-12-31", "annual"):
            get_cash_flow.invoke({"cik": "1", "accession_number": "a"})
        kwargs = client.get_financials.call_args.kwargs
        assert kwargs["period_end"] == "2024-12-31"

    def test_resolved_period_overrides_the_models_own_argument(self, client):
        """The model may pass any period it likes; the resolved one wins."""
        with use_period_target("2024-06-30", "quarterly"):
            get_income_statement.invoke(
                {"cik": "1", "accession_number": "a", "period": "annual"}
            )
        assert client.get_financials.call_args.kwargs["period"] == "quarterly"

    def test_without_a_target_the_models_argument_is_used(self, client):
        get_income_statement.invoke(
            {"cik": "1", "accession_number": "a", "period": "annual"}
        )
        kwargs = client.get_financials.call_args.kwargs
        assert kwargs["period_end"] is None
        assert kwargs["period"] == "annual"


class TestSecAgentNodeAppliesTheTarget:
    """run_sec_agent owns the state -> agent boundary, so it applies the period."""

    def test_resolved_period_is_active_while_the_agent_runs(self, monkeypatch):
        seen = {}

        def fake_run_agent(agent_cls, agent_type, source_desc, state, **kwargs):
            seen["target"] = sec_tools._current_period_target()
            return {"agent_evidence": {"provenance": []}, "agent_type": "sec"}

        from finvet.graph.nodes import domain_agents
        monkeypatch.setattr(domain_agents, "_run_agent", fake_run_agent)
        domain_agents.run_sec_agent({
            "request_id": "t", "parsed_claim": _sec_claim(),
            "canonical_period": _period("quarterly", end="2024-06-30"),
        })
        assert seen["target"] == ("2024-06-30", "quarterly")

    def test_target_is_cleared_after_the_node_returns(self, monkeypatch):
        from finvet.graph.nodes import domain_agents
        monkeypatch.setattr(
            domain_agents, "_run_agent",
            lambda *a, **k: {"agent_evidence": {"provenance": []}, "agent_type": "sec"},
        )
        domain_agents.run_sec_agent({
            "request_id": "t", "parsed_claim": _sec_claim(),
            "canonical_period": _period("annual"),
        })
        assert sec_tools._current_period_target() == (None, None)

    def test_undatable_period_leaves_the_target_unset(self, monkeypatch):
        seen = {}
        from finvet.graph.nodes import domain_agents

        def fake_run_agent(*a, **k):
            seen["target"] = sec_tools._current_period_target()
            return {"agent_evidence": {"provenance": []}, "agent_type": "sec"}

        monkeypatch.setattr(domain_agents, "_run_agent", fake_run_agent)
        domain_agents.run_sec_agent({
            "request_id": "t", "parsed_claim": _sec_claim(),
            "canonical_period": _period("event_relative"),
        })
        assert seen["target"] == (None, None)
