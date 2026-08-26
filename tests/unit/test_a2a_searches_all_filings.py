"""A corroboration asks whether the issuer disclosed this, not when.

The News -> SEC delegation resolved a period and scoped retrieval to it, on the
reasoning that "the nested agent targets the same filing the parent would
have". For a corroboration that is the wrong question. It does not ask what the
FY2025 filing said; it asks whether the issuer has disclosed the matter at all.

Live, the cost was total. "Apple was fined by the European Commission over App
Store practices" resolved to 2025-12-31, and Apple's newest indexed filings
close 2025-12-27 and 2025-09-27 -- both earlier. Every search returned zero, the
agent reworded and retried, and died at its recursion limit of 7 with no
provenance. The same claim routed to the SEC agent directly, where no period is
resolved, found evidence on the first try.

Temporal eligibility is already handled, and better, one level up:
`_filing_could_cover` compares the event date against the newest filing on
record and returns NOT_APPLICABLE_YET when no filing could carry it. That is
the honest check. Scoping retrieval on top of it does not add rigour -- it
removes the filings that actually carry the disclosure.

Safe for the same reason D17 is: CORROBORATION_METRICS is {fine_amount,
settlement_amount}, never a GAAP figure, so nothing numeric depends on the
nested run's period targeting. Each chunk carries its own period_end, so a
reader sees which filing answered.
"""

from unittest.mock import patch



class TestTheDelegationDoesNotScopeRetrieval:

    def _run(self, scope_retrieval):
        from finvet.graph.nodes import domain_agents

        seen = {}

        class _P:
            period_type = "annual"
            end_date = "2025-12-31"
            start_date = "2025-01-01"

        def _fake_run_agent(*args, **kwargs):
            from finvet.tools.sec_tools import _current_period_target
            seen["target"] = _current_period_target()
            return {"agent_evidence": {"verdict": "NOT_ENOUGH_INFO"}}

        with patch.object(domain_agents, "_run_agent", _fake_run_agent):
            domain_agents.run_sec_agent_scoped(
                {"canonical_period": _P()}, scope_retrieval=scope_retrieval)
        return seen["target"]

    def test_the_normal_sec_route_still_targets_its_period(self):
        """Unchanged. A numeric SEC claim must read the period it names."""
        assert self._run(scope_retrieval=True) == ("2025-12-31", "annual")

    def test_a_delegation_searches_every_filing(self):
        """The fix: no period target, so no filing is excluded."""
        assert self._run(scope_retrieval=False) == (None, None)


class TestTemporalEligibilityIsStillEnforced:
    """Removing the scope must not remove the honest 'no filing covers this
    yet' answer — that check lives above retrieval and stays."""

    def test_an_event_after_every_filing_is_not_applicable_yet(self):
        from finvet.tools.corroborate_sec import _filing_could_cover

        assert _filing_could_cover("2026-06-01", "2025-09-27") is False

    def test_an_event_before_the_newest_filing_is_eligible(self):
        from finvet.tools.corroborate_sec import _filing_could_cover

        assert _filing_could_cover("2024-03-25", "2025-09-27") is True

    def test_corroboration_never_covers_a_gaap_metric(self):
        """Why unscoping the nested run cannot affect a numeric verdict."""
        from finvet.config.constants import CORROBORATION_METRICS
        from finvet.config.metrics import SERVABLE_METRICS

        assert not (CORROBORATION_METRICS & SERVABLE_METRICS["sec"])
