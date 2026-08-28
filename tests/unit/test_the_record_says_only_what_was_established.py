"""Three places the record said more than the run established.

Each was measured on a live claim before being written here.

**1. A decline with no reason queues a reviewer who cannot help.**

    Amazon was fined 746 million euros ...
      -> PENDING, confidence 0.0, limitation=None, triggers=['low_confidence']

`fine_amount` has no XBRL concept. Where Python can lift the figure out of a
Legal Proceedings passage the claim resolves, but where the filing does not
carry it there is nothing to certify and no run will ever change that. The
guard is right to decline; it just says nothing, so `declined_with_reason`
never fires and a person is queued for a claim nobody can act on.

Most narrative metrics already avoid this -- `_unsupported_claim` catches
acquisition_value, price_target and the macro set and states
`unsupported_metric`. Only fine/settlement claims naming a figure fall through,
because they route to "news_search" rather than "unsupported".

**2. A gate that asserts a fact it never checked.**

    "No filing on record covers {date}; the most recent period closed earlier"

`_latest_period_end` returns the *claim's* resolved period end. It never asks
the corpus what the newest filing is, so the sentence describes a lookup that
did not happen.

**3. A figure attributed to a claim that never made one.**

    "Apple was fined by the European Commission over its App Store practices"
      -> a2a record carries claimed_value = 500,000,000

The claim names no amount. The model supplied that number as a tool argument,
and it was recorded as the claim's own.
"""

import pytest


class _Claim:
    claim_type = "news"
    metric = "fine_amount"
    ticker = "AAPL"
    operator = "eq"
    period = None

    def __init__(self, value=500_000_000.0):
        self.value = value


class TestAnUncertifiableAmountSaysSo:

    def test_a_fine_claim_with_no_certifiable_figure_is_explained(self):
        from finvet.models.evidence import uncertifiable_amount_reason

        assert uncertifiable_amount_reason(500_000_000.0, "news_search") == \
            "amount_not_certifiable"

    def test_a_claim_naming_no_figure_is_not_covered(self):
        """Nothing was claimed, so nothing failed to be certified. That case
        belongs to the qualitative decline, which judges it separately."""
        from finvet.models.evidence import uncertifiable_amount_reason

        assert uncertifiable_amount_reason(None, "news_search") is None

    @pytest.mark.parametrize("strategy", ["xbrl", "market", "macro"])
    def test_a_servable_metric_is_never_covered(self, strategy):
        """The load-bearing case. A metric a tool *does* serve that missed this
        time is a situational failure a reviewer can act on -- Nvidia's revenue
        was exactly that. It must keep escalating."""
        from finvet.models.evidence import uncertifiable_amount_reason

        assert uncertifiable_amount_reason(130_000_000_000.0, strategy) is None

    def test_the_two_decline_reasons_do_not_overlap(self):
        """`qualitative_decline_reason` returns None for any claim carrying a
        value, by design -- 'the two guards must not start overlapping'."""
        from finvet.models.evidence import (qualitative_decline_reason,
                                            uncertifiable_amount_reason)

        assert qualitative_decline_reason(500_000_000.0, "REFUTES", None) is None
        assert uncertifiable_amount_reason(None, "news_search") is None

    def test_the_reason_suppresses_the_reviewer_queue(self):
        """Its whole purpose: output_guardrails already skips low-confidence
        escalation when a limitation explains the decline."""
        from finvet.graph.nodes.output_guardrails import output_guardrails

        state = {"request_id": "r", "claim_raw": "c",
                 "verdict": "NOT_ENOUGH_INFO", "confidence": 0.5,
                 "agent_evidence": {"verdict": "NOT_ENOUGH_INFO",
                                    "confidence": 0.5, "reasoning": "",
                                    "limitation": "amount_not_certifiable"}}

        assert "low_confidence" not in (output_guardrails(state).get(
            "hitl_triggers") or [])


class TestTheTemporalGateOnlyClaimsWhatItChecked:

    def test_it_reports_the_period_it_actually_compared(self):
        """Either name the corpus or do not invoke it. The old text said 'no
        filing on record covers X' after comparing against the claim's own
        resolved period."""
        import inspect

        from finvet.tools import corroborate_sec

        source = inspect.getsource(corroborate_sec)
        assert "No filing on record covers" not in source, (
            "the gate still asserts a corpus fact it never queried")

    def test_the_helper_name_matches_what_it_returns(self):
        from finvet.tools.corroborate_sec import _claim_period_end

        class _CP:
            end_date = "2024-12-31"

        assert _claim_period_end({"canonical_period": _CP()}) == "2024-12-31"
        assert _claim_period_end({}) is None


class TestOnlyTheClaimsOwnFigureIsRecorded:

    def _lift(self, monkeypatch, claim_value, tool_claimed_value):
        from finvet.graph.nodes import domain_agents as node

        a2a = {"success": True, "status": "PENDING_CLASSIFICATION",
               "verdict": "NOT_ENOUGH_INFO", "confidence": 0.5,
               "claimed_value": tool_claimed_value, "retrieved_value": None,
               "provenance": []}
        evidence = {"agent": "news", "verdict": "NOT_ENOUGH_INFO",
                    "llm_original_verdict": "SUPPORTS", "confidence": 0.5,
                    "provenance": [{"tool": "corroborate_with_filing",
                                    "args": {}, "result": a2a}]}
        monkeypatch.setattr(node, "_run_agent",
                            lambda *a, **k: {"agent_evidence": evidence})
        monkeypatch.setattr(node, "_unsupported_claim", lambda s: None)

        out = node.run_news_agent({"parsed_claim": _Claim(claim_value),
                                   "request_id": "r", "claim_raw": "c"})
        return out.get("corroboration_result") or {}

    def test_a_model_supplied_figure_is_not_attributed_to_the_claim(self, monkeypatch):
        corr = self._lift(monkeypatch, claim_value=None,
                          tool_claimed_value=500_000_000.0)

        assert corr.get("claimed_value") is None, (
            "a figure the claim never stated was recorded as its own")

    def test_the_claims_real_figure_is_kept(self, monkeypatch):
        corr = self._lift(monkeypatch, claim_value=1_000_000_000_000.0,
                          tool_claimed_value=1_000_000_000_000.0)

        assert corr.get("claimed_value") == 1_000_000_000_000.0
