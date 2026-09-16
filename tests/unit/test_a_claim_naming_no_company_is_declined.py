"""A claim that names no company cannot be looked up, so no agent should try.

Found by the first golden run, id 88:

    "A large US bank posted $30 billion in net income last year"

    parsed : claim_type=sec  metric=net_income  value=30B
             ticker=null  reject_reason=null
    then   : 14 tool calls -> recursion limit -> NOT_ENOUGH_INFO @ 0.2
             -> low_confidence -> PENDING
    cost   : 150 seconds, the slowest row in the run by 5x

A SEC lookup needs a CIK, which comes from a ticker. "A large US bank" is not a
company, so the claim is unverifiable *in principle* -- no better retrieval, no
larger tool budget and no stronger model changes that. The agent spent its whole
budget hunting a company that was never named, and the reviewer it escalated to
would read the same claim and reach the same conclusion. `output_guardrails`
already names that cost: queueing work nobody can act on teaches people to
ignore the flag.

The rule exists twice already and was applied in neither place that mattered.
The parser's reject vocabulary contains `ambiguous_entity` and did not fire it,
and `_policy_wants_corroboration` (domain_agents.py) declines on a missing
ticker 150 lines away. `_unsupported_claim` -- the pre-flight whose entire job
is declining before an agent runs -- checked Q4 derivation and metric
servability, and never asked whether a company was named.

Enforced here deterministically rather than by improving the parser prompt,
because the guarantee should not depend on the model noticing.
"""

import pytest

from finvet.graph.nodes.domain_agents import _unsupported_claim


class _Claim:
    metric = "net_income"
    operator = "eq"
    value = 30_000_000_000.0
    period = "last year"
    reject_reason = None

    def __init__(self, claim_type="sec", ticker=None):
        self.claim_type = claim_type
        self.ticker = ticker


class TestASecClaimWithoutATickerIsDeclinedUpFront:

    def test_it_is_declined(self):
        declined = _unsupported_claim({"parsed_claim": _Claim()})

        assert declined is not None, (
            "a claim naming no company reached an agent, which then spent its "
            "tool budget looking for one")

    def test_the_reason_is_stated(self):
        declined = _unsupported_claim({"parsed_claim": _Claim()})

        assert declined["limitation"] == "no_company_identified"

    def test_a_stated_reason_keeps_it_out_of_the_review_queue(self):
        """`output_guardrails` suppresses the low-confidence escalation when a
        limitation explains the decline. That is the whole point: nobody can
        act on 'a large US bank'."""
        from finvet.graph.nodes.output_guardrails import output_guardrails

        declined = _unsupported_claim({"parsed_claim": _Claim()})
        out = output_guardrails({
            "request_id": "r", "claim_raw": "A large US bank posted $30 billion",
            "verdict": declined["verdict"], "confidence": declined["confidence"],
            "agent_evidence": declined})

        assert "low_confidence" not in (out.get("hitl_triggers") or [])


class TestNothingElseIsAffected:

    def test_a_sec_claim_with_a_ticker_proceeds(self):
        assert _unsupported_claim({"parsed_claim": _Claim(ticker="AAPL")}) is None

    @pytest.mark.parametrize("claim_type", ["news", "market"])
    def test_only_sec_claims_are_gated(self, claim_type):
        """A news claim legitimately reaches an agent without a ticker -- the
        news search takes a company name, not a CIK -- and `market` claims are
        already caught by the metric check when they cannot be served."""
        claim = _Claim(claim_type=claim_type, ticker=None)
        claim.metric = "fine_amount" if claim_type == "news" else "closing_price"

        declined = _unsupported_claim({"parsed_claim": claim})

        if declined is not None:
            assert declined["limitation"] != "no_company_identified"

    def test_an_empty_ticker_counts_as_absent(self):
        assert _unsupported_claim(
            {"parsed_claim": _Claim(ticker="")}) is not None

    def test_no_parsed_claim_is_declined_too(self):
        """This used to assert a pass-through: no parse, let the agent run.
        Reversed on 2026-09-16 -- an agent with nothing to look up spends its
        budget and ends in NOT_ENOUGH_INFO anyway, so the pre-flight declines
        it like the other known "cannot"s. See
        test_a_missing_parse_is_declined_not_run.py, which drives the nodes."""
        declined = _unsupported_claim({"parsed_claim": None})
        assert declined is not None
        assert declined["limitation"] == "no_parsed_claim"
