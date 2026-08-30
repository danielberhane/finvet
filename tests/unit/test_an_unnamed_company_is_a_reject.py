"""A claim about a company it never names is rejected, not merely undecided.

Golden run id 88: "A large US bank posted $30 billion in net income last year"
parsed as `claim_type="sec"`, `ticker=null`, `reject_reason=null` and was handed
to an agent, which spent 14 tool calls and 150 seconds looking for a company
that was never named.

`_unsupported_claim` now declines it before an agent runs, which removed the
150 seconds and the unactionable review. But the terminal status was still
wrong: the dataset expects **REJECTED** and the run produced NOT_ENOUGH_INFO.
Those are different statements. A decline says "this is a claim and we could not
verify it"; a rejection says "this is not a verifiable claim". Naming no company
is the second.

The prompt already carries the rule, with almost this exact example:

    "ambiguous_entity": the company cannot be identified
                        ("the oil major", "a big bank")

So this is not a prompt gap -- the model ignored a rule it was given. The
guarantee is enforced here instead, deterministically, in the same function that
already reconciles the model's other reject-field inconsistencies. Restating the
rule in the prompt would leave it depending on the model noticing.

Scoped to `sec` and `market`, the two claim types that address a specific
issuer's data: a SEC lookup needs a CIK and a quote needs a symbol, and both
come from the ticker. A news claim legitimately runs without one, because news
search takes a company name rather than an identifier.
"""

import pytest

from finvet.graph.nodes.claim_parser import reconcile_reject_fields


def _parsed(claim_type="sec", ticker=None, **extra):
    data = {"claim_type": claim_type, "ticker": ticker, "metric": "net_income",
            "operator": "eq", "value": 30_000_000_000.0,
            "period": "last year", "reject_reason": None}
    data.update(extra)
    return data


class TestAClaimNamingNoCompanyIsRejected:

    def test_a_sec_claim_without_a_ticker_becomes_a_reject(self):
        out = reconcile_reject_fields(_parsed())

        assert out["claim_type"] == "reject"
        assert out["reject_reason"] == "ambiguous_entity"

    def test_a_market_claim_without_a_ticker_becomes_a_reject(self):
        """A quote needs a symbol for the same reason a filing needs a CIK."""
        out = reconcile_reject_fields(
            _parsed(claim_type="market", metric="closing_price"))

        assert out["claim_type"] == "reject"
        assert out["reject_reason"] == "ambiguous_entity"

    def test_an_empty_ticker_counts_as_absent(self):
        out = reconcile_reject_fields(_parsed(ticker=""))

        assert out["claim_type"] == "reject"

    def test_the_models_raw_output_is_not_mutated(self):
        raw = _parsed()
        reconcile_reject_fields(raw)

        assert raw["claim_type"] == "sec", "the caller's dict was modified"


class TestNothingElseChanges:

    def test_a_named_company_is_untouched(self):
        out = reconcile_reject_fields(_parsed(ticker="MSFT"))

        assert out["claim_type"] == "sec"
        assert out["reject_reason"] is None

    def test_a_news_claim_may_omit_the_ticker(self):
        """News search takes a company name, not an identifier, so a missing
        ticker is not disqualifying there."""
        out = reconcile_reject_fields(
            _parsed(claim_type="news", metric="fine_amount"))

        assert out["claim_type"] == "news"

    def test_an_existing_reject_reason_is_preserved(self):
        """A reason the model gave is more specific than the one inferred here
        and must win."""
        out = reconcile_reject_fields(
            _parsed(reject_reason="advice_seeking"))

        assert out["claim_type"] == "reject"
        assert out["reject_reason"] == "advice_seeking"

    @pytest.mark.parametrize("claim_type", ["sec", "market"])
    def test_an_explicit_reject_keeps_its_own_reason(self, claim_type):
        out = reconcile_reject_fields(
            {"claim_type": "reject", "ticker": None,
             "reject_reason": "future_prediction"})

        assert out["reject_reason"] == "future_prediction"
