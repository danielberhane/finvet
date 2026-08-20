"""Tests for the stage-04 boundary: normalize_parser_output.

One entry point turns raw model JSON into contract-valid data, in an order
that is load-bearing twice over:

- reconcile BEFORE metric resolution, because reconciliation can change
  claim_type and claim_type scopes the whitelist;
- whitelist BEFORE hard-drops inside the resolver, because the source
  vocabulary lists operating_margin in both (dead upstream only because of
  this exact ordering).

Fail closed throughout: a wrong metric produces a confidently wrong verdict,
a null metric reverts to today's agent inference, so every doubtful path
yields null. Decision codes surface to the audit trail so a rising residual
rate is visible in production.
"""


from finvet.graph.nodes.claim_parser import (
    normalize_parser_output,
    resolve_metric_field,
)


def _resolve(claim_type="sec", metric=None, text=""):
    data, decision = resolve_metric_field(
        {"claim_type": claim_type, "metric": metric}, text
    )
    return data["metric"], decision


class TestResolveMetricField:

    def test_absent_metric_is_todays_behaviour(self):
        """The parser does not emit metric yet — every live claim takes this
        path until stage 06, and it must be a clean no-op."""
        assert _resolve(metric=None) == (None, "absent")

    def test_exact_whitelist_hit(self):
        assert _resolve(metric="revenue") == ("revenue", "whitelist")

    def test_alias_is_remapped(self):
        assert _resolve("market", "stock_price") == ("closing_price", "remap")

    def test_normalisation_recovers_near_misses(self):
        assert _resolve("market", "Price-to-Book Ratio") == ("price_to_book", "normalized")

    def test_normalisation_feeds_the_remap_table_too(self):
        """'R and D expense' variants reach canonical form via normalise+remap."""
        assert _resolve("sec", "research_and_development_expense") == (
            "research_and_development", "remap")

    def test_segment_traps_are_dropped_to_null(self):
        assert _resolve("sec", "iphone_revenue") == (None, "hard_drop")

    def test_whitelist_wins_over_the_drop_list(self):
        """operating_margin is in BOTH in the source vocabulary; whitelist-first
        ordering is what keeps it alive. A reordering regression flips this."""
        assert _resolve("sec", "operating_margin") == ("operating_margin", "whitelist")

    def test_unknown_metric_fails_closed(self):
        assert _resolve("sec", "vibes_per_share") == (None, "residual")

    def test_wrong_class_metric_fails_closed(self):
        """closing_price is real but market-scoped; on a sec claim it is not
        laundered across classes — it goes to null."""
        assert _resolve("sec", "closing_price") == (None, "residual")

    def test_reject_forces_null(self):
        assert _resolve("reject", "revenue") == (None, "reject_null")


class TestNormalizeParserOutput:

    def test_reconciliation_runs_before_metric_scoping(self):
        """A coerced reject must resolve its metric against the REJECT rules,
        not the original claim_type's whitelist."""
        raw = {"claim_type": "sec", "metric": "revenue", "value": 4e9,
               "reject_reason": "incomplete"}
        data, decisions = normalize_parser_output(raw, "claim text")
        assert data["claim_type"] == "reject"
        assert data["metric"] is None
        assert decisions["metric"] == "reject_null"

    def test_reject_nulls_every_companion_field(self):
        """§8: a reject carries nothing but its reason. This is what makes the
        stage-03-deferred model invariant safe to enforce."""
        raw = {"claim_type": "market", "ticker": "AAPL", "metric": "closing_price",
               "operator": "eq", "comparison": "eq", "value": 200.0,
               "period": "today", "currency": "USD",
               "reject_reason": "ambiguous_entity"}
        data, _ = normalize_parser_output(raw, "x")
        for field in ("ticker", "metric", "operator", "comparison",
                      "value", "period", "currency"):
            assert data[field] is None, field
        assert data["reject_reason"] == "ambiguous_entity"

    def test_value_without_operator_defaults_to_eq(self):
        """The prompt documents eq as the default; the boundary enforces it so
        the iff-invariant cannot 500 a live claim."""
        raw = {"claim_type": "sec", "value": 5e9}
        data, decisions = normalize_parser_output(raw, "x")
        assert data["operator"] == "eq"
        assert decisions["operator"] == "defaulted_eq"

    def test_operator_without_value_is_dropped(self):
        raw = {"claim_type": "news", "comparison": "gt", "value": None}
        data, decisions = normalize_parser_output(raw, "x")
        assert data.get("operator") is None and data.get("comparison") is None
        assert decisions["operator"] == "dropped_operator_without_value"

    def test_clean_input_passes_through_with_no_decisions(self):
        raw = {"claim_type": "sec", "ticker": "JPM", "metric": "revenue",
               "operator": "eq", "value": 158.1e9, "period": "fiscal 2024",
               "reject_reason": None}
        data, decisions = normalize_parser_output(raw, "x")
        assert data["metric"] == "revenue"
        assert decisions == {"reject": "none", "metric": "whitelist",
                             "operator": "none"}

    def test_raw_input_is_not_mutated(self):
        raw = {"claim_type": "sec", "metric": "iphone_revenue", "value": 1.0}
        normalize_parser_output(raw, "x")
        assert raw["metric"] == "iphone_revenue"
