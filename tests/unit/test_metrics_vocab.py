"""Tests for the vendored metric vocabulary.

The vocabulary is copied from the claim-parser project's scripts/vocab.py —
vendored, not imported, because the repos are separate. These tests pin the
copy's shape without depending on the sibling repo; a separate skipif-guarded
test compares against the source when it is present, so drift is caught on
machines that have both.
"""

from pathlib import Path

import pytest

from finvet.eval.dataset import eval_data_dir
from finvet.config.metrics import (
    METRIC_HARD_DROPS,
    METRIC_REMAPS,
    METRIC_WHITELIST,
    SERVABLE_METRICS,
)

ALL_WHITELISTED = frozenset().union(*METRIC_WHITELIST.values())


class TestWhitelist:

    def test_class_sizes_match_the_contract(self):
        assert len(METRIC_WHITELIST["sec"]) == 22
        assert len(METRIC_WHITELIST["market"]) == 21
        assert len(METRIC_WHITELIST["news"]) == 31

    def test_no_metric_appears_in_two_classes(self):
        """Zero overlap is what makes claim_type fully scope the candidate set."""
        assert len(ALL_WHITELISTED) == 74

    def test_reject_has_no_metrics(self):
        assert METRIC_WHITELIST["reject"] == frozenset()

    def test_known_members(self):
        assert "revenue" in METRIC_WHITELIST["sec"]
        assert "research_and_development" in METRIC_WHITELIST["sec"]
        assert "52_week_high" in METRIC_WHITELIST["market"]
        assert "estimated_revenue" in METRIC_WHITELIST["news"]


class TestRemaps:

    def test_every_remap_target_is_whitelisted_for_its_class(self):
        """An alias must land on a real metric of the same claim_type, or the
        remap layer would launder invalid names into the pipeline."""
        for claim_type, remaps in METRIC_REMAPS.items():
            for alias, target in remaps.items():
                assert target in METRIC_WHITELIST[claim_type], (
                    f"{claim_type}: {alias} -> {target} not whitelisted"
                )

    def test_no_alias_shadows_a_canonical_name(self):
        for claim_type, remaps in METRIC_REMAPS.items():
            shadowed = set(remaps) & METRIC_WHITELIST[claim_type]
            assert not shadowed, f"aliases shadow canonical names: {shadowed}"

    def test_known_aliases(self):
        assert METRIC_REMAPS["market"]["stock_price"] == "closing_price"
        assert METRIC_REMAPS["news"]["fine"] == "fine_amount"


class TestHardDrops:

    def test_the_one_whitelist_collision_is_known_and_shadowed(self):
        """The SOURCE vocabulary lists operating_margin in both the sec
        whitelist and the hard-drops. It is harmless there only because
        resolve_metric checks the whitelist FIRST, so the drop entry is dead
        code — which makes whitelist-before-drops ordering load-bearing for
        our resolver too (stage 04). Vendoring stays byte-faithful, so this
        pins the collision at exactly that one name; a second collision means
        the source changed and the ordering assumption needs re-checking."""
        assert METRIC_HARD_DROPS & ALL_WHITELISTED == {"operating_margin"}

    def test_segment_traps_are_present(self):
        """The traps behind the segment-shadowing bug class."""
        for trap in ("iphone_revenue", "segment_revenue", "cloud_revenue",
                     "advertising_revenue", "data_center_revenue"):
            assert trap in METRIC_HARD_DROPS


class TestServable:

    def test_servable_is_a_subset_of_the_whitelist_per_class(self):
        for claim_type, servable in SERVABLE_METRICS.items():
            assert servable <= METRIC_WHITELIST[claim_type]

    def test_news_serves_nothing_in_release_a(self):
        """The eight FRED-validated macro metrics are retrievable and still
        unservable: no contract says which vintage of a revised series a claim
        refers to. Company-event and analyst metrics were never servable.

        Whitelisted-but-unservable is the disclosed-limitation design: the
        parser still recognises these metrics, so the claim is declined with a
        stated reason instead of misparsed into silence."""
        assert SERVABLE_METRICS["news"] == frozenset()
        assert len(METRIC_WHITELIST["news"]) > 8, (
            "the parser must still recognise them in order to decline them")

    def test_stage_00_concepts_are_servable(self):
        """R&D and interest expense became retrievable in stage 00."""
        assert "research_and_development" in SERVABLE_METRICS["sec"]
        assert "interest_expense" in SERVABLE_METRICS["sec"]

    def test_derived_and_non_gaap_are_not_servable_yet(self):
        """Margins and FCF need the stage-05 derivation decision; EBITDA has
        no XBRL concept at all. None may claim servability before that."""
        for metric in ("gross_margin", "free_cash_flow", "ebitda",
                       "adjusted_ebitda", "book_value_per_share"):
            assert metric not in SERVABLE_METRICS["sec"]


_d = eval_data_dir()
VOCAB_SOURCE = _d.parent.parent / "scripts/vocab.py" if _d else Path("/nonexistent")


@pytest.mark.skipif(not VOCAB_SOURCE.exists(), reason="eval dataset repo not present")
class TestVendoredCopyMatchesSource:
    """Drift detection: only runs where the sibling repo exists."""

    @pytest.fixture(scope="class")
    def source(self):
        ns = {}
        exec(VOCAB_SOURCE.read_text(), ns)
        return ns

    def test_whitelist_matches(self, source):
        for ct in ("sec", "market", "news"):
            assert METRIC_WHITELIST[ct] == source["METRIC_WHITELIST"][ct]

    def test_remaps_match(self, source):
        for ct in ("sec", "market", "news"):
            assert METRIC_REMAPS[ct] == source["METRIC_REMAPS"][ct]

    def test_hard_drops_match(self, source):
        assert METRIC_HARD_DROPS == source["METRIC_HARD_DROPS"]


class TestMetricToConcepts:
    """The fallback's concept map must never reference a concept FinVet does
    not request — a stale entry would silently never match."""

    def test_every_concept_is_requested_from_sec(self):
        from finvet.config.metrics import METRIC_TO_CONCEPTS
        from finvet.mcp.sec_edgar import CONCEPTS_BY_TYPE
        requested = {c for lst in CONCEPTS_BY_TYPE.values() for c in lst}
        stale = {m: [c for c in cs if c not in requested]
                 for m, cs in METRIC_TO_CONCEPTS.items()}
        stale = {m: cs for m, cs in stale.items() if cs}
        assert not stale, f"concepts never requested: {stale}"

    def test_only_servable_direct_metrics_are_mapped(self):
        from finvet.config.metrics import METRIC_TO_CONCEPTS, SERVABLE_METRICS
        assert set(METRIC_TO_CONCEPTS) <= SERVABLE_METRICS["sec"]
