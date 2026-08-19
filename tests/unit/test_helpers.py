"""Tests for shared utility functions."""

from finvet.utils.helpers import get_confidence_label, build_preliminary_analysis
from finvet.config.constants import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MODERATE_THRESHOLD


class TestGetConfidenceLabel:

    def test_high_confidence(self):
        assert get_confidence_label(0.90) == "HIGH"
        assert get_confidence_label(0.85) == "HIGH"
        assert get_confidence_label(1.0) == "HIGH"

    def test_moderate_confidence(self):
        assert get_confidence_label(0.80) == "MODERATE"
        assert get_confidence_label(0.70) == "MODERATE"

    def test_low_confidence(self):
        assert get_confidence_label(0.69) == "LOW"
        assert get_confidence_label(0.50) == "LOW"
        assert get_confidence_label(0.0) == "LOW"

    def test_boundary_values(self):
        assert get_confidence_label(CONFIDENCE_HIGH_THRESHOLD) == "HIGH"
        assert get_confidence_label(CONFIDENCE_HIGH_THRESHOLD - 0.01) == "MODERATE"
        assert get_confidence_label(CONFIDENCE_MODERATE_THRESHOLD) == "MODERATE"
        assert get_confidence_label(CONFIDENCE_MODERATE_THRESHOLD - 0.01) == "LOW"


class TestBuildPreliminaryAnalysis:

    def _make_parsed_claim(self, value=None, comparison=None):
        """Helper to create a mock ParsedClaim."""
        from finvet.models.claim import ParsedClaim
        return ParsedClaim(
            claim_type="sec",
            ticker="AAPL",
            value=value,
            comparison=comparison,
            period="FY2024",
        )

    def test_equality_claim_with_values(self):
        """Equality claim should include claimed_value and retrieved_value."""
        parsed = self._make_parsed_claim(value=94_000_000_000, comparison="eq")
        state = {
            "parsed_claim": parsed,
            "confidence": 0.90,
        }
        evidence = {
            "verdict": "SUPPORTS",
            "confidence": 0.88,
            "agent": "sec",
            "reasoning": "Revenue matches",
            "retrieved_value": 94_200_000_000,
            "magnitude_difference_percent": 0.21,
            "tools_called": ["get_income_statement"],
        }

        result = build_preliminary_analysis(state, evidence)

        assert result["verdict"] == "SUPPORTS"
        assert result["confidence"] == 0.90
        assert result["confidence_label"] == "HIGH"
        assert result["claimed_value"] == 94_000_000_000
        assert result["retrieved_value"] == 94_200_000_000
        assert result["magnitude_difference_percent"] == 0.21

    def test_equality_claim_no_comparison_field(self):
        """Claim with comparison=None should still show comparison (defaults to eq)."""
        parsed = self._make_parsed_claim(value=100.0, comparison=None)
        state = {"parsed_claim": parsed}
        evidence = {
            "verdict": "SUPPORTS",
            "confidence": 0.85,
            "agent": "market",
            "retrieved_value": 101.0,
        }

        result = build_preliminary_analysis(state, evidence)
        assert result["claimed_value"] == 100.0
        assert result["retrieved_value"] == 101.0

    def test_directional_claim_shows_values(self):
        """Directional claims (gt) now include claimed/retrieved values."""
        parsed = self._make_parsed_claim(value=100.0, comparison="gt")
        state = {"parsed_claim": parsed}
        evidence = {
            "verdict": "SUPPORTS",
            "confidence": 0.90,
            "agent": "market",
            "retrieved_value": 120.0,
            "magnitude_difference_percent": 20.0,
        }

        result = build_preliminary_analysis(state, evidence)
        assert result["claimed_value"] == 100.0
        assert result["retrieved_value"] == 120.0
        assert result["magnitude_difference_percent"] == 20.0

    def test_no_parsed_claim(self):
        """Handle missing parsed_claim gracefully."""
        state = {"confidence": 0.5}
        evidence = {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.5,
            "agent": "sec",
        }

        result = build_preliminary_analysis(state, evidence)
        assert result["verdict"] == "NOT_ENOUGH_INFO"
        assert result["claimed_value"] is None
        assert result["retrieved_value"] is None

    def test_confidence_from_evidence_fallback(self):
        """When state has no confidence, use evidence confidence."""
        parsed = self._make_parsed_claim(value=50.0)
        state = {"parsed_claim": parsed}
        evidence = {"confidence": 0.72, "agent": "sec"}

        result = build_preliminary_analysis(state, evidence)
        assert result["confidence"] == 0.72
        assert result["confidence_label"] == "MODERATE"
