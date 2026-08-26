"""Nothing a person reads should look like a variable name.

`Claim rejected: non_financial` was shown to a user. The value is correct and
belongs in `metadata.reject_reason`, where a program reads it; the `summary`
field is prose by contract and had a serialization format pasted into it.

The split is the same one the verdict labels make: the wire value travels, the
prose is rendered. This file guards both halves — the machine-readable field
keeps its enum, and no user-facing string carries an underscore.
"""

import pytest


class TestTheRejectionSummaryReadsAsProse:

    def _reject(self, reason):
        from finvet.graph.nodes.response_generator import (
            _generate_rejection_response)

        return _generate_rejection_response({
            "request_id": "req_x", "claim_raw": "something",
            "disposition": "rejected_parser", "disposition_detail": reason,
        })["final_response"]

    @pytest.mark.parametrize("reason,shown", [
        ("non_financial", "Not a financial claim"),
        ("question", "A question, not a claim"),
        ("incomplete", "Incomplete claim"),
        ("advice_seeking", "Advice request"),
    ])
    def test_each_reason_has_a_readable_title(self, reason, shown):
        assert self._reject(reason)["summary"] == f"Claim rejected: {shown}"

    def test_an_unmapped_reason_is_still_readable(self):
        """A reason added to the parser without a label here must degrade to
        prose, not leak the raw token."""
        assert self._reject("some_new_reason")["summary"] == \
            "Claim rejected: Some New Reason"

    def test_the_machine_readable_value_is_unchanged(self):
        """The other half. Prettifying this would break every consumer."""
        response = self._reject("non_financial")

        assert response["metadata"]["reject_reason"] == "non_financial"
        assert response["metadata"]["disposition"] == "rejected_parser"

    def test_the_explanation_still_explains(self):
        assert self._reject("non_financial")["explanation"] == \
            "This is not a financial claim that can be verified."

    @pytest.mark.parametrize("reason", [
        "non_financial", "question", "incomplete", "advice_seeking", "unknown"])
    def test_no_summary_contains_an_underscore(self, reason):
        assert "_" not in self._reject(reason)["summary"]


class TestTheUiHumanizesAnythingElse:
    """Defence in depth: a field nobody thought about must not read as code."""

    def _humanize(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
        from components.formatting import humanize

        return humanize

    @pytest.mark.parametrize("raw,shown", [
        ("non_financial", "Non Financial"),
        ("unsupported_q4_derivation", "Unsupported Q4 Derivation"),
        ("non_corroboration_is_not_contradiction",
         "Non Corroboration Is Not Contradiction"),
        ("low_confidence", "Low Confidence"),
        ("output_safety_violation", "Output Safety Violation"),
        ("deterministic_fallback", "Deterministic Fallback"),
    ])
    def test_an_enum_becomes_words(self, raw, shown):
        assert self._humanize()(raw) == shown

    def test_q4_keeps_its_capital(self):
        """`.title()` alone gives 'Q4' -> 'Q4', but 'q4' -> 'Q4' matters for
        values that arrive lowercased."""
        assert self._humanize()("q4_derivation") == "Q4 Derivation"

    def test_prose_is_left_alone(self):
        """Something already written for a human must not be re-cased."""
        text = "This is not a financial claim that can be verified."
        assert self._humanize()(text) == text

    @pytest.mark.parametrize("empty", [None, ""])
    def test_absence_is_a_dash(self, empty):
        assert self._humanize()(empty) == "--"
