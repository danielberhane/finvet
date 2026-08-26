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


class TestNoAgentIsNamedWhenNoAgentRan:
    """"UNKNOWN Agent · 0.0s" was shown on a rejected claim.

    Nothing unknown happened. The claim parser read "What was Apple's revenue
    in 2024?", classified it as a question, and stopped — no agent was ever
    selected, which is why `metadata` carries no `agent` key. The UI defaulted
    the missing value to the string "unknown" and captioned it as though an
    agent had run and could not be identified.

    The rejection already says which stage stopped it, in `disposition`. That
    is what the line should name.
    """

    def _attribution(self, metadata):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
        from components.formatting import stage_label

        return stage_label(metadata)

    def test_a_parser_rejection_names_the_parser(self):
        assert self._attribution(
            {"disposition": "rejected_parser"}) == "Claim Parser"

    def test_a_guardrail_block_names_the_guard(self):
        assert self._attribution(
            {"disposition": "rejected_input_guard"}) == "Input Guardrails"

    def test_a_human_rejection_names_the_reviewer(self):
        assert self._attribution(
            {"disposition": "rejected_human"}) == "Human Reviewer"

    def test_an_agent_run_names_the_agent(self):
        for wire, shown in (("sec", "SEC Agent"), ("market", "Market Agent"),
                            ("news", "News Agent")):
            assert self._attribution({"agent": wire}) == shown

    def test_nothing_is_named_when_nothing_is_known(self):
        """Better an empty attribution than an invented one."""
        assert self._attribution({}) == ""

    def test_the_word_unknown_never_appears(self):
        for metadata in ({}, {"agent": None}, {"agent": "unknown"},
                         {"disposition": "rejected_parser"}):
            assert "unknown" not in self._attribution(metadata).lower()


class TestADelegationIsNamedAsOne:
    """"News Agent" alone hides the most interesting thing that happened.

    When the News agent corroborates a finding against the issuer's own filing,
    two agents ran and one asked the other. The line under the verdict named
    only the first, so the delegation — the feature — was visible in the badge
    and the raw JSON but not in the sentence describing who produced the
    answer.

    The direction is already in the response (`data_sources.a2a.direction`), so
    this reads it rather than inferring it.
    """

    def _attribution(self, metadata):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
        from components.formatting import stage_label

        return stage_label(metadata)

    def _with_a2a(self, direction="news_to_sec", used=True):
        return {"agent": "news",
                "data_sources": {"a2a": {"used": used, "direction": direction}}}

    def test_a_news_to_sec_delegation_names_both_agents(self):
        assert self._attribution(self._with_a2a()) == "News Agent → SEC Agent"

    def test_the_arrow_follows_the_recorded_direction(self):
        assert self._attribution(
            self._with_a2a("sec_to_news")) == "SEC Agent → News Agent"

    def test_an_unrecorded_direction_falls_back_to_the_agent(self):
        """Better the plain agent than an invented delegation."""
        assert self._attribution(self._with_a2a("sideways")) == "News Agent"

    def test_a_delegation_that_did_not_run_names_one_agent(self):
        assert self._attribution(self._with_a2a(used=False)) == "News Agent"

    def test_a_plain_agent_run_is_unchanged(self):
        assert self._attribution({"agent": "sec"}) == "SEC Agent"

    def test_a_rejection_is_unchanged(self):
        assert self._attribution(
            {"disposition": "rejected_parser"}) == "Claim Parser"
