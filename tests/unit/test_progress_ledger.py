"""The waiting time shows the evidence, not a spinner.

A run took ten or twenty seconds and displayed a rotating caption — "Resolving
time period…" — that named a stage and told the reader nothing about their
claim. Worse, the stage was a guess: the UI keyword-matched the raw claim text
to decide which caption to show, so "Apple announced revenue of $391 billion"
was captioned "Searching news sources…" while the SEC agent ran.

Now each step leaves the artifact it produced, and the step that matters most —
the deterministic comparison — is shown forming. When Python overrules the
model, that is FinVet's entire argument, and it used to happen silently.

Two rules the rendering must keep:

- **No underscores reach the reader.** Node names, limitations and triggers are
  wire values; they go through `humanize` before they are shown.
- **No blank lines inside the HTML.** A blank line terminates an `st.markdown`
  block early, which is why every component here builds with list-append and
  "\\n".join rather than a template.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))

from components.progress import progress_html, progress_rows  # noqa: E402


def _event(node, **detail):
    return {"type": "progress", "node": node, "detail": detail}


PARSED = _event("claim_parser", claim_type="sec", ticker="AAPL",
                metric="revenue", operator="eq", value=391_000_000_000.0,
                period="fiscal year 2024")
PERIOD = _event("period_resolver", start="2023-10-01", end="2024-09-28",
                assumption="Interpreted 'fiscal year 2024' as the fiscal year")
AGENT = _event("sec_agent", agent="sec", tools=["get_income_statement"],
               retrieved_value=391_035_000_000.0,
               magnitude_difference_percent=0.009,
               llm_original_verdict="NOT_ENOUGH_INFO", override_applied=True,
               concept="Revenues", period_end="2024-09-28", rag_chunks=0)
CONSENSUS = _event("consensus", verdict="SUPPORTS", confidence=0.95)
GUARDS = _event("output_guardrails", hitl_required=False, hitl_triggers=[])


class TestTheLedgerBuildsFromWhatArrived:

    def test_no_events_yields_no_rows(self):
        """`api_client` falls back to the non-streaming endpoint on any
        exception and yields only a `complete`, so this is a real case."""
        assert progress_rows([]) == []

    def test_each_event_becomes_one_row_in_order(self):
        rows = progress_rows([PARSED, PERIOD, AGENT])

        assert len(rows) == 3
        assert [r["state"] for r in rows] == ["done", "done", "running"]

    def test_the_last_row_is_the_one_still_running(self):
        rows = progress_rows([PARSED, PERIOD])

        assert rows[-1]["state"] == "running"
        assert rows[0]["state"] == "done"

    def test_a_completed_run_leaves_nothing_running(self):
        rows = progress_rows([PARSED, AGENT, CONSENSUS], finished=True)

        assert all(r["state"] == "done" for r in rows)

    def test_an_unknown_node_is_still_shown_readably(self):
        rows = progress_rows([_event("some_new_node")])

        assert rows[0]["label"] == "Some New Node"
        assert "_" not in rows[0]["label"]


class TestEachStepShowsWhatItProduced:

    def _detail(self, events, node):
        rows = progress_rows(events, finished=True)
        return next(r["detail"] for r in rows if r["node"] == node)

    def test_the_parse_is_shown_not_guessed(self):
        detail = self._detail([PARSED], "claim_parser")

        assert "AAPL" in detail
        assert "revenue" in detail

    def test_the_period_shows_its_window(self):
        detail = self._detail([PERIOD], "period_resolver")

        assert "2023-10-01" in detail
        assert "2024-09-28" in detail

    def test_a_qualitative_parse_says_where_it_routed(self):
        """A null metric is not a gap — it is the decision that sends the
        claim to filing text rather than XBRL."""
        detail = self._detail(
            [_event("claim_parser", claim_type="sec", ticker="AAPL",
                    metric=None, value=None)], "claim_parser")

        assert "filing text" in detail.lower()

    def test_retrieved_passages_are_counted(self):
        detail = self._detail(
            [_event("sec_agent", agent="sec", rag_chunks=5)], "sec_agent")

        assert "5" in detail


class TestTheComparisonIsThePayoff:

    def _rows(self, **overrides):
        return progress_rows([PARSED, dict(AGENT, detail=dict(
            AGENT["detail"], **overrides))], finished=True)

    def test_it_shows_both_numbers_and_the_distance(self):
        detail = self._rows()[-1]["detail"]

        assert "391.00B" in detail, "the claimed value must be shown"
        assert "391.04B" in detail or "391.03B" in detail
        assert "0.01%" in detail

    def test_an_override_is_stated_in_words(self):
        """The moment the whole system exists for. It used to be silent."""
        detail = self._rows(override_applied=True,
                            llm_original_verdict="NOT_ENOUGH_INFO")[-1]["detail"]

        assert "Not Enough Info" in detail, "the model's verdict, in prose"
        assert "NOT_ENOUGH_INFO" not in detail, "not the wire value"

    def test_no_override_says_nothing_about_the_model(self):
        detail = self._rows(override_applied=False,
                            llm_original_verdict="SUPPORTS")[-1]["detail"]

        assert "Not Enough Info" not in detail
        assert "model" not in detail.lower()

    def test_the_claimed_value_comes_from_the_earlier_parse(self):
        """It is absent from agent evidence on the normal path, so the ledger
        joins it from the claim_parser event it already holds."""
        rows = progress_rows([AGENT], finished=True)

        assert "391.00B" not in rows[-1]["detail"]


class TestNothingReadsLikeCode:

    def _all_text(self, rows):
        return " ".join(f"{r['label']} {r['detail']}" for r in rows)

    def test_no_label_or_detail_contains_an_underscore(self):
        rows = progress_rows([PARSED, PERIOD, AGENT, CONSENSUS, GUARDS],
                             finished=True)

        assert "_" not in self._all_text(rows)

    def test_a_limitation_is_rendered_as_words(self):
        rows = progress_rows(
            [_event("sec_agent", agent="sec",
                    limitation="unsupported_q4_derivation")], finished=True)

        assert "Unsupported Q4 Derivation" in self._all_text(rows)

    def test_a_review_trigger_is_rendered_as_words(self):
        rows = progress_rows(
            [_event("output_guardrails", hitl_required=True,
                    hitl_triggers=["low_confidence"])], finished=True)

        assert "Low Confidence" in self._all_text(rows)


class TestTheHtmlIsSafeToRender:

    def test_it_contains_no_blank_line(self):
        """A blank line ends an st.markdown HTML block early."""
        html = progress_html(progress_rows([PARSED, PERIOD, AGENT]))

        assert "\n\n" not in html

    def test_no_rows_and_no_claim_render_as_nothing(self):
        assert progress_html([]) == ""

    def test_it_reuses_the_existing_timeline_classes(self):
        html = progress_html(progress_rows([PARSED]))

        assert "pipeline-timeline" in html
        assert "pipeline-step" in html

    def test_a_running_step_is_marked_differently_from_a_finished_one(self):
        """Was asserted with the grey `-skip` dot, which reads as "skipped".
        A step still going is not a step passed over."""
        html = progress_html(progress_rows([PARSED, PERIOD]))

        assert "pipeline-step-dot-ok" in html
        assert "pipeline-step-dot-running" in html

    @pytest.mark.parametrize("hostile", [
        "<script>alert(1)</script>", "Apple & Microsoft", "a <b>bold</b> claim"])
    def test_values_are_escaped(self, hostile):
        html = progress_html(progress_rows(
            [_event("claim_parser", claim_type="sec", ticker=hostile)]))

        assert "<script>" not in html
        assert "<b>" not in html


class TestTheRunningStepIsVisiblyRunning:
    """The ledger used to live inside `st.status`, which supplied the spinner.

    That widget also auto-collapsed itself the moment the run completed —
    Streamlit does that on `state="complete"`, and no `expanded=True` overrides
    it — so the ledger closed at exactly the moment a reader wanted to read it.
    Removing the widget removes the collapse, and the ledger has to say for
    itself which step is still going.
    """

    def test_a_running_step_pulses(self):
        html = progress_html(progress_rows([PARSED, PERIOD]))

        assert "pipeline-step-dot-running" in html

    def test_a_finished_step_does_not(self):
        html = progress_html(progress_rows([PARSED, PERIOD]))

        assert "pipeline-step-dot-ok" in html

    def test_a_completed_run_has_nothing_running(self):
        html = progress_html(progress_rows([PARSED, PERIOD, AGENT],
                                           finished=True))

        assert "pipeline-step-dot-running" not in html

    def test_the_grey_skip_dot_is_no_longer_used(self):
        """It read as 'skipped', which a running step is not."""
        html = progress_html(progress_rows([PARSED, PERIOD]))

        assert "pipeline-step-dot-skip" not in html


class TestTheLedgerCarriesItsOwnHeading:
    """The claim text and title lived in the status widget's chrome. With the
    widget gone they belong in the rendered block, or they vanish with it."""

    def test_the_claim_is_shown_when_given(self):
        html = progress_html(progress_rows([PARSED]), claim="Apple's revenue")

        assert "Apple&#x27;s revenue" in html or "Apple's revenue" in html

    def test_the_claim_is_escaped(self):
        html = progress_html(progress_rows([PARSED]),
                             claim="<script>alert(1)</script>")

        assert "<script>" not in html

    def test_no_heading_without_a_claim(self):
        html = progress_html(progress_rows([PARSED]))

        assert "pipeline-timeline" in html

    def test_it_still_has_no_blank_lines_with_a_heading(self):
        html = progress_html(progress_rows([PARSED, PERIOD]), claim="a claim")

        assert "\n\n" not in html


class TestTheHeadingSaysWhetherWorkIsHappening:
    """Between steps an agent can think for five to ten seconds, and a ledger
    that sits perfectly still reads as stalled rather than working.

    The signal is pure CSS: three dots whose *opacity* animates on a stagger.
    Streamlit reruns re-execute the script and would restart the SSE stream, so
    nothing here may be driven from Python. Opacity rather than appearing and
    disappearing text, because dots that come and go reflow the line on every
    frame; these always occupy their space.
    """

    def test_a_running_ledger_pulses(self):
        html = progress_html(progress_rows([PARSED]), claim="a claim")

        assert "verifying-dots" in html
        assert html.count("verifying-dot-") == 3, "three staggered dots"

    def test_a_finished_ledger_does_not(self):
        html = progress_html(progress_rows([PARSED], finished=True),
                             claim="a claim")

        assert "verifying-dots" not in html

    def test_a_stopped_run_does_not_animate_anywhere(self):
        """The case that would otherwise pulse forever: every failure branch
        returns without re-rendering, so the last frame was a running one."""
        html = progress_html(progress_rows([PARSED], finished=True),
                             claim="a claim", stopped=True)

        assert "verifying-dots" not in html
        assert "pipeline-step-dot-running" not in html


class TestTheHeadingNamesTheOutcome:

    def _heading(self, **kwargs):
        return progress_html(progress_rows([PARSED],
                                           finished=kwargs.pop("finished", False)),
                             claim="a claim", **kwargs)

    def test_running_says_verifying(self):
        assert "Verifying" in self._heading()

    def test_finished_says_checked(self):
        """Not "Verified". The heading now sits *above* the verdict, so on a
        refuted claim "Verified" would read as the verdict itself rather than
        as a statement that checking finished. "Checked" is true whether the
        claim was supported, refuted, declined or rejected."""
        html = self._heading(finished=True)

        assert "Checked" in html
        assert "Verifying" not in html

    def test_the_finished_heading_never_claims_a_verdict(self):
        html = self._heading(finished=True)

        assert "Verified" not in html, (
            "this heading renders above a REFUTES verdict too")

    def test_a_stopped_run_does_not_claim_to_have_verified(self):
        html = self._heading(finished=True, stopped=True)

        assert "Stopped" in html
        assert "Checked" not in html


class TestTheBlockAppearsBeforeTheFirstStep:
    """`progress_html([])` returned "", so between clicking Verify and the
    first event there was nothing on screen — the moment a reader most needs
    to see that something started."""

    def test_a_claim_with_no_steps_yet_still_renders(self):
        html = progress_html([], claim="Apple's revenue was $391 billion")

        assert html != ""
        assert "Verifying" in html
        assert "verifying-dots" in html

    def test_it_shows_no_timeline_when_there_are_no_steps(self):
        html = progress_html([], claim="a claim")

        assert "pipeline-step" not in html

    def test_the_claim_is_escaped_in_that_state_too(self):
        html = progress_html([], claim="<script>alert(1)</script>")

        assert "<script>" not in html

    def test_it_has_no_blank_lines(self):
        html = progress_html([], claim="a claim")

        assert "\n\n" not in html
