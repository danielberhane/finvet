"""A verdict can be right by the wrong route, and nothing else notices.

`expected.sources` names the retrieval path each claim should take -- xbrl, rag,
a2a, or none -- and the runner records which were used. The two sat in the same
artifact, uncompared, for every run. Backtested across the recorded DeepSeek
runs, comparing them finds two reproducible cases: id 51 answered from `rag`
when the News->SEC delegation should have fired, and id 55 recorded no source at
all. Both returned the expected verdict every time, so the outcome layer saw
nothing wrong.

These drive `artifacts.load_runs` on a real artifact file rather than building
`Run` objects directly, because the loader is what decides the row shape this
measure reads.
"""

import json

from finvet.eval.measures import artifacts, routing


def _artifact(tmp_path, rows):
    """A run artifact in the shape run_golden.py writes."""
    path = tmp_path / "run-20260101T000000Z-test.json"
    path.write_text(json.dumps({
        "started_utc": "20260101T000000Z", "label": "test", "complete": True,
        "llm_config": {"agent": {"model": "test-model"}},
        "results": rows,
    }))
    return artifacts.load_runs([path])


def _row(row_id, expected_sources, used, verdict="SUPPORTS", **over):
    row = {
        "id": row_id, "claim": "c", "category": "sec/xbrl", "strength": "strict",
        "expected": {"verdict": verdict, "sources": expected_sources,
                     "limitation": None},
        "actual": {"verdict": verdict, "data_sources": used},
        "error": None,
    }
    row.update(over)
    return row


class TestARouteThatDisagreesIsReported:

    def test_matching_sources_score(self, tmp_path):
        runs = _artifact(tmp_path, [_row(1, ["xbrl"], ["xbrl"])])

        result = routing.measure(runs)

        assert result.scored == 1 and result.matched == 1
        assert result.mismatches == {}

    def test_the_delegation_that_never_fired_is_caught(self, tmp_path):
        """id 51's real shape: an a2a claim answered out of filing text. The
        verdict is correct, so only the route reveals it."""
        runs = _artifact(tmp_path, [_row(51, ["a2a"], ["rag"])])

        result = routing.measure(runs)

        assert result.mismatches == {51: (["a2a"], ["rag"])}
        assert result.rate == 0.0

    def test_a_row_that_used_no_source_is_caught(self, tmp_path):
        """id 55's shape: nothing recorded at all."""
        runs = _artifact(tmp_path, [_row(55, ["a2a"], [])])

        result = routing.measure(runs)

        assert result.mismatches == {55: (["a2a"], [])}

    def test_order_does_not_matter(self, tmp_path):
        runs = _artifact(tmp_path, [_row(1, ["xbrl", "rag"], ["rag", "xbrl"])])

        assert routing.measure(runs).matched == 1


class TestUnlabelledIsNotAnAssertion:
    """`sources: []` on an observe row means 'not labelled', not 'no source may
    be used' -- the known-defect rows consult XBRL and RAG by design. Treating
    the empty list as an assertion produced two false failures."""

    def test_a_row_with_no_expected_verdict_is_skipped(self, tmp_path):
        runs = _artifact(tmp_path, [_row(99, [], ["xbrl"], verdict=None)])

        result = routing.measure(runs)

        assert result.scored == 0, "an observe row was scored"
        assert result.mismatches == {}
        assert result.skipped_unlabelled == [99]

    def test_an_empty_expectation_on_an_asserted_row_still_counts(self, tmp_path):
        """A reject row must not reach a source, and that is a real claim."""
        runs = _artifact(tmp_path, [_row(75, [], ["xbrl"], verdict="REJECTED")])

        result = routing.measure(runs)

        assert result.mismatches == {75: ([], ["xbrl"])}


class TestRowsThatCannotBeScored:

    def test_an_errored_row_is_ignored(self, tmp_path):
        runs = _artifact(tmp_path, [
            _row(1, ["xbrl"], None, error="Timeout"),
        ])

        assert routing.measure(runs).scored == 0

    def test_a_row_with_no_recorded_sources_is_ignored(self, tmp_path):
        """An older artifact predating data_sources must not read as a failure."""
        row = _row(1, ["xbrl"], None)
        row["actual"].pop("data_sources")
        runs = _artifact(tmp_path, [row])

        assert routing.measure(runs).scored == 0
