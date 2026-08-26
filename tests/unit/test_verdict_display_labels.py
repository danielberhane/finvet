"""A verdict is displayed as prose; it travels as an enum.

`NOT_ENOUGH_INFO` is a wire value. It belongs in the API response, the audit
record, the raw-response panel and the review decision submitted back to the
server -- and nowhere a person reads a sentence. Showing it on the page leaks a
serialization format into the product.

The split matters in both directions. Prettifying the value that goes *to* the
server would break the contract; leaving the raw value on the page is the thing
being fixed here. The selectboxes therefore keep enum `options` and carry a
`format_func`, so what is shown and what is sent differ on purpose.
"""

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "ui"


def _label():
    import sys

    sys.path.insert(0, str(UI))
    from components.formatting import verdict_label

    return verdict_label


class TestTheLabelIsProse:

    @pytest.mark.parametrize("wire,shown", [
        ("NOT_ENOUGH_INFO", "Not Enough Info"),
        ("SUPPORTS", "Supports"),
        ("REFUTES", "Refutes"),
        ("PENDING", "Pending"),
        ("REJECTED", "Rejected"),
    ])
    def test_each_verdict_reads_as_words(self, wire, shown):
        assert _label()(wire) == shown

    def test_an_unknown_verdict_degrades_rather_than_vanishes(self):
        """A verdict added server-side should read imperfectly, never
        disappear from the page."""
        assert _label()("REVIEW_FINALIZATION_FAILED") == \
            "Review Finalization Failed"

    def test_no_label_contains_an_underscore(self):
        from components.formatting import _VERDICT_LABELS

        assert not [v for v in _VERDICT_LABELS.values() if "_" in v]

    @pytest.mark.parametrize("empty", [None, ""])
    def test_absence_renders_as_a_dash(self, empty):
        assert _label()(empty) == "--"


class TestTheWireValueIsNotRenderedRaw:
    """Guards the regression directly: a template that interpolates the raw
    verdict puts SCREAMING_SNAKE on the page."""

    def _sources(self):
        for path in sorted(UI.rglob("*.py")):
            yield path, path.read_text()

    def test_no_view_interpolates_a_bare_verdict_into_markup(self):
        # A verdict placed straight into an HTML string, not routed through
        # verdict_label. `{verdict_class}` and similar are fine.
        bad = re.compile(r">\{(?:_escape\()?(?:prelim_|match_)?verdict\)?\}<")
        offenders = [
            f"{path.name}:{i}"
            for path, text in self._sources()
            for i, line in enumerate(text.splitlines(), 1)
            if bad.search(line)
        ]

        assert not offenders, (
            f"these render the wire value on the page: {offenders}")

    def test_the_review_selectbox_still_submits_enums(self):
        """The other direction. Prettifying what is sent would break the
        server contract, so options stay raw and only the display is mapped."""
        text = (UI / "views" / "review_detail.py").read_text()

        assert '"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"' in text
        assert "format_func=verdict_label" in text
