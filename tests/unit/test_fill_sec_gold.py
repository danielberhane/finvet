"""Tests for filling missing source values in the SEC gold rows.

72 rows of the real-sourced held-out set name a company, an XBRL concept, a
period and an accession, but never recorded the value the company filed. Filling
them takes the retrieval harness from 128 scoreable cases to 200.

The fill MUST NOT reuse FinVet's own fact selection. FinVet reads SEC's
companyconcept endpoint through _select_fact_for_period; generating the answer
key with the same code that is then under test would make the test pass by
construction and bake any selection bug into the ground truth. So this reads the
independent frames endpoint, which is keyed by exactly the xbrl_frame the
dataset already records and applies SEC's own period normalisation.
"""

import json

from finvet.eval.fill_sec_gold import (
    LABEL_SOURCE,
    UNIT_CANDIDATES,
    build_entry,
    find_company_value,
    frame_url,
    frames_needed,
    load_sidecar,
    needs_fill,
)


def _row(rid=1, fact="us-gaap:EarningsPerShareDiluted", frame="CY2024Q2", value=None):
    prov = {
        "source_type": "sec_xbrl", "cik": "0000037996",
        "accession": "0000037996-25-000147", "xbrl_fact": fact,
        "xbrl_frame": frame, "period_end": "2024-06-30",
    }
    if value is not None:
        prov["source_value_exact"] = value
    else:
        prov["source_value_exact"] = None
    return {"id": rid, "input": "x", "gold": {"claim_type": "sec"}, "provenance": prov}


class TestNeedsFill:

    def test_row_with_a_concept_but_no_value_needs_filling(self):
        assert needs_fill(_row()) is True

    def test_row_that_already_has_a_value_is_left_alone(self):
        """Never overwrite a value the dataset authors recorded themselves."""
        assert needs_fill(_row(value=0.46)) is False

    def test_row_without_a_concept_cannot_be_filled(self):
        row = _row()
        del row["provenance"]["xbrl_fact"]
        assert needs_fill(row) is False

    def test_row_without_a_frame_cannot_be_filled(self):
        """The frames endpoint is keyed by frame; without one there is no lookup."""
        assert needs_fill(_row(frame=None)) is False


class TestFramesNeeded:

    def test_deduplicates_lookups(self):
        """72 rows collapse to far fewer network calls because concept and frame
        repeat heavily, and one frame response carries every company."""
        rows = [_row(1), _row(2), _row(3, fact="us-gaap:Revenues", frame="CY2024")]
        assert frames_needed(rows) == {
            ("EarningsPerShareDiluted", "CY2024Q2"),
            ("Revenues", "CY2024"),
        }

    def test_ignores_rows_that_do_not_need_filling(self):
        assert frames_needed([_row(1, value=0.46)]) == set()


class TestFrameUrl:

    def test_builds_the_documented_path(self):
        assert frame_url("Revenues", "USD", "CY2024") == (
            "https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2024.json"
        )

    def test_per_share_units_are_a_candidate(self):
        """EPS is the most common concept in the 72 and is not denominated in USD."""
        assert "USD-per-shares" in UNIT_CANDIDATES


class TestFindCompanyValue:

    PAYLOAD = {"data": [
        {"cik": 37996, "entityName": "FORD MOTOR CO", "val": 0.46,
         "accn": "0000037996-25-000147", "form": "10-Q"},
        {"cik": 320193, "entityName": "Apple Inc.", "val": 1.40,
         "accn": "0000320193-24-000081", "form": "10-Q"},
    ]}

    def test_matches_on_zero_padded_cik(self):
        """The dataset stores CIKs zero-padded to ten digits; frames returns ints."""
        got = find_company_value(self.PAYLOAD, "0000037996")
        assert got["val"] == 0.46
        assert got["accn"] == "0000037996-25-000147"

    def test_company_absent_from_the_frame_returns_none(self):
        assert find_company_value(self.PAYLOAD, "0000000123") is None

    def test_empty_payload_returns_none(self):
        assert find_company_value({}, "0000037996") is None


class TestBuildEntry:

    def test_records_the_independent_source(self):
        """The filled value must never be mistaken for one the dataset authors
        recorded: its provenance says which endpoint produced it."""
        entry = build_entry(_row(rid=7), value=0.46, unit="USD-per-shares",
                            accn="0000037996-25-000147")
        assert entry["row_id"] == 7
        assert entry["source_value_exact"] == 0.46
        assert entry["label_source"] == LABEL_SOURCE
        assert entry["label_source"] != "xbrl_companyfacts"
        assert entry["unit"] == "USD-per-shares"

    def test_carries_the_lookup_key_for_auditing(self):
        entry = build_entry(_row(rid=7), value=0.46, unit="USD", accn="a")
        assert entry["concept"] == "EarningsPerShareDiluted"
        assert entry["frame"] == "CY2024Q2"
        assert entry["cik"] == "0000037996"


class TestLoadSidecar:

    def test_reads_entries_keyed_by_row_id(self, tmp_path):
        p = tmp_path / "fill.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in [
            {"row_id": 7, "source_value_exact": 0.46},
            {"row_id": 9, "source_value_exact": 1.23},
        ]))
        assert load_sidecar(p) == {7: 0.46, 9: 1.23}

    def test_missing_file_is_not_an_error(self):
        """The harness runs with or without a fill; absence just means 128 cases."""
        assert load_sidecar(None) == {}
