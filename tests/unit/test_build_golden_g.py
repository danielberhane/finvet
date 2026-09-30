"""The derivation rule, checked without the network."""
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_golden_g.py"
spec = importlib.util.spec_from_file_location("build_golden_g", SCRIPT)
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


class TestTolerance:
    def test_large_values_get_the_tight_band(self):
        assert bg.tolerance_for(1_000_000_000) == 1.5
        assert bg.tolerance_for(999_999_999) == 2.0


class TestClaimValue:
    def test_supports_lands_inside_band_after_rounding(self):
        filed = 391_035_000_000
        value, text = bg.claim_value(filed, "revenue", side="in", factor=0.3)
        assert bg.magnitude_diff(value, filed) <= bg.tolerance_for(filed)
        assert text.endswith(" billion")

    def test_refutes_lands_outside_band_after_rounding(self):
        filed = 391_035_000_000
        value, text = bg.claim_value(filed, "revenue", side="out", factor=4.0)
        assert bg.magnitude_diff(value, filed) > bg.tolerance_for(filed)

    def test_boundary_pair_straddles_the_band(self):
        filed = 391_035_000_000
        v_in, _ = bg.claim_value(filed, "revenue", side="in", factor=0.85, sig=4)
        v_out, _ = bg.claim_value(filed, "revenue", side="out", factor=1.15, sig=4)
        tol = bg.tolerance_for(filed)
        assert bg.magnitude_diff(v_in, filed) <= tol < bg.magnitude_diff(v_out, filed)

    def test_eps_formats_with_two_decimals(self):
        value, text = bg.claim_value(6.11, "diluted_eps", side="in", factor=0.3)
        assert value == 6.07 and text == "$6.07"

    def test_no_bare_nine_digit_number_in_text(self):
        _, text = bg.claim_value(391_035_000, "net_income", side="in", factor=0.3)
        assert not re.search(r"\b\d{9,10}\b", text)

    def test_approx_seed_places_inside_the_widened_band(self):
        filed = 391_035_000_000
        value, _ = bg.claim_value(filed, "revenue", side="in", factor=1.8, band_multiplier=2.0)
        diff = bg.magnitude_diff(value, filed)
        tol = bg.tolerance_for(filed)
        assert tol < diff <= tol * 2.0


class TestFormatAmount:
    def test_format_amount_reconstructs_an_exact_number(self):
        value, _ = bg.format_amount(67_099_999_999.99, "revenue")
        assert value == 67_100_000_000


class TestPrimaryFact:
    def _payload(self, rows):
        return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}}}

    def test_comparative_row_is_not_chosen(self):
        rows = [
            {"fy": 2024, "fp": "FY", "form": "10-K", "start": "2023-01-01", "end": "2023-12-31",
             "val": 90, "accn": "a"},
            {"fy": 2024, "fp": "FY", "form": "10-K", "start": "2024-01-01", "end": "2024-12-31",
             "val": 100, "accn": "a"},
        ]
        unit, fact = bg.primary_fact(self._payload(rows), "Revenues", 2024, "FY")
        assert fact["val"] == 100

    def test_ytd_quarter_is_excluded_by_duration(self):
        rows = [
            {"fy": 2025, "fp": "Q2", "form": "10-Q", "start": "2025-01-01", "end": "2025-06-30",
             "val": 60, "accn": "b"},
            {"fy": 2025, "fp": "Q2", "form": "10-Q", "start": "2025-04-01", "end": "2025-06-30",
             "val": 31, "accn": "b", "frame": "CY2025Q2"},
        ]
        unit, fact = bg.primary_fact(self._payload(rows), "Revenues", 2025, "Q2")
        assert fact["val"] == 31

    def test_primary_fact_prefers_the_filings_own_period_over_a_framed_comparative(self):
        # The SEC attaches `frame` to the most recent filing to report a
        # period, so an older accession's own current-period fact can lose
        # its frame once a later filing repeats the same period as a
        # comparative. Both rows below share one accession (one filing):
        # the earlier-ending row (the comparative) carries a frame, the
        # later-ending row (the filing's own current period) does not.
        rows = [
            {"fy": 2025, "fp": "Q1", "form": "10-Q", "start": "2024-01-01", "end": "2024-03-31",
             "val": 50, "accn": "c", "frame": "CY2024Q1"},
            {"fy": 2025, "fp": "Q1", "form": "10-Q", "start": "2025-01-01", "end": "2025-03-31",
             "val": 70, "accn": "c"},
        ]
        unit, fact = bg.primary_fact(self._payload(rows), "Revenues", 2025, "Q1")
        assert fact["end"] == "2025-03-31"

    def test_restated_values_reports_both(self):
        rows = [
            {"fy": 2023, "fp": "FY", "form": "10-K", "start": "2023-01-01", "end": "2023-12-31",
             "val": 100, "accn": "orig", "frame": "CY2023"},
            {"fy": 2023, "fp": "FY", "form": "10-K/A", "start": "2023-01-01", "end": "2023-12-31",
             "val": 97, "accn": "amend"},
        ]
        vals = bg.restated_values(self._payload(rows), "Revenues")
        all_vals = [r["val"] for rs in vals.values() for r in rs]
        assert sorted(all_vals) == [97, 100]


class TestMakeRow:
    def _filer(self):
        return {"ticker": "ZZZ", "name": "Zzz Corp", "class": "mid_cap"}

    def test_make_row_refuses_a_fact_that_contradicts_the_fiscal_label(self):
        # For Qn the accepted years are (fy-1, fy, fy+1); a fact 2+ years off
        # the fiscal label must be refused outright.
        fact = {"val": 100, "end": "2022-03-31", "start": "2022-01-01",
                "accn": "x", "form": "10-Q"}
        with pytest.raises(ValueError):
            bg.make_row(1, self._filer(), "revenue", "Revenues", fact, "USD",
                       "Q1", 2025, "SUPPORTS", 100, "$100", "eq", "plain")

    def test_make_row_period_matches_the_fact(self):
        fact = {"val": 100, "end": "2025-03-31", "start": "2025-01-01",
                "accn": "x", "form": "10-Q"}
        row = bg.make_row(1, self._filer(), "revenue", "Revenues", fact, "USD",
                          "Q1", 2025, "SUPPORTS", 100, "$100", "eq", "plain")
        assert row["gold_parse"]["period"] == "Q1 2025"
        assert "period_end 2025-03-31" in row["ground_truth"]

    def test_loss_phrasing_follows_the_sign(self):
        fact = {"val": -293_000_000, "end": "2025-12-31", "start": "2025-01-01",
                "accn": "x", "form": "10-K"}
        row = bg.make_row(1, self._filer(), "net_income", "NetIncomeLoss", fact, "USD",
                          "FY", 2025, "SUPPORTS", -293_000_000, "-$293 million",
                          "eq", "plain")
        assert "reported a net loss of $293 million" in row["claim"]
        assert "-$" not in row["claim"]
