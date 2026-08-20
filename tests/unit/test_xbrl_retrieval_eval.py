"""Tests for the XBRL retrieval evaluation harness.

The harness answers one question per case: given a filing and an XBRL concept,
does FinVet return the value the company actually filed? These tests cover the
pure logic — concept routing, value comparison, case selection — so a harness
failure means the retrieval layer is wrong rather than the scorer.
"""

import json

import pytest

from finvet.eval.xbrl_retrieval import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_UNSUPPORTED_CONCEPT,
    build_case,
    compare_values,
    load_cases,
    normalize_fact,
    period_kind_for_frame,
    statement_type_for,
    summarize,
)


class TestNormalizeFact:

    def test_strips_the_taxonomy_prefix(self):
        assert normalize_fact("us-gaap:GrossProfit") == "GrossProfit"

    def test_leaves_a_bare_concept_alone(self):
        assert normalize_fact("GrossProfit") == "GrossProfit"


class TestStatementTypeFor:
    """The harness must call the same getter the SEC agent would."""

    def test_income_concept(self):
        assert statement_type_for("RevenueFromContractWithCustomerExcludingAssessedTax") == "income"

    def test_balance_concept(self):
        assert statement_type_for("Assets") == "balance"

    def test_cashflow_concept(self):
        assert statement_type_for("NetCashProvidedByUsedInOperatingActivities") == "cashflow"

    def test_concept_finvet_never_requests_returns_none(self):
        """ResearchAndDevelopmentExpense is in the parser's metric whitelist but
        not in CONCEPTS_BY_TYPE — such claims are unverifiable today, and the
        harness must report that rather than scoring them as failures."""
        assert statement_type_for("ResearchAndDevelopmentExpense") is None


class TestPeriodKindForFrame:
    """A bare end date cannot identify a fact — a 10-Q tags both the quarter and
    the year-to-date figure with the same end. The dataset's frame supplies the
    duration, so the harness can ask FinVet for the right one."""

    def test_annual_frame(self):
        assert period_kind_for_frame("CY2024") == "annual"

    def test_quarterly_frame(self):
        assert period_kind_for_frame("CY2024Q2") == "quarterly"

    def test_missing_frame_defaults_to_annual(self):
        assert period_kind_for_frame(None) == "annual"

    def test_unrecognised_frame_defaults_to_annual(self):
        assert period_kind_for_frame("FY2024H1") == "annual"


class TestCompareValues:

    def test_exact_match_passes(self):
        status, delta, pct = compare_values(184992000000.0, 184992000000.0)
        assert status == STATUS_PASS
        assert delta == 0.0
        assert pct == 0.0

    def test_float_noise_passes(self):
        status, _, _ = compare_values(990.8699951171875, 990.87)
        assert status == STATUS_PASS

    def test_prior_year_comparative_fails_with_the_real_delta(self):
        """The live defect: a prior-year figure returned for the target period."""
        status, delta, pct = compare_values(165901000000.0, 184992000000.0)
        assert status == STATUS_FAIL
        assert delta == pytest.approx(19091000000.0)
        assert pct == pytest.approx(10.32, abs=0.01)

    def test_zero_expected_is_not_a_division_error(self):
        status, delta, pct = compare_values(0.0, 0.0)
        assert status == STATUS_PASS
        assert pct == 0.0


class TestBuildCase:

    def _row(self, **prov):
        base = {
            "cik": "0000037996",
            "accession": "0000037996-26-000015",
            "xbrl_fact": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "period_end": "2024-12-31",
            "source_value_exact": 184992000000.0,
        }
        base.update(prov)
        return {"id": 12, "input": "Ford reported revenue of $185 billion for fiscal 2024.",
                "gold": {"claim_type": "sec", "ticker": "F", "metric": "revenue"},
                "provenance": base}

    def test_carries_the_fields_needed_to_diagnose(self):
        case = build_case(self._row())
        assert case.row_id == 12
        assert case.cik == "0000037996"
        assert case.accession == "0000037996-26-000015"
        assert case.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
        assert case.expected_value == 184992000000.0
        assert case.expected_period_end == "2024-12-31"
        assert case.statement_type == "income"

    def test_carries_the_period_kind_from_the_frame(self):
        case = build_case(self._row(xbrl_frame="CY2024Q2"))
        assert case.period_kind == "quarterly"

    def test_flags_consolidation_sensitive_concepts(self):
        """These are the rows that exercise the entity-wide resolution path."""
        assert build_case(self._row()).consolidation_sensitive is True

    def test_unsupported_concept_is_marked_not_failed(self):
        case = build_case(self._row(xbrl_fact="us-gaap:ResearchAndDevelopmentExpense"))
        assert case.statement_type is None
        assert case.status == STATUS_UNSUPPORTED_CONCEPT


class TestLoadCases:

    def _write(self, tmp_path, rows):
        p = tmp_path / "gold.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def _row(self, rid, fact="us-gaap:GrossProfit", value=1.0, **extra):
        prov = {"cik": "1", "accession": "a", "xbrl_fact": fact,
                "period_end": "2024-12-31", "source_value_exact": value}
        prov.update(extra)
        return {"id": rid, "input": "x", "gold": {"claim_type": "sec"}, "provenance": prov}

    def test_keeps_only_rows_with_a_fact_and_a_source_value(self, tmp_path):
        rows = [
            self._row(1),
            {"id": 2, "input": "y", "gold": {}, "provenance": {"source_type": "reject_authored"}},
            self._row(3, source_value_exact=None),
        ]
        assert [c.row_id for c in load_cases(self._write(tmp_path, rows))] == [1]

    def test_limit_truncates(self, tmp_path):
        rows = [self._row(i) for i in range(1, 6)]
        assert len(load_cases(self._write(tmp_path, rows), limit=2)) == 2

    def test_concept_filter_selects(self, tmp_path):
        rows = [self._row(1, fact="us-gaap:GrossProfit"),
                self._row(2, fact="us-gaap:NetIncomeLoss")]
        got = load_cases(self._write(tmp_path, rows), concept="NetIncomeLoss")
        assert [c.row_id for c in got] == [2]

    def test_selection_is_deterministic(self, tmp_path):
        """A subset run must be reproducible so results are comparable over time."""
        rows = [self._row(i) for i in range(1, 20)]
        path = self._write(tmp_path, rows)
        assert [c.row_id for c in load_cases(path, limit=5)] == [
            c.row_id for c in load_cases(path, limit=5)
        ]


class TestSummarize:

    def _case(self, status, sensitive=False):
        c = build_case({"id": 1, "input": "x", "gold": {},
                        "provenance": {"cik": "1", "accession": "a",
                                       "xbrl_fact": "us-gaap:GrossProfit",
                                       "period_end": "2024-12-31",
                                       "source_value_exact": 1.0}})
        c.status = status
        c.consolidation_sensitive = sensitive
        return c

    def test_pass_rate_excludes_unsupported_concepts(self):
        """Unsupported concepts are a coverage gap, not a retrieval failure —
        counting them as failures would understate the pass rate."""
        cases = [self._case(STATUS_PASS), self._case(STATUS_FAIL),
                 self._case(STATUS_UNSUPPORTED_CONCEPT)]
        s = summarize(cases)
        assert s["scored"] == 2
        assert s["passed"] == 1
        assert s["pass_rate"] == 0.5
        assert s["unsupported_concept"] == 1

    def test_reports_the_consolidation_sensitive_slice(self):
        cases = [self._case(STATUS_PASS, sensitive=True),
                 self._case(STATUS_FAIL, sensitive=True),
                 self._case(STATUS_PASS)]
        s = summarize(cases)
        assert s["consolidation_sensitive"]["scored"] == 2
        assert s["consolidation_sensitive"]["passed"] == 1

    def test_empty_input_does_not_divide_by_zero(self):
        assert summarize([])["pass_rate"] == 0.0


class TestLoadCasesMergesTheFill:
    """71 SEC rows had a concept and period but no recorded value; filling them
    from SEC's frames endpoint takes the harness from 128 cases to 199."""

    def _gold(self, tmp_path):
        rows = [
            {"id": 1, "input": "has a value", "gold": {"claim_type": "sec"},
             "provenance": {"cik": "1", "accession": "a",
                            "xbrl_fact": "us-gaap:GrossProfit",
                            "period_end": "2024-12-31", "source_value_exact": 5.0}},
            {"id": 2, "input": "needs filling", "gold": {"claim_type": "sec"},
             "provenance": {"cik": "1", "accession": "a",
                            "xbrl_fact": "us-gaap:NetIncomeLoss",
                            "period_end": "2024-12-31", "source_value_exact": None}},
        ]
        p = tmp_path / "gold.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def test_unfilled_rows_are_skipped_without_a_sidecar(self, tmp_path):
        assert [c.row_id for c in load_cases(self._gold(tmp_path))] == [1]

    def test_sidecar_makes_the_extra_rows_scoreable(self, tmp_path):
        fill = tmp_path / "fill.jsonl"
        fill.write_text(json.dumps({"row_id": 2, "source_value_exact": 99.0}))
        cases = load_cases(self._gold(tmp_path), fill_path=fill)
        assert [c.row_id for c in cases] == [1, 2]
        assert next(c for c in cases if c.row_id == 2).expected_value == 99.0

    def test_the_sidecar_never_overrides_a_recorded_value(self, tmp_path):
        """A value the dataset authors sourced themselves always wins."""
        fill = tmp_path / "fill.jsonl"
        fill.write_text(json.dumps({"row_id": 1, "source_value_exact": 12345.0}))
        cases = load_cases(self._gold(tmp_path), fill_path=fill)
        assert next(c for c in cases if c.row_id == 1).expected_value == 5.0
