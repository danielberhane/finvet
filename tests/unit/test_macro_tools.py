"""Tests for the FRED macro tool.

Eight news-whitelist macro metrics map to FRED series — the exact series the
gold dataset's provenance URLs name, validated against its 16 recorded
values (12 exact; the 4 misses are 2026 data revisions, not mapping errors).
fredgraph.csv needs no API key.

Unlike SEC XBRL, FRED RESTATES history: CPI, retail sales and GDP are revised
after first release, so a claim true against the initial print can drift from
the current series. FinVet verifies against current authoritative data, so
the tool returns today's series value — the revision caveat lives in the
result for the agent to see.
"""

from unittest.mock import patch

from finvet.config.metrics import SERVABLE_METRICS
from finvet.mcp.fred import FRED_SERIES, period_to_observation_date, value_for
from finvet.tools.macro_tools import get_macro_indicator

UNRATE_CSV = "DATE,UNRATE\n2023-07-01,3.5\n2026-04-01,4.3\n"
CPI_CSV = ("DATE,CPIAUCSL\n2023-01-01,300.536\n2024-01-01,309.685\n"
           "2025-04-01,320.321\n2026-04-01,332.406\n")


class TestSeriesMap:

    def test_the_eight_validated_metrics_are_mapped(self):
        assert set(FRED_SERIES) == {
            "unemployment_rate", "federal_funds_rate", "consumer_confidence",
            "gdp_growth", "cpi_inflation", "core_pce",
            "retail_sales_growth", "wage_growth"}

    def test_no_macro_metric_is_servable_in_release_a(self):
        """Retrievable is no longer sufficient.

        These eight series are dataset-validated and FRED returns them
        happily. What is missing is a contract for *which* observation a claim
        means: "inflation was 3.1%" names no vintage, and CPI is revised, so
        the same claim is true or false depending on which release you read.
        A decisive verdict on that rests on an unstated choice, so Release A
        declines the whole class -- see RELEASE_A_DECISIONS.md, D10.

        The map itself is retained: the retrieval works, and lifting the
        decision needs a vintage contract, not new plumbing.
        """
        assert SERVABLE_METRICS["news"] == frozenset()
        assert frozenset(FRED_SERIES), "the series map is kept for Release B"

    def test_series_ids_are_the_datasets_own(self):
        assert FRED_SERIES["unemployment_rate"][0] == "UNRATE"
        assert FRED_SERIES["cpi_inflation"] == ("CPIAUCSL", "yoy")
        assert FRED_SERIES["gdp_growth"] == ("A191RL1Q225SBEA", "direct")


class TestPeriodParsing:

    def test_month_year(self):
        assert period_to_observation_date("April 2026") == "2026-04-01"

    def test_quarter(self):
        assert period_to_observation_date("Q4 2023") == "2023-10-01"

    def test_unparseable_returns_none(self):
        assert period_to_observation_date("recently") is None
        assert period_to_observation_date(None) is None


class TestValueFor:

    def test_direct_series(self):
        with patch("finvet.mcp.fred._fetch_csv", return_value=UNRATE_CSV):
            assert value_for("unemployment_rate", "April 2026") == 4.3

    def test_yoy_transform(self):
        """Jan 2024 CPI yoy = 309.685/300.536 - 1 = 3.04%-ish of the fixture;
        the real-series validation ran live and matched gold to 2 decimals."""
        with patch("finvet.mcp.fred._fetch_csv", return_value=CPI_CSV):
            got = value_for("cpi_inflation", "January 2024")
            assert abs(got - ((309.685 / 300.536 - 1) * 100)) < 1e-9

    def test_missing_observation_returns_none(self):
        with patch("finvet.mcp.fred._fetch_csv", return_value=UNRATE_CSV):
            assert value_for("unemployment_rate", "March 1999") is None

    def test_yoy_without_prior_year_returns_none(self):
        with patch("finvet.mcp.fred._fetch_csv", return_value=CPI_CSV):
            assert value_for("cpi_inflation", "January 2023") is None


class TestTool:

    def test_success_carries_provenance(self):
        with patch("finvet.mcp.fred._fetch_csv", return_value=UNRATE_CSV):
            r = get_macro_indicator.invoke(
                {"metric": "unemployment_rate", "period": "April 2026"})
        assert r["success"] and r["value"] == 4.3
        assert r["series_id"] == "UNRATE"
        assert r["observation_date"] == "2026-04-01"
        assert "revis" in r["note"].lower()      # the restatement caveat travels

    def test_unknown_metric_fails_closed(self):
        r = get_macro_indicator.invoke({"metric": "pmi", "period": "April 2026"})
        assert not r["success"] and "pmi" in (r["error"] or "")

    def test_unparseable_period_fails_closed(self):
        r = get_macro_indicator.invoke(
            {"metric": "unemployment_rate", "period": "recently"})
        assert not r["success"] and "month and year" in (r["error"] or "")


class TestNewsAgentCarriesTheTool:

    def test_get_macro_indicator_is_in_the_news_toolset(self):
        """Import-level check: the tool list literal in the news agent module
        references the tool, so unwiring is a visible diff, not a silent one."""
        import inspect
        from finvet.agents.news_agent import react_agent
        assert "get_macro_indicator" in inspect.getsource(react_agent)
