import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_golden.py"
spec = importlib.util.spec_from_file_location("run_golden", SCRIPT)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def test_fill_recipe_places_threshold_and_records_snapshot():
    row = {"id": 1217, "claim": "ACME Test Co's stock is trading above {threshold}",
           "recipe": {"ticker": "ACME", "operator": "gt", "offset_pct": -10},
           "gold_parse": {"value": None}}
    filled = rg.fill_recipe(row, price=250.0, latest_trading_day="2026-09-30")
    assert filled["claim"] == "ACME Test Co's stock is trading above $225.00"
    assert filled["gold_parse"]["value"] == 225.0
    assert filled["snapshot"]["price"] == 250.0
    assert filled["snapshot"]["threshold"] == 225.0


def test_mock_quote_is_refused():
    row = {"id": 1, "claim": "ACME Test Co's stock is trading above {threshold}",
           "recipe": {"ticker": "ACME", "operator": "gt", "offset_pct": 10},
           "gold_parse": {"value": None}}
    with pytest.raises(RuntimeError):
        rg.fill_recipe(row, price=1.0, latest_trading_day=None, source_mode="mock")


def test_prefetch_quotes_returns_empty_when_nothing_needs_a_quote():
    assert rg.prefetch_quotes([{"id": 1, "claim": "ACME Test Co's revenue was $1"}]) == {}


def test_prefetch_quotes_stores_an_exception_and_still_fetches_the_rest(monkeypatch):
    """One bad ticker must not stop the other quotes from being fetched.

    This is the behavior F5 exists for: a recipe row whose quote fails is
    recorded like an errored row and the run continues, rather than one
    `FinnhubClient.get_quote` call aborting everything after it.
    """
    class FakeClient:
        def get_quote(self, ticker):
            if ticker == "BAD":
                raise RuntimeError("no route to host")
            return SimpleNamespace(price=100.0, latest_trading_day="2026-09-25",
                                   source_mode="live")

    monkeypatch.setattr("finvet.mcp.finnhub.FinnhubClient", FakeClient)
    selected = [
        {"id": 1, "claim": "ACME Test Co's stock is trading above {threshold}",
         "recipe": {"ticker": "GOOD", "operator": "gt", "offset_pct": 5}},
        {"id": 2, "claim": "ACME Test Co's stock is trading above {threshold}",
         "recipe": {"ticker": "BAD", "operator": "gt", "offset_pct": 5}},
        {"id": 3, "claim": "ACME Test Co's revenue was $1 in fiscal 2024"},
    ]
    quotes = rg.prefetch_quotes(selected)
    assert set(quotes) == {"GOOD", "BAD"}
    assert isinstance(quotes["BAD"], RuntimeError)
    assert quotes["GOOD"].price == 100.0


def test_quote_error_record_matches_an_errored_row_shape():
    row = {"id": 2, "claim": "ACME Test Co's stock is trading above {threshold}",
           "category": "market/quote", "strength": "safe",
           "expected": {"verdict": "SUPPORTS"}, "gold_parse": {"value": None}}
    record = rg._quote_error_record(row, RuntimeError("no route to host"))
    assert record["actual"] is None
    assert record["error"] == "quote unavailable: no route to host"
    assert record["elapsed_s"] == 0.0
    assert record["id"] == 2
