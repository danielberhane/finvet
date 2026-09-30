"""The union-collision checks (`assess`, `_fact_key`), exercised without the
network and without touching either golden set on disk.
"""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_golden_u.py"
_spec = importlib.util.spec_from_file_location("build_golden_u", SCRIPT)
bu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bu)


def _row(rid, claim, strength="strict", category="sec/xbrl", verdict="SUPPORTS",
         limitation=None, ticker="AAPL", metric="revenue", period="FY2024",
         operator="eq", value=391_000_000_000, claim_type="sec", tags=None, recipe=None):
    row = {
        "id": rid,
        "claim": claim,
        "category": category,
        "strength": strength,
        "expected": {"verdict": verdict, "limitation": limitation, "sources": []},
        "ground_truth": "g",
        "source": "s",
        "gold_parse": {"claim_type": claim_type, "ticker": ticker, "metric": metric,
                       "operator": operator, "value": value, "period": period,
                       "reject_reason": None},
    }
    if tags is not None:
        row["tags"] = tags
    if recipe is not None:
        row["recipe"] = recipe
    return row


def setup_function(_):
    # WARNINGS is module-level and accumulates across calls to assess().
    bu.WARNINGS.clear()


def test_recipe_rows_same_ticker_different_offset_no_failure():
    rows = [
        _row(1301, "XOM's stock price is near its recent trading level", claim_type="market",
             value=None, metric="price", period=None, operator="gte",
             recipe={"ticker": "XOM", "offset_pct": -10}),
        _row(1302, "XOM's stock price is near its recent trading level", claim_type="market",
             value=None, metric="price", period=None, operator="lte",
             recipe={"ticker": "XOM", "offset_pct": 10}),
    ]
    assert bu.assess(rows) == []


def test_two_nonrecipe_rows_identical_normalised_text_fails():
    rows = [
        _row(1, "Apple's total revenue was $391 billion in fiscal year 2024"),
        _row(2, "Apple's total revenue was $391 billion in fiscal year 2024!"),
    ]
    fails = bu.assess(rows)
    assert any("same claim text" in f for f in fails)


def test_two_strict_rows_same_fact_key_fails():
    rows = [
        _row(1, "Apple's total revenue was $391 billion in fiscal year 2024"),
        _row(2, "Apple booked roughly $391B of revenue for FY2024"),
    ]
    fails = bu.assess(rows)
    assert any("same fact asked twice" in f for f in fails)


def test_same_fact_on_two_observe_rows_warns_not_fails():
    rows = [
        _row(1, "Chevron's Item 3 cites a $1.0 million or more threshold", strength="observe",
             verdict=None, ticker="CVX", metric="fine_amount", period=None,
             claim_type="news"),
        _row(2, "Chevron's Legal Proceedings names the same $1.0 million threshold",
             strength="observe", verdict=None, ticker="CVX", metric="fine_amount",
             period=None, claim_type="news"),
    ]
    fails = bu.assess(rows)
    assert fails == []
    assert any("observe rows share one fact" in w for w in bu.WARNINGS)


def test_twin_and_original_sharing_a_fact_no_failure():
    rows = [
        _row(1, "Apple's total revenue was $391 billion in fiscal year 2024"),
        _row(2, "Apple's revenue came in around $391 billion for FY2024",
             tags={"seed": "twin", "twin_of": 1}),
    ]
    assert bu.assess(rows) == []


class TestFactKey:
    def test_reject_row_has_no_fact_key(self):
        row = _row(1, "some ambiguous claim", claim_type="reject", ticker=None,
                    metric=None, period=None, operator=None, value=None)
        assert bu._fact_key(row) is None

    def test_two_rows_share_a_fact_key(self):
        a = _row(1, "text a")
        b = _row(2, "text b")
        assert bu._fact_key(a) == bu._fact_key(b)
