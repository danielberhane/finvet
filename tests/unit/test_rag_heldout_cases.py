"""The held-out retrieval set keeps the shape its card promises.

`docs/eval/RAG_CARD.md` describes 50 cases: 30 known-item positives from 30
companies none of which the 70-case calibration set used, 12 off-topic queries
and 8 near-miss queries scoped to a filing the corpus does not hold. A case
file that drifts from that -- a positive with no expected id, a near-miss
without a filter, a calibration company creeping in -- would make the
published figures describe a different test than the one named. No database
is needed here; the live scoring is `scripts/eval_rag_heldout.py`.
"""
import json
from pathlib import Path

import pytest

CASES = Path("tests/accuracy/rag_heldout_cases.json")
CALIBRATION_TICKERS = {"AAPL", "AMZN", "MSFT", "NVDA", "TSLA"}


@pytest.fixture(scope="module")
def spec():
    return json.loads(CASES.read_text())


def test_the_set_is_fifty_cases_in_the_promised_proportions(spec):
    kinds = [c["kind"] for c in spec["cases"]]
    assert len(kinds) == 50
    assert kinds.count("positive") == 30
    assert kinds.count("off_topic") == 12
    assert kinds.count("near_miss") == 8


def test_ids_are_unique(spec):
    ids = [c["id"] for c in spec["cases"]]
    assert len(ids) == len(set(ids))


def test_every_positive_names_its_passage_and_a_fresh_company(spec):
    positives = [c for c in spec["cases"] if c["kind"] == "positive"]
    tickers = [c["ticker"] for c in positives]
    assert len(tickers) == len(set(tickers)), "one company per positive"
    assert not set(tickers) & CALIBRATION_TICKERS, "held out from the calibration companies"
    for c in positives:
        assert c["expected_evidence_ids"], c["id"]
        assert c["source"]["evidence_id"] in c["expected_evidence_ids"], c["id"]
        assert c["filters"] == {}, f"{c['id']}: positives measure retrieval by content, no section filter"


def test_negatives_carry_the_right_filters(spec):
    for c in spec["cases"]:
        if c["kind"] == "off_topic":
            assert c["filters"] == {} and c["expected_evidence_ids"] == [], c["id"]
        if c["kind"] == "near_miss":
            assert set(c["filters"]) & {"period_end", "filing_type"}, c["id"]
            assert c["expected_evidence_ids"] == [], c["id"]


def test_the_corpus_is_pinned(spec):
    assert len(spec["corpus_sha256"]) == 64
    assert spec["corpus"]["filings"] == len(spec["corpus"]["list"])
    assert spec["corpus"]["chunks"] == sum(f["chunks"] for f in spec["corpus"]["list"])
    assert spec["frozen"]
