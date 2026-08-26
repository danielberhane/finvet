#!/usr/bin/env python
"""Regenerate the Release-A RAG calibration manifest from the live corpus.

The manifest is the evidence behind `RAG_MIN_VECTOR_SIMILARITY`. It records,
for every labelled case, what retrieval actually returned — the passage's
identity, both arms' raw signals and positions, and the fused score — so the
threshold can be re-derived rather than taken on trust.

Two kinds of negative are recorded, and the second is the one that matters:

  * **off_topic** — a query unrelated to any filing. Easy, and the only kind
    the previous artifact held.
  * **near_miss** — a query that is *exactly* on topic but scoped to a period
    or form the corpus does not hold for that issuer. A relevance floor cannot
    catch these; the period and form filters must. A calibration set made only
    of sourdough recipes proves the floor works on questions nobody would ask.

Run after any re-ingestion; the corpus checksum ties the manifest to the exact
contents it describes.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_rag_manifest.py
    PYTHONPATH=src .venv/bin/python scripts/build_rag_manifest.py --out /tmp/m.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from finvet.config.constants import (  # noqa: E402
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    RAG_MIN_VECTOR_SIMILARITY,
)
from finvet.config.database import get_db_session  # noqa: E402
from finvet.rag.service import get_rag_service  # noqa: E402

CASES = Path("tests/accuracy/rag_relevance_cases.json")
OUT = Path("tests/accuracy/rag_release_a_manifest.json")

# On-topic queries aimed at a period or form the issuer has no filing for. The
# text is relevant; the scope is wrong. Retrieval must return nothing.
NEAR_MISSES = [
    ("AAPL", "risks from supplier concentration and single-source components",
     "10-K", "2019-09-28", "wrong period: no FY2019 filing is indexed"),
    ("AAPL", "management discussion of quarterly net sales growth",
     "10-Q", "2021-06-26", "wrong period: no 2021 quarter is indexed"),
    ("MSFT", "intelligent cloud segment revenue and operating income",
     "10-K", "2020-06-30", "wrong period: no FY2020 filing is indexed"),
    ("NVDA", "data center demand and supply constraints",
     "10-Q", "2022-10-30", "wrong period: no 2022 quarter is indexed"),
    ("TSLA", "vehicle deliveries and production ramp",
     "10-K", "2020-12-31", "wrong period: no FY2020 filing is indexed"),
    ("AMZN", "AWS segment results and operating margin",
     "10-Q", "2020-06-30", "wrong period: no 2020 quarter is indexed"),
    ("AAPL", "annual risk factors discussion",
     "10-Q", "2025-09-27", "wrong form: that period is a 10-K, not a 10-Q"),
    ("MSFT", "quarterly management discussion",
     "10-K", "2025-09-30", "wrong form: that period is a 10-Q, not a 10-K"),
    ("TSLA", "annual risk factors discussion",
     "10-Q", "2025-12-31", "wrong form: that period is a 10-K, not a 10-Q"),
    ("NVDA", "quarterly results of operations",
     "10-K", "2025-07-27", "wrong form: that period is a 10-Q, not a 10-K"),
]


def corpus_checksum() -> str:
    with get_db_session() as session:
        rows = session.execute(text(
            "SELECT evidence_id FROM filing_chunks ORDER BY evidence_id"
        )).fetchall()
    return hashlib.sha256(
        "\n".join(r[0] or "" for r in rows).encode("utf-8")).hexdigest()


def top_vector_similarity(query: str, ticker: str) -> float:
    """The best cosine score the dense arm can reach for this query.

    This is the number the threshold is compared against, so it must come from
    the dense arm directly. Reading it off the top RRF result measures
    something else: fusion can rank a keyword-only chunk first, and that chunk
    carries a vector score of 0 — which would record a strong positive as a
    complete miss.
    """
    from finvet.rag.service import _embed_single

    embedding = _embed_single(query)
    with get_db_session() as session:
        row = session.execute(text(
            "SELECT max(1 - (embedding <=> CAST(:v AS vector))) "
            "FROM filing_chunks WHERE ticker = :t"),
            {"v": str(embedding), "t": ticker.upper()}).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def _record(rag, *, query, ticker, label, kind, expected_form="",
            expected_period="", note=""):
    """Run one case and record what retrieval actually returned."""
    kwargs = {"query": query, "ticker": ticker, "top_k": 5,
              "min_vector_similarity": 0.0}   # unfiltered, to observe the score
    if expected_period:
        kwargs["period_end"] = expected_period
    if expected_form:
        kwargs["filing_type"] = expected_form

    results = rag.search(**kwargs)
    top = results[0] if results else None

    return {
        "query": query,
        "ticker": ticker,
        "label": label,
        "case_kind": kind,
        "expected_form": expected_form or (top.filing_type if top else ""),
        "expected_period": expected_period or (top.period_end if top else ""),
        "expected_part": (top.part if top else None),
        "expected_item": (top.item_number if top else ""),
        "evidence_id": (top.evidence_id if top else ""),
        # The calibration signal: the dense arm's best score for this query,
        # independent of how fusion happened to rank the results.
        "vector_similarity": round(top_vector_similarity(query, ticker), 6),
        "top_rrf_vector_similarity": (top.vector_similarity if top else 0.0),
        "vector_rank": (top.vector_rank if top else None),
        "keyword_score": (top.keyword_score if top else 0.0),
        "keyword_rank": (top.keyword_rank if top else None),
        "rrf_score": (top.rrf_score if top else 0.0),
        "results_returned": len(results),
        # A human judgment, not a model's: the label is why this case is in the
        # set, and the note says what a reader should conclude from it.
        "human_relevance": (
            "relevant passage expected" if label == "positive"
            else note or "no relevant passage should be retrieved"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    labelled = json.loads(CASES.read_text())["cases"]
    rag = get_rag_service()

    cases = []
    for case in labelled:
        cases.append(_record(
            rag, query=case["query"], ticker=case["ticker"],
            label=case["label"],
            kind="on_topic" if case["label"] == "positive" else "off_topic"))
        print(f"  {case['label']:<9} {case['ticker']} "
              f"{cases[-1]['vector_similarity']:.4f}  {case['query'][:52]}")

    for ticker, query, form, period, note in NEAR_MISSES:
        cases.append(_record(
            rag, query=query, ticker=ticker, label="negative",
            kind="near_miss", expected_form=form, expected_period=period,
            note=note))
        print(f"  near_miss {ticker} returned={cases[-1]['results_returned']}"
              f"  {note}")

    positives = [c["vector_similarity"] for c in cases if c["label"] == "positive"]
    off_topic = [c["vector_similarity"] for c in cases
                 if c["label"] == "negative" and c["case_kind"] == "off_topic"]
    near_miss = [c for c in cases if c["case_kind"] == "near_miss"]

    manifest = {
        "manifest_version": 1,
        "embedding_model": EMBEDDING_MODEL.split(":")[0],
        "embedding_dimensions": EMBEDDING_DIMS,
        "corpus_sha256": corpus_checksum(),
        "production_threshold": RAG_MIN_VECTOR_SIMILARITY,
        "selection_rule": (
            "no labelled negative accepted while all labelled positives retained"),
        "separation_evidence": {
            "highest_scoring_negative": round(max(off_topic), 6),
            "lowest_scoring_positive": round(min(positives), 6),
            "tightest_separating_value": round(max(off_topic) + 0.0001, 6),
            "selected_operating_point": RAG_MIN_VECTOR_SIMILARITY,
            "rationale": (
                "An interior point in the separating gap, chosen to keep margin "
                "on both sides. The tightest separating value sits immediately "
                "above the worst negative and admits it under any corpus drift."),
        },
        "near_miss_evidence": {
            "cases": len(near_miss),
            "returned_nothing": sum(1 for c in near_miss
                                    if c["results_returned"] == 0),
            "note": (
                "Wrong-period and wrong-form queries are on topic; the relevance "
                "floor cannot exclude them and the scope filters must. Counted "
                "separately because they measure a different mechanism."),
        },
        "cases": cases,
    }

    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")

    sep = manifest["separation_evidence"]
    print(f"\n  positives: {len(positives)}  min {sep['lowest_scoring_positive']}")
    print(f"  off-topic: {len(off_topic)}  max {sep['highest_scoring_negative']}")
    print(f"  near-miss: {len(near_miss)}  "
          f"{manifest['near_miss_evidence']['returned_nothing']} returned nothing")
    print(f"  threshold: {RAG_MIN_VECTOR_SIMILARITY}")
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
