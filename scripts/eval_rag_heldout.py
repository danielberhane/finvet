#!/usr/bin/env python
"""Score FinVet's filing-text retrieval on the frozen held-out set.

Reads `tests/accuracy/rag_heldout_cases.json`, runs every query through the
production search (`RAGService.search`, top five, the same floor and fusion
the SEC agent gets), and scores by chunk id. No model takes part in scoring.

Three kinds of case, three questions:

  positive   the passage exists -- is one of its ids in the top five, and how
             high? (recall@5, MRR, nDCG@5)
  off_topic  nothing relevant exists -- does the floor return nothing?
  near_miss  the topic exists but the filter names a filing the corpus does
             not hold -- do the period and form filters return nothing?

The case file pins the corpus by a hash of every chunk id. A different corpus
is refused, because a retrieval number is a property of the corpus it was
measured on.

The metrics are the textbook definitions (binary relevance, one query at a
time, averaged): recall@k = any expected id in the top k; reciprocal rank =
1 / rank of the first expected id, 0 if none; nDCG@k with gain 1 and log2
discount, ideal DCG from the number of expected ids. The implementation was
checked against ranx 0.3.21 on this set; the library is not a dependency
because it pulls in plotting and JIT packages the project does not need.

Usage:
    set -a; source .env; set +a
    PYTHONPATH=src .venv/bin/python scripts/eval_rag_heldout.py
    PYTHONPATH=src .venv/bin/python scripts/eval_rag_heldout.py --out docs/eval/runs/rag-heldout-2026-10-01.json
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from finvet.audit.database import get_db_session  # noqa: E402
from finvet.config.constants import (  # noqa: E402
    EMBEDDING_MODEL,
    RAG_MIN_VECTOR_SIMILARITY,
)
from finvet.rag.service import _embed_single, get_rag_service  # noqa: E402

CASES = Path("tests/accuracy/rag_heldout_cases.json")


def corpus_checksum() -> str:
    with get_db_session() as session:
        ids = [r[0] for r in session.execute(
            text("SELECT evidence_id FROM filing_chunks ORDER BY evidence_id"))]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def embedder_is_live() -> bool:
    """The vector arm falls back to keywords only when the embedder is down,
    and the search still reports success. A score taken in that state would
    describe half the retriever. Refuse rather than measure it."""
    try:
        return len(_embed_single("probe")) > 0
    except Exception:
        return False


def score_positive(expected: list[str], returned: list[str], k: int) -> dict:
    top = returned[:k]
    ranks = [i + 1 for i, rid in enumerate(top) if rid in expected]
    first = ranks[0] if ranks else None
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return {
        "hit": first is not None,
        "first_rank": first,
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--out", default=None, help="artifact path (default: docs/eval/runs/rag-heldout-<date>.json)")
    parser.add_argument("--allow-corpus-drift", action="store_true",
                        help="score anyway when the corpus hash differs (the result is then not comparable)")
    args = parser.parse_args()

    spec = json.loads(Path(args.cases).read_text())
    k = int(spec.get("top_k", 5))

    live = corpus_checksum()
    if live != spec["corpus_sha256"]:
        msg = (f"corpus hash {live[:12]} differs from the frozen {spec['corpus_sha256'][:12]}; "
               "the set was labelled against a different index")
        if not args.allow_corpus_drift:
            print(f"REFUSED: {msg}", file=sys.stderr)
            return 2
        print(f"WARNING: {msg}", file=sys.stderr)

    if not embedder_is_live():
        print(f"REFUSED: the embedder ({EMBEDDING_MODEL}) is not answering; the vector arm would be "
              "silently absent", file=sys.stderr)
        return 2

    rag = get_rag_service()
    per_case = []
    for case in spec["cases"]:
        results = rag.search(case["query"], ticker=case["ticker"], top_k=k, **case.get("filters", {}))
        returned = [r.evidence_id for r in results]
        record = {
            "id": case["id"], "kind": case["kind"], "ticker": case["ticker"],
            "filters": case.get("filters", {}),
            "returned": [{"evidence_id": r.evidence_id, "section": r.section,
                          "filing_type": r.filing_type, "period_end": r.period_end,
                          "rrf_score": round(float(r.rrf_score), 6),
                          "vector_similarity": round(float(r.vector_similarity), 4),
                          "keyword_score": round(float(r.keyword_score), 4)} for r in results],
        }
        if case["kind"] == "positive":
            record.update(score_positive(case["expected_evidence_ids"], returned, k))
        else:
            record["rejected"] = len(returned) == 0
        per_case.append(record)

    pos = [c for c in per_case if c["kind"] == "positive"]
    off = [c for c in per_case if c["kind"] == "off_topic"]
    near = [c for c in per_case if c["kind"] == "near_miss"]
    summary = {
        "positives": len(pos),
        "recall_at_k": sum(c["hit"] for c in pos) / len(pos),
        "hits": sum(c["hit"] for c in pos),
        "mrr": sum(c["reciprocal_rank"] for c in pos) / len(pos),
        "ndcg_at_k": sum(c["ndcg"] for c in pos) / len(pos),
        "off_topic_rejected": f"{sum(c['rejected'] for c in off)}/{len(off)}",
        "near_miss_rejected": f"{sum(c['rejected'] for c in near)}/{len(near)}",
        "k": k,
    }

    artifact = {
        "kind": "rag_heldout",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases_file": args.cases,
        "cases_frozen": spec["frozen"],
        "corpus_sha256": live,
        "corpus_matches_frozen": live == spec["corpus_sha256"],
        "embedding_model": EMBEDDING_MODEL,
        "vector_floor": RAG_MIN_VECTOR_SIMILARITY,
        "summary": summary,
        "cases": per_case,
    }
    out = Path(args.out) if args.out else Path("docs/eval/runs") / f"rag-heldout-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1))

    print(f"recall@{k}  {summary['hits']}/{summary['positives']} = {summary['recall_at_k']:.3f}")
    print(f"MRR        {summary['mrr']:.3f}")
    print(f"nDCG@{k}    {summary['ndcg_at_k']:.3f}")
    print(f"off-topic rejected  {summary['off_topic_rejected']}")
    print(f"near-miss rejected  {summary['near_miss_rejected']}")
    misses = [c for c in pos if not c["hit"]] + [c for c in off + near if not c["rejected"]]
    for c in misses:
        got = ", ".join(f"{r['section']}#{r['evidence_id'][:8]}" for r in c["returned"][:3]) or "nothing"
        print(f"  MISS {c['id']} ({c['ticker']}): returned {got}")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
