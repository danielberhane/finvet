#!/usr/bin/env python
"""Exercise the Release-A behaviours against a running API and record evidence.

This is the data half of the manual UI gate. Each check drives a real request
through the live service and captures what came back, so the publication report
quotes observed behaviour rather than an intention. What it cannot do is judge
what the browser *renders*; those observations stay with a person, and the
report says which is which.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/release_gate_evidence.py
    PYTHONPATH=src .venv/bin/python scripts/release_gate_evidence.py --json out.json
"""

import argparse
import json
import os
from pathlib import Path

import httpx

API = os.environ.get("FINVET_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 300


def _verify(claim):
    r = httpx.post(f"{API}/verify", json={"claim": claim}, timeout=TIMEOUT)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})


def _audit(request_id):
    r = httpx.get(f"{API}/audit/{request_id}", timeout=60)
    return r.status_code, (r.json() if r.status_code == 200 else {})


def check_sec_observation_provenance():
    """A supported SEC claim must show exactly what Python compared."""
    _, body = _verify("Apple's total revenue was $391 billion in fiscal year 2024")
    obs = (body.get("metadata") or {}).get("trusted_observation") or {}
    return {
        "verdict": body.get("verdict"),
        "trusted_observation": obs,
        "has_identity": bool(obs.get("tool") and obs.get("concept")),
        "period_end": obs.get("period_end"),
    }


def check_market_observation_time():
    """A current quote must carry the source's own observation time."""
    _, body = _verify("Apple's stock is trading above $100")
    obs = (body.get("metadata") or {}).get("trusted_observation") or {}
    return {
        "verdict": body.get("verdict"),
        "observed_at": obs.get("observed_at"),
        "period_end": obs.get("period_end"),
        "source_time_present": obs.get("observed_at") is not None,
    }


def check_q4_is_declined():
    """Q4 derivation is unsupported and must say so."""
    _, body = _verify("Apple's Q4 2024 revenue was $94 billion")
    meta = body.get("metadata") or {}
    return {
        "verdict": body.get("verdict"),
        "limitation": meta.get("limitation"),
        "explanation": (body.get("explanation") or "")[:160],
    }


def check_macro_is_declined():
    """Macro numeric claims fail closed with a stated limitation."""
    _, body = _verify("US CPI inflation was 3.1 percent in July 2025")
    meta = body.get("metadata") or {}
    return {
        "verdict": body.get("verdict"),
        "limitation": meta.get("limitation"),
        "explanation": (body.get("explanation") or "")[:160],
    }


def check_memory_is_silent():
    """The default flow must offer no cached-result card."""
    r = httpx.post(f"{API}/memory-check",
                   json={"claim": "Apple's total revenue was $391 billion in fiscal year 2024"},
                   timeout=30)
    return {"status_code": r.status_code, "matches": r.json().get("matches")}


def check_rag_retrieval_subsystem():
    """The retrieval layer itself: scored chunks carrying filing identity.

    Driven at the tool boundary rather than through /verify, because that is
    the only place Release A reaches it -- see check_rag_is_not_reachable.
    """
    from finvet.tools.filing_search import search_filing_text

    result = search_filing_text.invoke(
        {"query": "supplier concentration risk", "ticker": "AAPL"})
    chunk = (result.get("chunks") or [{}])[0]
    return {
        "success": result.get("success"),
        "total_found": result.get("total_found"),
        "reason": result.get("reason"),
        "section": chunk.get("section"),
        "filing_type": chunk.get("filing_type"),
        "period_end": chunk.get("period_end"),
        "evidence_id_present": bool(chunk.get("evidence_id")),
        "hash_scope": chunk.get("hash_scope"),
    }


def check_rag_answers_a_filing_claim():
    """A claim about what a filing says reaches a verdict on retrieved text.

    This check used to assert the opposite. D14 recorded that filing retrieval
    was a tested subsystem with no route from a claim -- true when written, and
    the check existed so the boundary could not move unnoticed. Parser rule R2b
    moved it: a claim about what a filing *says* is now a `sec` claim with a
    null metric, which `verification_strategy_for` already routed to
    `filing_rag`.

    The guard did its job; this is the updated assertion. Nothing was loosened
    to get here -- the claim names no number, so the numeric guard has nothing
    to demand, and filing prose still cannot become a trusted observation.
    """
    _, body = _verify("Apple's annual report discusses risks from supplier concentration")
    meta = body.get("metadata") or {}
    rag = (meta.get("data_sources") or {}).get("rag") or {}
    evidence = (rag.get("evidence") or [{}])[0]
    return {
        "verdict": body.get("verdict"),
        "status": body.get("status"),
        "rag_used": bool(rag.get("used")),
        "chunks_retrieved": rag.get("chunks_retrieved"),
        "evidence_id_present": bool(evidence.get("evidence_id")),
        "filing_type": evidence.get("filing_type"),
    }


def check_a_qualitative_claim_is_not_refuted():
    """Non-corroboration is not contradiction.

    This returned REFUTES at 0.95 on ten retrieved articles, none of which
    mentioned the claim. A claim naming no value is now declined instead.
    """
    _, body = _verify("Apple's annual report discusses its plans to open a "
                      "theme park in Ohio")
    meta = body.get("metadata") or {}
    return {
        "verdict": body.get("verdict"),
        "status": body.get("status"),
        "limitation": meta.get("limitation"),
        "llm_original_verdict": meta.get("llm_original_verdict"),
    }


def check_audit_integrity_of_a_completed_run():
    _, body = _verify("Apple's total revenue was $391 billion in fiscal year 2024")
    request_id = body.get("request_id")
    _, audit = _audit(request_id)
    return {
        "request_id": request_id,
        "integrity_status": (audit.get("integrity") or {}).get("status"),
        "integrity_reason": (audit.get("integrity") or {}).get("reason"),
        "mismatches": (audit.get("integrity") or {}).get("mismatches"),
    }


def check_review_queue_exposes_status():
    r = httpx.get(f"{API}/reviews", timeout=30)
    rows = r.json() if r.status_code == 200 else []
    return {
        "status_code": r.status_code,
        "pending_count": len(rows),
        "review_statuses": sorted({row.get("review_status") for row in rows}),
    }


def _expect(result, **conditions):
    """Assertions as data. Returns the list of what did not hold."""
    failures = []
    for key, expected in conditions.items():
        actual = result.get(key)
        ok = expected(actual) if callable(expected) else actual == expected
        if not ok:
            # Only a callable has a meaningful __doc__. Reading it off a plain
            # value printed str's own class docstring as the expectation.
            want = (expected.__doc__ or "a predicate") if callable(expected) \
                else repr(expected)
            failures.append(f"{key}: expected {want}, got {actual!r}")
    return failures


# Each entry is (name, run, expectations). The script used to print whatever
# came back and exit 0 regardless, so a semantic failure -- the RAG scenario
# returning REJECTED with no evidence -- was recorded as evidence and read as
# a pass. A gate that cannot fail is a log.
CHECKS = [
    ("sec_observation_provenance", check_sec_observation_provenance,
     lambda r: _expect(r, verdict="SUPPORTS", has_identity=True)),
    ("market_observation_time", check_market_observation_time,
     lambda r: _expect(r, source_time_present=True, period_end=None)),
    ("q4_declined", check_q4_is_declined,
     lambda r: _expect(r, verdict="NOT_ENOUGH_INFO",
                       limitation="unsupported_q4_derivation")),
    ("macro_declined", check_macro_is_declined,
     lambda r: _expect(r, verdict="NOT_ENOUGH_INFO",
                       limitation="unsupported_metric")),
    ("memory_silent_by_default", check_memory_is_silent,
     lambda r: _expect(r, status_code=200, matches=[])),
    ("rag_retrieval_subsystem", check_rag_retrieval_subsystem,
     lambda r: _expect(r, success=True, evidence_id_present=True,
                       total_found=lambda n: isinstance(n, int) and n > 0,
                       section=lambda s: bool(s))),
    ("rag_answers_a_filing_claim", check_rag_answers_a_filing_claim,
     lambda r: _expect(r, verdict="SUPPORTS", status="success", rag_used=True,
                       evidence_id_present=True,
                       chunks_retrieved=lambda n: isinstance(n, int) and n > 0)),
    ("qualitative_claim_not_refuted", check_a_qualitative_claim_is_not_refuted,
     lambda r: _expect(r, verdict="NOT_ENOUGH_INFO", status="success",
                       limitation="non_corroboration_is_not_contradiction")),
    ("completed_run_verifies", check_audit_integrity_of_a_completed_run,
     lambda r: _expect(r, integrity_status="verified", mismatches=[])),
    ("review_queue_review_status", check_review_queue_exposes_status,
     lambda r: _expect(r, status_code=200)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--only", help="run one named check")
    args = parser.parse_args()

    try:
        httpx.get(f"{API}/health", timeout=5)
    except Exception as exc:
        print(f"API not reachable at {API}: {exc}")
        return 2

    results = {}
    failures = {}
    for name, fn, expect in CHECKS:
        if args.only and name != args.only:
            continue
        print(f"\n--- {name} ---")
        try:
            results[name] = fn()
            problems = expect(results[name])
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            problems = [f"raised {type(exc).__name__}: {exc}"]
        print(json.dumps(results[name], indent=2, default=str))
        if problems:
            failures[name] = problems
            for problem in problems:
                print(f"  FAIL  {problem}")
        else:
            print("  ok")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(
            {"results": results, "failures": failures}, indent=2, default=str))
        print(f"\nwritten to {args.json_path}")

    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    for name, problems in failures.items():
        print(f"  {name}: {'; '.join(problems)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
