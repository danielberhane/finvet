"""XBRL retrieval accuracy — does FinVet return the number the company filed?

One question per case: given a filing (CIK + accession) and an XBRL concept,
compare what `SECEdgarClient.get_financials` returns against the value read from
SEC's own primary source.

No LLM calls and no agent — this isolates the retrieval layer, so a failure is
unambiguous. It is the regression suite for the consolidated/period fact
selection in `mcp/sec_edgar.py`.

Ground truth comes from the fine-tuned claim-parser project's real-sourced
held-out set, whose `provenance` block carries `xbrl_fact`, `accession`,
`period_end` and `source_value_exact`. That set is contamination-sensitive and
is NOT redistributed with FinVet — it is read by path from the sibling repo.

Usage:
    python -m finvet.eval.xbrl_retrieval --limit 20
    python -m finvet.eval.xbrl_retrieval --concept GrossProfit --report out.json
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..mcp.sec_edgar import (
    CONCEPTS_BY_TYPE,
    CONSOLIDATION_SENSITIVE_CONCEPTS,
    SECEdgarClient,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_GOLD = (
    Path.home()
    / "Projects/Active/claim_parser_fine-tuned/data/clean/heldout_real_sourced.jsonl"
)

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_RETURNED = "NOT_RETURNED"
STATUS_UNSUPPORTED_CONCEPT = "UNSUPPORTED_CONCEPT"
STATUS_ERROR = "ERROR"
STATUS_PENDING = "PENDING"

# Both sides quote the same XBRL fact, so any real difference is a retrieval
# error, not rounding. The window only absorbs float representation noise.
MATCH_TOLERANCE_PCT = 0.01

# Concept -> the get_financials statement_type that requests it. Derived from
# CONCEPTS_BY_TYPE so the harness asks for exactly what the SEC agent asks for;
# a concept absent here is one FinVet never requests at all.
_CONCEPT_TO_STATEMENT: Dict[str, str] = {
    concept: statement
    for statement, concepts in CONCEPTS_BY_TYPE.items()
    for concept in concepts
}


def normalize_fact(xbrl_fact: str) -> str:
    """`us-gaap:GrossProfit` -> `GrossProfit`."""
    return xbrl_fact.split(":")[-1]


def statement_type_for(concept: str) -> Optional[str]:
    """Which statement request yields this concept, or None if FinVet never asks."""
    return _CONCEPT_TO_STATEMENT.get(concept)


def period_kind_for_frame(frame: Optional[str]) -> str:
    """`CY2024Q2` -> quarterly, `CY2024` -> annual.

    A period end alone does not identify a fact: a 10-Q tags the three-month
    quarter and the cumulative year-to-date figure with the same end date. The
    frame carries the duration, so the harness can request the right one.
    """
    if frame and re.search(r"Q[1-4]$", frame):
        return "quarterly"
    return "annual"


def compare_values(actual: float, expected: float) -> tuple:
    """Return (status, absolute delta, percentage delta)."""
    delta = abs(actual - expected)
    divisor = max(abs(actual), abs(expected))
    pct = (delta / divisor * 100) if divisor > 0 else 0.0
    status = STATUS_PASS if pct <= MATCH_TOLERANCE_PCT else STATUS_FAIL
    return status, delta, pct


@dataclass
class RetrievalCase:
    """One row: the question, and once run, the answer."""

    row_id: int
    claim: str
    cik: str
    accession: str
    concept: str
    expected_value: float
    expected_period_end: str
    period_kind: str
    statement_type: Optional[str]
    consolidation_sensitive: bool
    status: str = STATUS_PENDING
    finvet_value: Optional[float] = None
    delta: Optional[float] = None
    delta_pct: Optional[float] = None
    returned_period_end: Optional[str] = None
    consolidated_flag: Optional[bool] = None
    context_ref: Optional[str] = None
    concepts_returned: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None


def build_case(row: Dict[str, Any]) -> RetrievalCase:
    """Turn a gold row into a case, marking concepts FinVet cannot request."""
    prov = row["provenance"]
    concept = normalize_fact(prov["xbrl_fact"])
    statement = statement_type_for(concept)
    return RetrievalCase(
        row_id=row["id"],
        claim=row.get("input", ""),
        cik=prov["cik"],
        accession=prov["accession"],
        concept=concept,
        expected_value=float(prov["source_value_exact"]),
        expected_period_end=prov.get("period_end", ""),
        period_kind=period_kind_for_frame(prov.get("xbrl_frame")),
        statement_type=statement,
        consolidation_sensitive=concept in CONSOLIDATION_SENSITIVE_CONCEPTS,
        status=STATUS_PENDING if statement else STATUS_UNSUPPORTED_CONCEPT,
    )


def load_cases(
    gold_path: Path,
    limit: Optional[int] = None,
    concept: Optional[str] = None,
) -> List[RetrievalCase]:
    """Select scoreable rows: those carrying both an XBRL fact and a source value.

    File order is preserved so a subset run is reproducible.
    """
    cases: List[RetrievalCase] = []
    with open(gold_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prov = row.get("provenance") or {}
            if not prov.get("xbrl_fact") or prov.get("source_value_exact") is None:
                continue
            case = build_case(row)
            if concept and case.concept != concept:
                continue
            cases.append(case)
            if limit and len(cases) >= limit:
                break
    return cases


def run_case(case: RetrievalCase, client: SECEdgarClient) -> RetrievalCase:
    """Ask FinVet for the concept and score what comes back."""
    if case.statement_type is None:
        return case

    start = time.time()
    try:
        items = client.get_financials(
            identifier=case.cik,
            accession_number=case.accession,
            statement_type=case.statement_type,
            period=case.period_kind,
            period_end=case.expected_period_end,
        )
        hit = next((i for i in items if i.line_item == case.concept), None)
        if hit is None:
            case.status = STATUS_NOT_RETURNED
            case.concepts_returned = [i.line_item for i in items]
        else:
            case.status, case.delta, case.delta_pct = compare_values(
                hit.value, case.expected_value
            )
            case.finvet_value = hit.value
            case.returned_period_end = hit.period_end
            case.consolidated_flag = hit.consolidated
            case.context_ref = hit.context_ref
    except Exception as e:  # a transport failure is a result, not a crash
        case.status = STATUS_ERROR
        case.error = f"{type(e).__name__}: {e}"
        logger.warning(f"row {case.row_id} ({case.concept}) errored: {e}")

    case.elapsed_ms = int((time.time() - start) * 1000)
    return case


def summarize(cases: List[RetrievalCase]) -> Dict[str, Any]:
    """Aggregate. Unsupported concepts are a coverage gap, not a retrieval failure,
    so they are reported separately and excluded from the pass rate."""
    scored = [c for c in cases if c.status not in (STATUS_UNSUPPORTED_CONCEPT, STATUS_PENDING)]
    passed = [c for c in scored if c.status == STATUS_PASS]
    sensitive = [c for c in scored if c.consolidation_sensitive]

    by_status: Dict[str, int] = {}
    for c in cases:
        by_status[c.status] = by_status.get(c.status, 0) + 1

    by_concept: Dict[str, Dict[str, int]] = {}
    for c in scored:
        entry = by_concept.setdefault(c.concept, {"scored": 0, "passed": 0})
        entry["scored"] += 1
        entry["passed"] += 1 if c.status == STATUS_PASS else 0

    return {
        "total": len(cases),
        "scored": len(scored),
        "passed": len(passed),
        "failed": len(scored) - len(passed),
        "pass_rate": (len(passed) / len(scored)) if scored else 0.0,
        "unsupported_concept": by_status.get(STATUS_UNSUPPORTED_CONCEPT, 0),
        "by_status": by_status,
        "by_concept": by_concept,
        "consolidation_sensitive": {
            "scored": len(sensitive),
            "passed": sum(1 for c in sensitive if c.status == STATUS_PASS),
        },
    }


def _print_report(cases: List[RetrievalCase], summary: Dict[str, Any]) -> None:
    failures = [c for c in cases if c.status in (STATUS_FAIL, STATUS_NOT_RETURNED, STATUS_ERROR)]
    if failures:
        print(f"\n{'-' * 78}\nFAILURES ({len(failures)})\n{'-' * 78}")
        for c in failures:
            print(f"\n  row {c.row_id}  {c.concept}  [{c.status}]")
            print(f"    {c.claim[:70]}")
            if c.status == STATUS_FAIL:
                print(f"    SEC filed : {c.expected_value:>20,.0f}   period {c.expected_period_end}")
                print(f"    FinVet got: {c.finvet_value:>20,.0f}   period {c.returned_period_end}")
                print(f"    delta     : {c.delta:>20,.0f}   ({c.delta_pct:.2f}%)"
                      f"   consolidated={c.consolidated_flag}")
            elif c.status == STATUS_NOT_RETURNED:
                print(f"    concept absent from the {c.statement_type} response")
            else:
                print(f"    {c.error}")

    print(f"\n{'=' * 78}\nXBRL RETRIEVAL ACCURACY\n{'=' * 78}")
    print(f"  cases loaded        : {summary['total']}")
    print(f"  scored              : {summary['scored']}")
    print(f"  passed              : {summary['passed']}")
    print(f"  failed              : {summary['failed']}")
    print(f"  pass rate           : {summary['pass_rate']:.1%}")
    cs = summary["consolidation_sensitive"]
    if cs["scored"]:
        print(f"  consolidation-sensitive: {cs['passed']}/{cs['scored']} "
              f"({cs['passed'] / cs['scored']:.1%})")
    if summary["unsupported_concept"]:
        print(f"  unsupported concepts: {summary['unsupported_concept']}"
              f"  (FinVet never requests these — coverage gap, not a failure)")
    if summary["by_concept"]:
        print("\n  per concept:")
        for concept, s in sorted(summary["by_concept"].items(), key=lambda kv: -kv[1]["scored"]):
            print(f"    {concept:<54} {s['passed']:>3}/{s['scored']:<3}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score FinVet's XBRL retrieval against SEC primary-source values"
    )
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                    help="Real-sourced gold JSONL (default: sibling claim_parser repo)")
    ap.add_argument("--limit", type=int, help="Score only the first N cases")
    ap.add_argument("--concept", help="Only score this XBRL concept (bare name)")
    ap.add_argument("--report", type=Path, help="Write the full JSON report here")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="Seconds between cases, for SEC fair access (default: 0.2)")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Gold set not found: {args.gold}\n"
              f"This harness needs the claim-parser project's real-sourced held-out set.")
        return 2

    cases = load_cases(args.gold, limit=args.limit, concept=args.concept)
    if not cases:
        print("No scoreable cases matched.")
        return 2

    print(f"Scoring {len(cases)} case(s) against {args.gold.name}\n")
    client = SECEdgarClient()
    try:
        for i, case in enumerate(cases, 1):
            run_case(case, client)
            print(f"  [{i}/{len(cases)}] row {case.row_id:<5} {case.concept:<52} {case.status}")
            if args.delay:
                time.sleep(args.delay)
    finally:
        client.close()

    summary = summarize(cases)
    _print_report(cases, summary)

    if args.report:
        args.report.write_text(json.dumps(
            {"gold": str(args.gold), "summary": summary,
             "cases": [asdict(c) for c in cases]}, indent=2))
        print(f"  report written to {args.report}\n")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
