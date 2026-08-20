"""Reject classification — what does FinVet do with a claim it should not verify?

Two independent mechanisms can stop a claim, and they are not interchangeable:
the input guardrail raises before the parser ever sees the text, or the parser
classifies the claim as "reject" and the graph routes it to reject_handler.
Both are correct outcomes; a single pass rate would hide which layer is carrying
the load, and therefore what breaks if one is disabled.

Every row lands in exactly one outcome:

    gold reject     + guard raised            -> blocked_by_guard    (correct)
    gold reject     + parser said reject      -> rejected_by_parser  (correct)
    gold reject     + parser said sec/mkt/nws -> missed              (FAILURE)
    gold non-reject + rejected either way     -> false_reject        (FAILURE)
    gold non-reject + routed to an agent      -> accepted            (correct)

Only input_guardrails and claim_parser are run. Those two make the entire reject
decision, so invoking the full pipeline would spend agent calls on claims that
get rejected anyway. A "missed" row is one that *would* have run the full
pipeline in production — this harness stops short of doing so.

Ground truth is the fine-tuned claim-parser project's gold sets, where the label
already exists as claim_type + reject_reason. Those sets are contamination-
sensitive and are NOT redistributed with FinVet — they are read by path from the
sibling repo.

Usage:
    python -m finvet.eval.reject_classification --claim-type reject --limit 20
    python -m finvet.eval.reject_classification --report out.json
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..graph.nodes import claim_parser, input_guardrails
from ..utils.exceptions import GuardrailViolation
from ..utils.logging import get_logger

logger = get_logger(__name__)

_GOLD_DIR = Path.home() / "Projects/Active/claim_parser_fine-tuned/data/clean"
DEFAULT_GOLD = _GOLD_DIR / "heldout_real_sourced.jsonl"
SYNTHETIC_GOLD = _GOLD_DIR / "test.jsonl"

OUTCOME_BLOCKED_BY_GUARD = "blocked_by_guard"
OUTCOME_REJECTED_BY_PARSER = "rejected_by_parser"
OUTCOME_MISSED = "missed"
OUTCOME_FALSE_REJECT = "false_reject"
OUTCOME_ACCEPTED = "accepted"
OUTCOME_ERROR = "error"
OUTCOME_PENDING = "pending"

# Outcomes in which the claim was stopped, however it was stopped.
_REJECTED_OUTCOMES = frozenset({OUTCOME_BLOCKED_BY_GUARD, OUTCOME_REJECTED_BY_PARSER})

# CompositeGuardProvider reports provider="composite" once any member fails, so
# the violation type is the only thing that identifies the layer that fired.
_REGEX_VIOLATIONS = frozenset({
    "INJECTION_DETECTED",
    "PII_DETECTED",
    "CLAIM_TOO_SHORT",
    "CLAIM_TOO_LONG",
    "UNSUPPORTED_LANGUAGE",
})
_LLAMA_GUARD_VIOLATIONS = frozenset({"LLAMA_GUARD_UNSAFE"})


def guard_layer_for(violation_type: Optional[str]) -> str:
    """Which guard provider raised, inferred from the violation type."""
    if violation_type in _REGEX_VIOLATIONS:
        return "regex"
    if violation_type in _LLAMA_GUARD_VIOLATIONS:
        return "llama_guard"
    return "unknown"


def classify_outcome(
    gold_claim_type: str,
    guard_blocked: bool,
    parsed_claim_type: Optional[str],
) -> str:
    """Cross the gold label with what the pipeline actually did.

    Routing errors among sec/market/news are deliberately not scored here — this
    harness measures the reject decision only.
    """
    stopped = guard_blocked or parsed_claim_type == "reject"

    if gold_claim_type == "reject":
        if guard_blocked:
            return OUTCOME_BLOCKED_BY_GUARD
        if parsed_claim_type == "reject":
            return OUTCOME_REJECTED_BY_PARSER
        return OUTCOME_MISSED

    return OUTCOME_FALSE_REJECT if stopped else OUTCOME_ACCEPTED


@dataclass
class RejectCase:
    """One row: the claim and its gold label, and once run, what FinVet did."""

    row_id: int
    claim: str
    gold_claim_type: str
    gold_reject_reason: Optional[str]
    source_type: Optional[str]
    outcome: str = OUTCOME_PENDING
    violation_type: Optional[str] = None
    guard_layer: Optional[str] = None
    guard_categories: List[str] = field(default_factory=list)
    parsed_claim_type: Optional[str] = None
    parsed_reject_reason: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None


def build_case(row: Dict[str, Any]) -> RejectCase:
    """Turn a gold row into a case. The label is already in the data."""
    gold = row.get("gold") or {}
    prov = row.get("provenance") or {}
    return RejectCase(
        row_id=row["id"],
        claim=row.get("input", ""),
        gold_claim_type=gold.get("claim_type", ""),
        gold_reject_reason=gold.get("reject_reason"),
        source_type=prov.get("source_type"),
    )


def load_cases(
    gold_path: Path,
    limit: Optional[int] = None,
    claim_type: Optional[str] = None,
) -> List[RejectCase]:
    """Load gold rows. Non-reject rows are included by default so that false
    rejects are measurable; file order is preserved so subsets are reproducible.
    """
    cases: List[RejectCase] = []
    with open(gold_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case = build_case(json.loads(line))
            if claim_type and case.gold_claim_type != claim_type:
                continue
            cases.append(case)
            if limit and len(cases) >= limit:
                break
    return cases


def run_case(case: RejectCase) -> RejectCase:
    """Run the two nodes that make the reject decision, and nothing further."""
    start = time.time()
    state: Dict[str, Any] = {
        "claim_raw": case.claim,
        "request_id": f"reject-eval-{case.row_id}",
        "user_id": "reject-eval",
    }

    try:
        state.update(input_guardrails(state))
    except GuardrailViolation as gv:
        case.violation_type = gv.violation_type
        case.guard_layer = guard_layer_for(gv.violation_type)
        case.guard_categories = list((gv.details or {}).get("categories") or [])
        case.outcome = classify_outcome(case.gold_claim_type, True, None)
        case.elapsed_ms = int((time.time() - start) * 1000)
        return case
    except Exception as e:
        case.outcome = OUTCOME_ERROR
        case.error = f"guard: {type(e).__name__}: {e}"
        case.elapsed_ms = int((time.time() - start) * 1000)
        logger.warning(f"row {case.row_id} guard errored: {e}")
        return case

    try:
        parsed = claim_parser(state)["parsed_claim"]
        case.parsed_claim_type = parsed.claim_type
        case.parsed_reject_reason = parsed.reject_reason
        case.outcome = classify_outcome(
            case.gold_claim_type, False, parsed.claim_type
        )
    except Exception as e:
        case.outcome = OUTCOME_ERROR
        case.error = f"parser: {type(e).__name__}: {e}"
        logger.warning(f"row {case.row_id} parser errored: {e}")

    case.elapsed_ms = int((time.time() - start) * 1000)
    return case


def summarize(cases: List[RejectCase]) -> Dict[str, Any]:
    """Aggregate into a confusion matrix plus recall and precision.

    Errors are excluded from every rate: a transport failure is not evidence
    about the reject decision either way.
    """
    scored = [c for c in cases if c.outcome not in (OUTCOME_ERROR, OUTCOME_PENDING)]
    gold_reject = [c for c in scored if c.gold_claim_type == "reject"]
    gold_other = [c for c in scored if c.gold_claim_type != "reject"]

    blocked = [c for c in gold_reject if c.outcome == OUTCOME_BLOCKED_BY_GUARD]
    by_parser = [c for c in gold_reject if c.outcome == OUTCOME_REJECTED_BY_PARSER]
    missed = [c for c in gold_reject if c.outcome == OUTCOME_MISSED]
    false_reject = [c for c in gold_other if c.outcome == OUTCOME_FALSE_REJECT]

    rejected = len(blocked) + len(by_parser)
    all_rejections = rejected + len(false_reject)

    by_layer: Dict[str, int] = {}
    for c in scored:
        if c.outcome in (OUTCOME_BLOCKED_BY_GUARD, OUTCOME_FALSE_REJECT) and c.guard_layer:
            by_layer[c.guard_layer] = by_layer.get(c.guard_layer, 0) + 1

    by_reason: Dict[str, Dict[str, int]] = {}
    for c in gold_reject:
        entry = by_reason.setdefault(c.gold_reject_reason or "unknown",
                                     {"total": 0, "rejected": 0})
        entry["total"] += 1
        entry["rejected"] += 1 if c.outcome in _REJECTED_OUTCOMES else 0

    by_outcome: Dict[str, int] = {}
    for c in cases:
        by_outcome[c.outcome] = by_outcome.get(c.outcome, 0) + 1

    return {
        "total": len(cases),
        "scored": len(scored),
        "errors": sum(1 for c in cases if c.outcome == OUTCOME_ERROR),
        "gold_reject": len(gold_reject),
        "gold_non_reject": len(gold_other),
        "blocked_by_guard": len(blocked),
        "rejected_by_parser": len(by_parser),
        "missed": len(missed),
        "false_reject": len(false_reject),
        "accepted": sum(1 for c in gold_other if c.outcome == OUTCOME_ACCEPTED),
        "rejected": rejected,
        "reject_recall": (rejected / len(gold_reject)) if gold_reject else 0.0,
        "reject_precision": (rejected / all_rejections) if all_rejections else 0.0,
        "by_guard_layer": by_layer,
        "by_reject_reason": by_reason,
        "by_outcome": by_outcome,
    }


def _print_report(cases: List[RejectCase], summary: Dict[str, Any]) -> None:
    missed = [c for c in cases if c.outcome == OUTCOME_MISSED]
    if missed:
        print(f"\n{'-' * 78}\nMISSED — gold reject, routed to an agent ({len(missed)})")
        print(f"{'-' * 78}")
        print("  Each of these would have run the full verification pipeline.\n")
        for c in missed:
            print(f"  row {c.row_id:<5} gold={c.gold_reject_reason or '?':<32} "
                  f"parsed as {c.parsed_claim_type}")
            print(f"    {c.claim[:72]}")

    false_rejects = [c for c in cases if c.outcome == OUTCOME_FALSE_REJECT]
    if false_rejects:
        print(f"\n{'-' * 78}\nFALSE REJECTS — verifiable claim stopped ({len(false_rejects)})")
        print(f"{'-' * 78}")
        for c in false_rejects:
            how = c.violation_type or f"parser:{c.parsed_reject_reason}"
            print(f"  row {c.row_id:<5} gold={c.gold_claim_type:<8} {how}")
            print(f"    {c.claim[:72]}")

    errors = [c for c in cases if c.outcome == OUTCOME_ERROR]
    if errors:
        print(f"\n{'-' * 78}\nERRORS ({len(errors)})\n{'-' * 78}")
        for c in errors:
            print(f"  row {c.row_id:<5} {c.error}")

    s = summary
    print(f"\n{'=' * 78}\nREJECT CLASSIFICATION\n{'=' * 78}")
    print(f"  rows loaded         : {s['total']}")
    print(f"  scored              : {s['scored']}"
          + (f"   ({s['errors']} errored)" if s["errors"] else ""))
    print()
    print(f"  gold reject         : {s['gold_reject']}")
    print(f"    blocked by guard  : {s['blocked_by_guard']}")
    print(f"    rejected by parser: {s['rejected_by_parser']}")
    print(f"    MISSED            : {s['missed']}")
    print(f"  gold non-reject     : {s['gold_non_reject']}")
    print(f"    accepted          : {s['accepted']}")
    print(f"    FALSE REJECT      : {s['false_reject']}")
    print()
    print(f"  reject recall       : {s['reject_recall']:.1%}"
          f"   ({s['rejected']}/{s['gold_reject']})")
    print(f"  reject precision    : {s['reject_precision']:.1%}")

    if s["by_guard_layer"]:
        print("\n  guard hits by layer:")
        for layer, n in sorted(s["by_guard_layer"].items(), key=lambda kv: -kv[1]):
            print(f"    {layer:<18}{n:>4}")

    if s["by_reject_reason"]:
        print("\n  by gold reject_reason:")
        for reason, r in sorted(s["by_reject_reason"].items(),
                                key=lambda kv: (kv[1]["rejected"] / kv[1]["total"],
                                                -kv[1]["total"])):
            rate = r["rejected"] / r["total"]
            print(f"    {reason:<40}{r['rejected']:>3}/{r['total']:<4} {rate:>6.0%}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score FinVet's reject decision against claim-parser gold labels"
    )
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                    help="Gold JSONL (default: the real-sourced held-out set)")
    ap.add_argument("--synthetic", action="store_true",
                    help=f"Use the synthetic split instead ({SYNTHETIC_GOLD.name})")
    ap.add_argument("--limit", type=int, help="Score only the first N rows")
    ap.add_argument("--claim-type", dest="claim_type",
                    choices=["reject", "sec", "market", "news"],
                    help="Only score rows with this gold claim_type")
    ap.add_argument("--report", type=Path, help="Write the full JSON report here")
    args = ap.parse_args()

    gold = SYNTHETIC_GOLD if args.synthetic else args.gold
    if not gold.exists():
        print(f"Gold set not found: {gold}\n"
              f"This harness needs the claim-parser project's gold labels.")
        return 2

    cases = load_cases(gold, limit=args.limit, claim_type=args.claim_type)
    if not cases:
        print("No rows matched.")
        return 2

    print(f"Scoring {len(cases)} row(s) against {gold.name}\n")
    for i, case in enumerate(cases, 1):
        run_case(case)
        flag = "  <-- MISSED" if case.outcome == OUTCOME_MISSED else (
            "  <-- FALSE REJECT" if case.outcome == OUTCOME_FALSE_REJECT else "")
        print(f"  [{i}/{len(cases)}] row {case.row_id:<5} "
              f"gold={case.gold_claim_type:<8} {case.outcome}{flag}")

    summary = summarize(cases)
    _print_report(cases, summary)

    if args.report:
        args.report.write_text(json.dumps(
            {"gold": str(gold), "summary": summary,
             "cases": [asdict(c) for c in cases]}, indent=2))
        print(f"  report written to {args.report}\n")

    return 0 if summary["missed"] == 0 and summary["false_reject"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
