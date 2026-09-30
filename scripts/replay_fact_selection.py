#!/usr/bin/env python
"""Replay gate: does FinVet select the right filed number, with no model involved?

For every dataset row whose evidence is an XBRL fact, this drives the production
path a claim takes after parsing -- period resolution, the SEC statement tool,
the trusted-observation resolver, the comparator -- and compares the verdict
with the label. No LLM is called and nothing is spent; the only traffic is to
the SEC MCP server and data.sec.gov.

It exists because a selection change shipped on 2026-09-28 with fourteen green
unit tests and refuted a true claim the same night: the tests used payloads
written by hand, and real SEC data held a proxy statement tagged in millions.
A selector is only as good as the facts it has been run against, so this runs
it against all of them. Run it before any paid benchmark and after any change
to fact selection.

Each row is replayed twice, from two different filings: the one the label
cites, and the issuer's most recent filing of that form. The agent chooses
which filing to open, so a verdict that changes with that choice is a verdict
the model decided. Both passes must agree.

Usage:
    FINVET_GOLDEN_DIR=/path/to/dir .venv/bin/python scripts/replay_fact_selection.py \
        --dataset golden_u.jsonl [--ids 12,34] [--report /path/to/report.json]

Exit status is 1 when any row misses and is not listed, with a reason, in
scripts/replay_explained.json.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.agents.base import (  # noqa: E402
    SUPERSEDED_LIMITATION,
    compare_observation,
    superseded_value_match,
)
from finvet.config.metrics import METRIC_TO_CONCEPTS  # noqa: E402
from finvet.eval.dataset import golden_data_file  # noqa: E402
from finvet.graph.nodes.period_resolver import period_resolver  # noqa: E402
from finvet.mcp.sec_edgar import CONCEPTS_BY_TYPE  # noqa: E402
from finvet.models.claim import ParsedClaim  # noqa: E402
from finvet.models.evidence import (  # noqa: E402
    resolve_trusted_observation,
    tool_record_from_result,
)
from finvet.tools import sec_tools  # noqa: E402

EXPLAINED = Path(__file__).with_name("replay_explained.json")
TOOLS = {"income": sec_tools.get_income_statement,
         "balance": sec_tools.get_balance_sheet,
         "cashflow": sec_tools.get_cash_flow}
PARSE_FIELDS = ("claim_type", "ticker", "metric", "operator", "value",
                "period", "reject_reason")


def statement_for(metric: str) -> Optional[str]:
    concepts = METRIC_TO_CONCEPTS.get(metric) or ()
    for kind, names in CONCEPTS_BY_TYPE.items():
        if any(c in names for c in concepts):
            return kind
    return None


def replayable(row: Dict[str, Any]) -> bool:
    gold = row.get("gold_parse") or {}
    expected = row["expected"]
    decided_by_a_fact = (
        expected["verdict"] in ("SUPPORTS", "REFUTES")
        or expected.get("limitation") == SUPERSEDED_LIMITATION)
    return (decided_by_a_fact
            and (expected["sources"] or []) == ["xbrl"]
            and gold.get("claim_type") == "sec"
            and statement_for(gold.get("metric") or "") is not None)


class Filings:
    """CIK and filing lookups, once per issuer."""

    def __init__(self) -> None:
        self.client = sec_tools._get_client()
        self._cik: Dict[str, str] = {}
        self._latest: Dict[tuple, Optional[str]] = {}

    def cik(self, ticker: str) -> str:
        if ticker not in self._cik:
            self._cik[ticker] = self.client.get_company_info(ticker).cik
        return self._cik[ticker]

    def latest(self, ticker: str, form: str) -> Optional[str]:
        key = (ticker, form)
        if key not in self._latest:
            found = self.client.get_recent_filings(self.cik(ticker), form, 1)
            self._latest[key] = found[0].accession_number if found else None
        return self._latest[key]


def replay(row: Dict[str, Any], accession: str, cik: str) -> Dict[str, Any]:
    gold = row["gold_parse"]
    claim = ParsedClaim(**{k: gold.get(k) for k in PARSE_FIELDS})
    period = period_resolver({"parsed_claim": claim,
                              "request_id": f"replay-{row['id']}"})["canonical_period"]
    target = sec_tools.period_target_for(period) or (None, None)
    fiscal = sec_tools.fiscal_target_for(period)
    tool = TOOLS[statement_for(claim.metric)]
    args = {"cik": cik, "accession_number": accession}
    if tool is sec_tools.get_income_statement:
        args["period"] = period.period_type

    with sec_tools.use_period_target(*target, fiscal=fiscal):
        result = tool.invoke(args)

    record = tool_record_from_result(tool.name, args, result)
    usable = period.period_type in sec_tools._DATABLE_PERIOD_TYPES
    observation = resolve_trusted_observation(
        claim, [record],
        expected_period_end=period.end_date if usable else None,
        expected_period_start=period.start_date if usable else None,
        expected_fiscal=fiscal if usable else None,
    )
    verdict, _, diff = compare_observation(claim, observation, "sec")
    limitation = None
    if verdict == "REFUTES" and superseded_value_match(claim, observation, "sec") is not None:
        verdict, limitation = "NOT_ENOUGH_INFO", SUPERSEDED_LIMITATION
    return {
        "verdict": verdict,
        "limitation": limitation,
        "superseded": observation.superseded_values if observation else [],
        "value": observation.value if observation else None,
        "period_end": observation.period_end if observation else None,
        "concept": observation.concept if observation else None,
        "diff_pct": diff,
        "tool_success": bool(result.get("success")),
        "tool_error": result.get("error"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="golden_u.jsonl",
                    help="dataset file inside $FINVET_GOLDEN_DIR")
    ap.add_argument("--ids", default="", help="comma-separated row ids; default all")
    ap.add_argument("--report", default=None, help="write the per-row results here")
    args = ap.parse_args()

    path = golden_data_file(args.dataset)
    if path is None or not path.exists():
        sys.exit("set FINVET_GOLDEN_DIR to the directory holding the dataset")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    wanted = {int(i) for i in args.ids.split(",") if i.strip()}
    rows = [r for r in rows if replayable(r) and (not wanted or r["id"] in wanted)]
    explained = json.loads(EXPLAINED.read_text()) if EXPLAINED.exists() else {}

    filings = Filings()
    out: List[Dict[str, Any]] = []
    started = time.time()
    for n, row in enumerate(rows, 1):
        gold = row["gold_parse"]
        form = "10-Q" if gold["period"].upper().startswith("Q") else "10-K"
        cited = re.search(r"accn (\d{10}-\d{2}-\d{6})", row.get("source") or "")
        entry: Dict[str, Any] = {"id": row["id"], "ticker": gold["ticker"],
                                 "metric": gold["metric"], "period": gold["period"],
                                 "class": (row.get("tags") or {}).get("class"),
                                 "expected": row["expected"]["verdict"],
                                 "expected_limitation": row["expected"].get("limitation"),
                                 "passes": {}}
        try:
            cik = filings.cik(gold["ticker"])
            accessions = {"cited": cited.group(1) if cited else None,
                          "latest": filings.latest(gold["ticker"], form)}
            for name, accn in accessions.items():
                if accn:
                    entry["passes"][name] = replay(row, accn, cik)
        except Exception as exc:  # a crash is a finding, not a reason to stop
            entry["error"] = f"{type(exc).__name__}: {exc}"

        verdicts = {p["verdict"] for p in entry["passes"].values()}
        values = {p["value"] for p in entry["passes"].values()}
        limitations = {p["limitation"] for p in entry["passes"].values()}
        entry["ok"] = (not entry.get("error") and bool(entry["passes"])
                       and verdicts == {entry["expected"]}
                       and limitations == {entry["expected_limitation"]})
        entry["filing_dependent"] = len(values) > 1 or len(verdicts) > 1
        out.append(entry)
        mark = "ok  " if entry["ok"] else "MISS"
        shown = " | ".join(f"{k}: {p['verdict']} {p['value']} end={p['period_end']}"
                           for k, p in entry["passes"].items())
        print(f"{n:4}/{len(rows)} {mark} id {row['id']:<5} {gold['ticker']:<5} "
              f"{gold['metric']:<16} {gold['period']:<8} exp={entry['expected']:<8} "
              f"{shown}{'  ERROR ' + entry['error'] if entry.get('error') else ''}",
              flush=True)

    misses = [e for e in out if not e["ok"]]
    unexplained = [e for e in misses if str(e["id"]) not in explained]
    dependent = [e["id"] for e in out if e["filing_dependent"]]
    by_class = Counter((e["class"] or "set_c") for e in misses)
    print(f"\nreplayed {len(out)} rows in {time.time() - started:.0f}s: "
          f"{len(out) - len(misses)} ok, {len(misses)} miss "
          f"({len(misses) - len(unexplained)} explained, {len(unexplained)} unexplained)")
    print(f"misses by class: {dict(by_class)}")
    print(f"rows whose result depends on the filing opened: {dependent}")
    print(f"unexplained: {[e['id'] for e in unexplained]}")
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=1))
        print(f"report written to {args.report}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
