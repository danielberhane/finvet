#!/usr/bin/env python
"""Run the private golden claim set through the live API and record what happened.

**This script asserts nothing.** It is the only step that costs money, so its
job is to produce an artifact that can be re-analysed indefinitely without
spending again -- and that a later run under a different model can be diffed
against. Judgement lives in `tests/integration/test_golden.py`, which reads the
artifact and is free to run as often as you like.

`scripts/claim_matrix.py` does not make that split: its assertions call the API,
so every pytest run re-spends. That is the mistake this avoids.

Two outcome classes are kept apart, because collapsing them hides the
distinction the system is built on:

    ANSWERED   the pipeline reached a verdict
    ESCALATED  it declined and sent the claim to a human

An escalation is a legitimate outcome, not a failure. The dataset's status
vocabulary (success/rejected/blocked) has no word for it, so the runner records
`terminal_status` verbatim rather than forcing it into that vocabulary.

Rows are also split by input stability, which decides what they can be
compared against:

    frozen  identical inputs on every run (XBRL, filings, parser, guards).
            These carry every comparison, now and across models.
    live    news and market inputs change by the second. Recorded, never
            scored across runs.

Usage:
    FINVET_GOLDEN_DIR=~/Projects/Active/finvet-golden \\
      .venv/bin/python scripts/run_golden.py --to 10          # cost probe
    .venv/bin/python scripts/run_golden.py                    # full pass
    .venv/bin/python scripts/run_golden.py --frozen-only --label flake-b
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from finvet.eval import exclusions  # noqa: E402
from finvet.eval.dataset import golden_data_file  # noqa: E402
from finvet.llm.factory import active_llm_config  # noqa: E402

API = os.environ.get("FINVET_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Categories whose inputs are identical on every run. Everything else depends on
# what a news search or a price feed returned at that moment.
FROZEN_CATEGORIES = frozenset({
    "sec/xbrl", "sec/tolerance", "sec/operator", "sec/qualitative",
    "reject", "guard", "declined", "known-defect",
})


def is_frozen(row: dict) -> bool:
    return row.get("category") in FROZEN_CATEGORIES


def fill_recipe(row: dict, price: float, latest_trading_day, source_mode: str = "live") -> dict:
    """Fill a live-price recipe from a snapshot quote.

    The dataset row stores only the recipe; the number is unknowable in
    advance. The verdict is fixed by the offset sign, so `expected` stays
    frozen — only the threshold is live, and it is recorded so the label can
    be checked against the quote that produced it.
    """
    if source_mode != "live":
        raise RuntimeError("snapshot from a mock quote is not an observation")
    threshold = round(price * (1 + row["recipe"]["offset_pct"] / 100), 2)
    out = dict(row)
    out["claim"] = row["claim"].replace("{threshold}", f"${threshold:,.2f}")
    out["gold_parse"] = dict(row["gold_parse"], value=threshold)
    out["snapshot"] = {"price": price, "latest_trading_day": latest_trading_day,
                       "threshold": threshold,
                       "taken_utc": datetime.now(timezone.utc).isoformat()}
    return out


def api_llm_config():
    """Ask the API which models it will use, or None if it cannot say.

    None means an API too old to report it, which is not the same as agreement
    -- an unattributable run is refused rather than recorded under whatever
    this shell happens to hold.
    """
    try:
        response = httpx.get(f"{API}/health", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        print(f"  Could not reach {API}/health: {exc}", file=sys.stderr)
        return None
    served = (response.json() or {}).get("llm")
    return served if isinstance(served, dict) and served else None


def prefetch_quotes(selected: list) -> dict:
    """Fetch every distinct ticker's quote up front, one FinnhubClient for all of them.

    A recipe row's quote used to be fetched inline, in the middle of the scored
    loop: one bad ticker raised out of `get_quote` and killed the whole run,
    discarding every row already paid for. Fetching everything first, with each
    failure caught and stored rather than raised, means one bad quote costs
    exactly the rows that needed it.
    """
    tickers = sorted({r["recipe"]["ticker"] for r in selected if "recipe" in r})
    if not tickers:
        return {}
    from finvet.mcp.finnhub import FinnhubClient
    client = FinnhubClient()
    out: dict = {}
    for ticker in tickers:
        try:
            out[ticker] = client.get_quote(ticker)
        except Exception as e:
            out[ticker] = e
    return out


def _quote_error_record(row: dict, exc: Exception) -> dict:
    """An errored record for a recipe row whose quote could not be fetched.

    Shaped like `run_one`'s record so the artifact stays uniform, but the API
    is never called -- there is no threshold to verify a claim against.
    """
    return {
        "id": row.get("id"),
        "claim": row.get("claim"),
        "category": row.get("category"),
        "strength": row.get("strength"),
        "frozen": is_frozen(row),
        "expected": row.get("expected"),
        "snapshot": row.get("snapshot"),
        "gold_parse": row.get("gold_parse"),
        "actual": None,
        "error": f"quote unavailable: {exc}",
        "elapsed_s": 0.0,
    }


def _claim_length_refused(body: dict) -> bool:
    """Whether a validation error is the claim being too short or too long."""
    errors = body.get("detail")
    if not isinstance(errors, list):
        return False
    return any(
        isinstance(e, dict)
        and e.get("type") in ("string_too_short", "string_too_long")
        and list(e.get("loc") or [])[-1:] == ["claim"]
        for e in errors
    )


def run_one(row: dict) -> dict:
    """One claim through the live route. Never raises: a failure is an outcome."""
    started = time.time()
    record = {
        "id": row.get("id"),
        "claim": row.get("claim"),
        "category": row.get("category"),
        "strength": row.get("strength"),
        "frozen": is_frozen(row),
        "expected": row.get("expected"),
        "snapshot": row.get("snapshot"),
        "gold_parse": row.get("gold_parse"),
    }
    try:
        response = httpx.post(f"{API}/verify", json={"claim": row["claim"]},
                              timeout=600)
        body = response.json() if response.headers.get(
            "content-type", "").startswith("application/json") else {}
    except Exception as exc:
        record.update(actual=None, error=f"{type(exc).__name__}: {exc}",
                      elapsed_s=round(time.time() - started, 1))
        return record

    meta = body.get("metadata") or {}
    observation = meta.get("trusted_observation") or {}
    status = body.get("status")

    # An input guardrail refuses the request with HTTP 400 and no verdict field,
    # so reading `verdict` alone recorded a correct block as an empty answer and
    # the harness scored all six guard rows as failures. `claim_matrix.py`
    # already maps this; the mapping belongs wherever a response is read.
    verdict = body.get("verdict")
    if verdict is None and response.status_code == 400:
        verdict = "BLOCKED"
    # The API's own length validation answers before the guard does, with
    # HTTP 422. It is the same outcome -- refused at the door, no model call --
    # and recording it as an empty verdict scored two correct refusals as
    # wrong answers. Only a 422 about the claim's length counts: a rejected
    # field elsewhere in the request is a malformed request, not a block.
    if verdict is None and response.status_code == 422 and _claim_length_refused(body):
        verdict = "BLOCKED"

    record.update(
        elapsed_s=round(time.time() - started, 1),
        http=response.status_code,
        request_id=body.get("request_id"),
        actual={
            "verdict": verdict,
            # Recorded verbatim. "pending" is a real outcome the dataset's
            # vocabulary cannot express, and forcing it into one would make
            # every escalated row read as a wrong answer.
            "status": status,
            "confidence": body.get("confidence"),
            "limitation": meta.get("limitation"),
            "retrieved_value": meta.get("retrieved_value"),
            "observation_tool": observation.get("tool"),
            "observation_period_end": observation.get("period_end"),
            # Every tool the run called. Layer 2 scores the path taken, and an
            # artifact recording only the answer cannot say whether a correct
            # answer was reached soundly. Kept here so the artifact stays
            # self-contained -- otherwise trajectory scoring needs this
            # machine's audit database. `metadata.tools_called` is already a
            # flat list of names on both the success and pending paths.
            "tools_called": list(meta.get("tools_called") or []),
            # What the parser decided. Trajectory expectations key on the
            # routing strategy, and the strategy follows claim_type + metric --
            # not the dataset category. Two `a2a` rows parse as sec claims
            # about what a filing says, so category is the wrong proxy.
            "parsed_claim": meta.get("parsed_claim"),
            "data_sources": sorted((meta.get("data_sources") or {}).keys()),
            "a2a_status": ((meta.get("data_sources") or {}).get("a2a")
                           or {}).get("status"),
            "escalated": status == "pending_review"
            or body.get("verdict") == "PENDING",
            # The deterministic override, which is the system's central claim
            # and was the one thing the artifact could not show. The API has
            # always returned both fields; nothing copied them here, so across
            # 476 published per-claim records there was no way to count how
            # often Python overruled the model, or to demonstrate it ever did.
            "override_applied": meta.get("override_applied"),
            "llm_original_verdict": meta.get("llm_original_verdict"),
        },
        error=None,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=10**6)
    parser.add_argument("--frozen-only", action="store_true",
                        help="skip news/market rows; the comparable subset")
    parser.add_argument("--label", default="",
                        help="tag for the artifact, e.g. flake-a / flake-b")
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-model-drift", action="store_true",
                        help="run even though the API serves a different model "
                             "than this shell's env names")
    parser.add_argument("--dataset", default="golden_c.jsonl",
                        help="golden file inside $FINVET_GOLDEN_DIR")
    args = parser.parse_args()

    path = golden_data_file(args.dataset)
    if path is None or not path.exists():
        print("FINVET_GOLDEN_DIR is unset or the dataset is missing.",
              file=sys.stderr)
        print("  export FINVET_GOLDEN_DIR=~/Projects/Active/finvet-golden",
              file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # A burned row's claim text is public (DATASET_CARD.md publishes it), so it
    # never counts in a scored run again -- and running it is spend that buys
    # nothing. The card promises this exclusion; enforce it where rows are chosen.
    burned = exclusions.burned_ids(path.name)
    selected = [r for r in rows if args.start <= r.get("id", 0) <= args.end
                and (not args.frozen_only or is_frozen(r))
                and r.get("id") not in burned]

    # Count the burned rows this file actually carries, not the size of the
    # registry: a second dataset's ids are registered too, and reporting those
    # would claim rows were skipped that were never in the file.
    burned_here = sum(1 for r in rows if r.get("id") in burned)
    print(f"  dataset : {path}  ({len(rows)} rows, {len(selected)} selected, "
          f"{burned_here} burned ids skipped)")
    print(f"  api     : {API}")

    # What the *serving process* will use. Reading our own environment answers
    # a different question -- this script only posts HTTP -- and the two came
    # apart: a run launched with the MiniMax env against an API still holding
    # the DeepSeek one recorded "MiniMax-M2.7" over 56 DeepSeek rows. A model
    # label nobody can trust makes every cross-model comparison worthless, so
    # the label now comes from the process that does the work.
    served = api_llm_config()
    if served is None:
        print("  The API did not report its LLM config. Restart it so /health "
              "carries `llm`, otherwise this run cannot be attributed to a "
              "model.", file=sys.stderr)
        return 2

    local = active_llm_config()
    drift = {role for role in served
             if served[role].get("model") != local.get(role, {}).get("model")}
    if drift and not args.allow_model_drift:
        print(f"  The API is serving a different model than this shell expects "
              f"for {sorted(drift)}:", file=sys.stderr)
        for role in sorted(drift):
            print(f"    {role:<8} api={served[role].get('model')} "
                  f"shell={local.get(role, {}).get('model')}", file=sys.stderr)
        print("  Restart the API with the env you mean to benchmark, or pass "
              "--allow-model-drift if the difference is deliberate.",
              file=sys.stderr)
        return 2

    for role in ("parser", "agent", "verdict"):
        cfg = served.get(role, {})
        print(f"  {role:<8}: {cfg.get('model')}  @ {cfg.get('base_url') or 'default'}")
    print()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"-{args.label}" if args.label else ""
    out = Path(args.out) if args.out else path.parent / f"run-{stamp}{tag}.json"

    def write(results, elapsed, complete):
        """Persist after every claim.

        The first version wrote once, at the end. A run killed 24 claims in
        left no artifact at all -- an hour of API spend with nothing to show,
        because the only copy lived in memory. Rows are paid for one at a time,
        so they are saved one at a time, and `complete` says whether the file
        is a whole run or a partial one.
        """
        out.write_text(json.dumps({
            "started_utc": stamp,
            "label": args.label,
            "complete": complete,
            "api": API,
            "dataset": str(path),
            "dataset_rows": len(rows),
            "selected": len(selected),
            "frozen_only": args.frozen_only,
            # What produced this, as reported by the process that produced it.
            # Without it a later run under another model has nothing to diff
            # against and the money is spent twice; with the *client's* copy of
            # it, worse -- the diff runs and the answer is wrong.
            "llm_config": served,
            "llm_config_source": "api:/health",
            # Kept beside it so a drifted run allowed on purpose still says so.
            "llm_config_client": local,
            "elapsed_s": round(elapsed, 1),
            "results": results,
        }, indent=1))

    results, began = [], time.time()
    quotes = prefetch_quotes(selected)
    for n, row in enumerate(selected, 1):
        if "recipe" in row:
            q = quotes[row["recipe"]["ticker"]]
            if isinstance(q, Exception):
                record = _quote_error_record(row, q)
                results.append(record)
                write(results, time.time() - began, complete=False)
                print(f"  ERR {n:>3}/{len(selected)}  id {record['id']:>3} "
                      f"[{record['category']:<15}] {'None':<16} "
                      f"{record['elapsed_s']:>5.1f}s  {row.get('claim', '')[:44]}",
                      flush=True)
                continue
            row = fill_recipe(row, q.price, q.latest_trading_day, q.source_mode)
        record = run_one(row)
        results.append(record)
        write(results, time.time() - began, complete=False)

        actual = record.get("actual") or {}
        mark = "ERR " if record.get("error") else (
            "ESC " if actual.get("escalated") else "    ")
        # flush: stdout is block-buffered when redirected, so an hour-long run
        # showed an empty log the whole time.
        print(f"  {mark}{n:>3}/{len(selected)}  id {record['id']:>3} "
              f"[{record['category']:<15}] {str(actual.get('verdict')):<16} "
              f"{record.get('elapsed_s', 0):>5.1f}s  {record['claim'][:44]}",
              flush=True)

    answered = [r for r in results if (r.get("actual") or {}).get("verdict")
                and not (r.get("actual") or {}).get("escalated")]
    escalated = [r for r in results if (r.get("actual") or {}).get("escalated")]
    errored = [r for r in results if r.get("error")]
    elapsed = time.time() - began

    write(results, elapsed, complete=True)

    print(f"\n  answered {len(answered)}   escalated {len(escalated)}   "
          f"errors {len(errored)}   of {len(selected)}")
    print(f"  {elapsed:.0f}s total, {elapsed / max(len(selected), 1):.1f}s per claim")
    print(f"  written to {out}")
    print("\n  Nothing was asserted. Run tests/integration/test_golden.py "
          "against this artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
