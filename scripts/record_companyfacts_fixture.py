#!/usr/bin/env python
"""Record SEC companyfacts as a test fixture, trimmed and unedited.

Fact-selection tests run on what SEC publishes, not on payloads written by
hand: a hand-written payload contains the cases its author thought of, and the
defect that reached a benchmark on 2026-09-28 was one nobody had (a proxy
statement carrying net income in millions under the same tag as the 10-K).

Trimming is by concept and by period end only. No value, date, form or filing
reference is altered, and every form type is kept -- the non-statement facts
are the point.

Usage:
    .venv/bin/python scripts/record_companyfacts_fixture.py HD FDX MET --since 2022
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.config.settings import settings  # noqa: E402
from finvet.mcp.sec_edgar import CONCEPTS_BY_TYPE  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sec_companyfacts"
TRACKED = sorted({c for names in CONCEPTS_BY_TYPE.values() for c in names})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--since", type=int, default=2022,
                    help="keep facts whose period ends in this year or later")
    args = ap.parse_args()

    headers = {"User-Agent": settings.sec_edgar_user_agent}
    listing = httpx.get("https://www.sec.gov/files/company_tickers.json",
                        headers=headers, timeout=30).json()
    cik_of = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in listing.values()}

    OUT.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        cik = cik_of[ticker]
        full = httpx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                         headers=headers, timeout=60).json()
        gaap = full.get("facts", {}).get("us-gaap", {})
        kept = {}
        for concept in TRACKED:
            units = (gaap.get(concept) or {}).get("units") or {}
            trimmed = {unit: [f for f in facts if f.get("end", "")[:4] >= str(args.since)]
                       for unit, facts in units.items()}
            trimmed = {unit: facts for unit, facts in trimmed.items() if facts}
            if trimmed:
                kept[concept] = {"units": trimmed}
        path = OUT / f"{ticker}.json"
        path.write_text(json.dumps(
            {"cik": int(cik), "entityName": full.get("entityName"),
             "recorded_from": "https://data.sec.gov/api/xbrl/companyfacts/",
             "trimmed_to": {"concepts": "finvet CONCEPTS_BY_TYPE", "since": args.since},
             "facts": {"us-gaap": kept}}, indent=0))
        n = sum(len(f) for c in kept.values() for f in c["units"].values())
        print(f"{ticker}: {len(kept)} concepts, {n} facts, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
