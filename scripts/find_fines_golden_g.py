#!/usr/bin/env python
"""Candidates for the fines & settlements stratum, from EDGAR full-text search.

Restricted to main 10-K documents (file_type == "10-K") and to the CIKs given,
because an unrestricted query is dominated by exhibit-95 mine-safety pages and
micro-caps. Prints one line per hit for a person to open and read.

Usage:
    .venv/bin/python scripts/find_fines_golden_g.py --tickers JPM,WFC,BA --since 2025-01-01
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finvet.config.settings import settings  # noqa: E402

EFTS = "https://efts.sec.gov/LATEST/search-index"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
PHRASES = ['"civil penalty"', '"agreed to pay"', '"settlement of"', '"fine of"']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--since", default="2025-01-01")
    args = ap.parse_args()
    with httpx.Client(headers={"User-Agent": settings.sec_edgar_user_agent}, timeout=30) as c:
        by_ticker = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in c.get(TICKER_URL).json().values()}
        ciks = ",".join(by_ticker[t] for t in args.tickers.upper().split(","))
        seen = set()
        for phrase in PHRASES:
            r = c.get(EFTS, params={"q": phrase, "forms": "10-K", "dateRange": "custom",
                                    "startdt": args.since, "enddt": "2026-12-31", "ciks": ciks})
            r.raise_for_status()
            for h in r.json()["hits"]["hits"]:
                s = h["_source"]
                if s.get("file_type") != "10-K":
                    continue
                accn, doc = h["_id"].split(":")
                key = (s["ciks"][0], accn)
                if key in seen:
                    continue
                seen.add(key)
                cik = s["ciks"][0].lstrip("0")
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}"
                print(f"{s['display_names'][0][:40]:<40} {s['period_ending']} {phrase:<18} {url}")
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
