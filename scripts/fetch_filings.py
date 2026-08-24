"""Download SEC 10-K/10-Q filings for the RAG corpus.

The corpus itself is gitignored — this script is how it gets rebuilt.

Usage:
    python scripts/fetch_filings.py --tickers AAPL,MSFT --years 2024
    python scripts/fetch_filings.py --tickers AAPL --forms 10-Q --limit 3

Files land in data/filings/<TICKER>/ named TICKER_FORM_PERIOD-END.html, which
is the layout finvet.rag.ingest expects. SEC requires a real contact address on
automated requests; this reads SEC_EDGAR_USER_AGENT from your .env.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
load_dotenv()

from finvet.config.settings import settings  # noqa: E402

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# SEC asks for no more than 10 requests/second.
REQUEST_DELAY_S = 0.15


def _client() -> httpx.Client:
    if settings.sec_user_agent_is_placeholder:
        sys.exit(
            "SEC_EDGAR_USER_AGENT is still the shipped placeholder. SEC blocks "
            "automated requests without a real name and contact address — set it "
            "in .env before fetching."
        )
    return httpx.Client(
        headers={"User-Agent": settings.sec_edgar_user_agent},
        timeout=60.0,
        follow_redirects=True,
    )


def resolve_ciks(client: httpx.Client, tickers: list[str]) -> dict[str, str]:
    """Map tickers to zero-padded CIKs using SEC's own ticker index.

    Looked up rather than hardcoded so any ticker works without a code change.
    """
    resp = client.get(TICKER_URL)
    resp.raise_for_status()
    index = {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in resp.json().values()
    }
    out = {}
    for t in tickers:
        cik = index.get(t.upper())
        if cik is None:
            print(f"  ! {t}: not found in SEC ticker index — skipping")
        else:
            out[t.upper()] = cik
    return out


def find_filings(
    client: httpx.Client, ticker: str, cik: str, forms: list[str],
    years: list[str], limit: int,
) -> list[dict]:
    """Return filing metadata for a ticker, newest first."""
    resp = client.get(SUBMISSIONS_URL.format(cik=cik))
    resp.raise_for_status()
    recent = resp.json().get("filings", {}).get("recent", {})

    found = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        period = recent.get("reportDate", [])[i]
        if years and not any(period.startswith(y) for y in years):
            continue
        found.append({
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "period_end": period,
            "filing_date": recent.get("filingDate", [])[i],
            "accession": recent.get("accessionNumber", [])[i].replace("-", ""),
            "document": recent.get("primaryDocument", [])[i],
        })
        if len(found) >= limit:
            break
    return found


def download(client: httpx.Client, filing: dict, out_dir: Path) -> Path | None:
    """Download one filing's primary document. Returns None if already present."""
    dest_dir = out_dir / filing["ticker"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{filing['ticker']}_{filing['form']}_{filing['period_end']}.html"
    if dest.exists():
        print(f"  = {dest.name} (already downloaded)")
        return None

    url = ARCHIVE_URL.format(
        cik=int(filing["cik"]),
        accession=filing["accession"],
        doc=filing["document"],
    )
    resp = client.get(url)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  + {dest.name} ({len(resp.content) / 1_000_000:.1f} MB)")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", required=True, help="Comma-separated, e.g. AAPL,MSFT")
    ap.add_argument("--forms", default="10-K,10-Q", help="Comma-separated form types")
    ap.add_argument("--years", default="", help="Comma-separated period-end years")
    ap.add_argument("--limit", type=int, default=4, help="Max filings per ticker")
    ap.add_argument("--out", default="data/filings", help="Output directory")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    out_dir = Path(args.out)

    downloaded = 0
    with _client() as client:
        ciks = resolve_ciks(client, tickers)
        for ticker, cik in ciks.items():
            print(f"{ticker} (CIK {cik}):")
            time.sleep(REQUEST_DELAY_S)
            filings = find_filings(client, ticker, cik, forms, years, args.limit)
            if not filings:
                print("  ! no matching filings")
                continue
            for filing in filings:
                time.sleep(REQUEST_DELAY_S)
                try:
                    if download(client, filing, out_dir):
                        downloaded += 1
                except httpx.HTTPError as e:
                    print(f"  ! {filing['form']} {filing['period_end']}: {e}")

    print(f"\nDownloaded {downloaded} new filing(s) into {out_dir}/")
    print("Next: python -m finvet.rag.ingest")


if __name__ == "__main__":
    main()
