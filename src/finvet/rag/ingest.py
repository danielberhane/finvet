"""CLI script to ingest SEC filing HTML files into the RAG vector store.

Usage:
    python -m finvet.rag.ingest
    python -m finvet.rag.ingest --dir data/filings/AAPL
"""

import argparse
import re
import sys
from pathlib import Path

from .service import get_rag_service
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Filename pattern: TICKER_FORM-TYPE_PERIOD-END.html
# e.g., AAPL_10-K_2024-09-28.html, TSLA_10-Q_2024-06-30.html
_FILENAME_PATTERN = re.compile(
    r"^([A-Z]+)_(10-[KQ])_(\d{4}-\d{2}-\d{2})\.html$"
)

# CIK mapping for downloaded filings
CIK_MAP = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
}


def ingest_all_filings(filings_dir: str | Path) -> dict:
    """Ingest all SEC filing HTML files from a directory.

    Walks the directory tree looking for files matching the pattern
    TICKER_FORM-TYPE_PERIOD-END.html (e.g., AAPL_10-K_2024-09-28.html).

    Args:
        filings_dir: Root directory containing filing HTML files.

    Returns:
        Summary dict with total_files, total_chunks, and per-ticker breakdown.
    """
    filings_dir = Path(filings_dir)
    if not filings_dir.exists():
        logger.error(f"Filings directory not found: {filings_dir}")
        return {"error": f"Directory not found: {filings_dir}"}

    rag = get_rag_service()
    if not rag.available:
        logger.error("OpenAI API key not configured — cannot embed")
        return {"error": "OpenAI API key not configured"}

    stats = {"total_files": 0, "total_chunks": 0, "by_ticker": {}}

    # Find all HTML files
    html_files = sorted(filings_dir.rglob("*.html"))
    if not html_files:
        logger.warning(f"No HTML files found in {filings_dir}")
        return stats

    print(f"\nFound {len(html_files)} HTML files in {filings_dir}\n")

    for filepath in html_files:
        match = _FILENAME_PATTERN.match(filepath.name)
        if not match:
            logger.warning(f"Skipping {filepath.name} — doesn't match naming pattern")
            continue

        ticker = match.group(1)
        filing_type = match.group(2)
        period_end = match.group(3)
        cik = CIK_MAP.get(ticker, "")

        if not cik:
            logger.warning(f"No CIK mapping for {ticker}, skipping {filepath.name}")
            continue

        print(f"  Ingesting: {ticker} {filing_type} {period_end} ...", end=" ", flush=True)

        try:
            chunk_count = rag.ingest_filing(
                filepath=filepath,
                ticker=ticker,
                cik=cik,
                filing_type=filing_type,
                filing_date=period_end,  # Use period_end as filing_date approximation
                period_end=period_end,
            )
            print(f"{chunk_count} chunks")

            stats["total_files"] += 1
            stats["total_chunks"] += chunk_count
            if ticker not in stats["by_ticker"]:
                stats["by_ticker"][ticker] = {"files": 0, "chunks": 0}
            stats["by_ticker"][ticker]["files"] += 1
            stats["by_ticker"][ticker]["chunks"] += chunk_count

        except Exception as e:
            print(f"FAILED: {e}")
            logger.error(f"Failed to ingest {filepath}: {e}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Ingestion Complete")
    print(f"{'='*50}")
    print(f"  Files processed: {stats['total_files']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    for ticker, info in sorted(stats["by_ticker"].items()):
        print(f"  {ticker}: {info['files']} files, {info['chunks']} chunks")
    print()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC filings into RAG vector store")
    parser.add_argument(
        "--dir",
        default="data/filings",
        help="Directory containing filing HTML files (default: data/filings)",
    )
    args = parser.parse_args()
    ingest_all_filings(args.dir)
