"""Fills missing source values in the SEC gold rows from SEC's frames API.

72 rows of the real-sourced held-out set name a company, an XBRL concept, a
period and an accession, but never recorded the value the company filed.
Filling them takes the retrieval harness from 128 scoreable cases to 200.

Uses frames rather than companyconcept. FinVet reads companyconcept and picks a
fact with _select_fact_for_period(), so generating the answer key with that same
code would make the retrieval test pass by construction and bake any selection
bug into the ground truth. frames is a different endpoint, keyed by the
xbrl_frame the dataset already records, where SEC performs the period
normalisation itself. One response carries every filer, so 72 rows collapse to
about 19 network calls.

The frozen gold file is never modified. Results go to a sidecar keyed by row id,
and each entry records label_source so a filled value cannot be confused with
one the dataset authors sourced themselves.

Usage:
    python -m finvet.eval.fill_sec_gold --out data/sec_gold_fill.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import httpx

from ..config.settings import settings
from ..utils.logging import get_logger
from .xbrl_retrieval import DEFAULT_GOLD

logger = get_logger(__name__)

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/{unit}/{frame}.json"

# frames requires a unit in the path and it varies by concept: monetary values
# are USD, per-share figures USD-per-shares, ratios pure. Tried in this order.
UNIT_CANDIDATES = ("USD", "USD-per-shares", "pure")

# Marks a value as produced here rather than by the dataset authors, whose rows
# carry "xbrl_companyfacts". Never collapse the two.
LABEL_SOURCE = "xbrl_frames"


def needs_fill(row: Dict[str, Any]) -> bool:
    """True when a row names a concept and frame but has no recorded value."""
    prov = row.get("provenance") or {}
    return (
        bool(prov.get("xbrl_fact"))
        and bool(prov.get("xbrl_frame"))
        and prov.get("source_value_exact") is None
    )


def _concept(row: Dict[str, Any]) -> str:
    return row["provenance"]["xbrl_fact"].split(":")[-1]


def frames_needed(rows: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """Distinct (concept, frame) lookups. One response covers every filer."""
    return {
        (_concept(r), r["provenance"]["xbrl_frame"])
        for r in rows
        if needs_fill(r)
    }


def frame_url(concept: str, unit: str, frame: str) -> str:
    return FRAMES_URL.format(concept=concept, unit=unit, frame=frame)


def find_company_value(payload: Dict[str, Any], cik: str) -> Optional[Dict[str, Any]]:
    """The entry for one filer. Gold stores CIKs zero-padded; frames returns ints."""
    want = str(cik).lstrip("0").zfill(10)
    for entry in (payload or {}).get("data", []):
        if str(entry.get("cik", "")).lstrip("0").zfill(10) == want:
            return entry
    return None


def build_entry(
    row: Dict[str, Any], value: float, unit: str, accn: Optional[str]
) -> Dict[str, Any]:
    """A sidecar record: the value plus everything needed to audit where it came from."""
    prov = row["provenance"]
    return {
        "row_id": row["id"],
        "cik": prov["cik"],
        "concept": _concept(row),
        "frame": prov["xbrl_frame"],
        "period_end": prov.get("period_end"),
        "source_value_exact": value,
        "unit": unit,
        "frames_accn": accn,
        "gold_accn": prov.get("accession"),
        "label_source": LABEL_SOURCE,
        "filled_at": time.strftime("%Y-%m-%d"),
    }


def load_sidecar(path: Optional[Path]) -> Dict[int, float]:
    """row_id -> filled value. A missing sidecar simply means nothing was filled."""
    if not path or not Path(path).exists():
        return {}
    out: Dict[int, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)
                out[entry["row_id"]] = entry["source_value_exact"]
    return out


def fetch_frame(
    concept: str, frame: str, client: httpx.Client
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Fetch a frame, trying each unit. Returns (unit, payload) or None."""
    for unit in UNIT_CANDIDATES:
        resp = client.get(frame_url(concept, unit, frame))
        if resp.status_code == 200:
            return unit, resp.json()
        if resp.status_code != 404:
            logger.warning(f"frames {concept}/{unit}/{frame}: HTTP {resp.status_code}")
    return None


def fill(
    gold_path: Path, out_path: Path, delay: float = 0.2
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [json.loads(line) for line in open(gold_path) if line.strip()]
    targets = [r for r in rows if needs_fill(r)]
    lookups = sorted(frames_needed(targets))

    print(f"{len(targets)} row(s) to fill, {len(lookups)} frame lookup(s)\n")

    cache: Dict[Tuple[str, str], Optional[Tuple[str, Dict[str, Any]]]] = {}
    client = httpx.Client(
        headers={"User-Agent": settings.sec_edgar_user_agent}, timeout=30.0
    )
    try:
        for i, (concept, frame) in enumerate(lookups, 1):
            cache[(concept, frame)] = fetch_frame(concept, frame, client)
            found = cache[(concept, frame)] is not None
            print(f"  [{i}/{len(lookups)}] {concept:<46} {frame:<10} "
                  f"{'ok' if found else 'NO FRAME'}")
            if delay:
                time.sleep(delay)
    finally:
        client.close()

    entries: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for row in targets:
        hit = cache.get((_concept(row), row["provenance"]["xbrl_frame"]))
        if not hit:
            missing.append({"row_id": row["id"], "reason": "frame_unavailable",
                            "concept": _concept(row)})
            continue
        unit, payload = hit
        entry = find_company_value(payload, row["provenance"]["cik"])
        if entry is None or entry.get("val") is None:
            missing.append({"row_id": row["id"], "reason": "company_not_in_frame",
                            "concept": _concept(row)})
            continue
        entries.append(build_entry(row, float(entry["val"]), unit, entry.get("accn")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    # Where the frames accession differs from the one the gold names, both
    # describe the same period — SEC attributes a fact to whichever filing it
    # reads it from. Surfaced rather than hidden.
    accn_differs = sum(1 for e in entries if e["frames_accn"] != e["gold_accn"])
    summary = {
        "targets": len(targets),
        "lookups": len(lookups),
        "filled": len(entries),
        "unfilled": len(missing),
        "accession_differs": accn_differs,
        "missing": missing,
    }
    return entries, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fill missing SEC gold values from SEC's frames API"
    )
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=Path("data/sec_gold_fill.jsonl"))
    ap.add_argument("--delay", type=float, default=0.2,
                    help="Seconds between lookups, for SEC fair access")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Gold set not found: {args.gold}")
        return 2

    entries, summary = fill(args.gold, args.out, args.delay)

    print(f"\n{'=' * 70}\nSEC GOLD FILL\n{'=' * 70}")
    print(f"  rows needing a value : {summary['targets']}")
    print(f"  frame lookups        : {summary['lookups']}")
    print(f"  filled               : {summary['filled']}")
    print(f"  unfilled             : {summary['unfilled']}")
    if summary["accession_differs"]:
        print(f"  accession differs from gold: {summary['accession_differs']}"
              f"  (same period, different source filing)")
    if summary["missing"]:
        print("\n  unfilled rows:")
        for m in summary["missing"]:
            print(f"    row {m['row_id']:<5} {m['concept']:<46} {m['reason']}")
    print(f"\n  written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
