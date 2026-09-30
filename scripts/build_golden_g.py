#!/usr/bin/env python
"""Build golden_g.jsonl — the supervisory-grade evaluation set.

Labels are derived from SEC companyfacts using the SEC's own fy/fp/form/frame
tags. FinVet's retrieval (mcp/sec_edgar.py) never reads those fields — it
selects by end date and duration — so a label produced here and a value
FinVet retrieves come from two independent mechanisms. fill_sec_gold.py made
the same choice for the same reason: an answer key built with the code under
test passes by construction.

Usage:
    .venv/bin/python scripts/build_golden_g.py --audit-filers
    .venv/bin/python scripts/build_golden_g.py --out /path/scratch/numeric.jsonl
    .venv/bin/python scripts/build_golden_g.py --verify /path/scratch/numeric.jsonl
"""
import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finvet.config.constants import (  # noqa: E402  (Task 4: tolerance checks)
    SEC_LARGE_VALUE_THRESHOLD, TOLERANCE_APPROX_MULTIPLIER,
    TOLERANCE_SEC_LARGE_VALUES, TOLERANCE_SEC_SMALL_VALUES)
from finvet.config.metrics import METRIC_TO_CONCEPTS  # noqa: E402
from finvet.config.settings import settings  # noqa: E402

FILERS = Path(__file__).with_name("filers_golden_g.json")
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DELAY_S = 0.15
ANNUAL_DAYS = (330, 400)
QUARTER_DAYS = (75, 105)
REVENUE_PRIMARY = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _client() -> httpx.Client:
    if settings.sec_user_agent_is_placeholder:
        sys.exit("SEC_EDGAR_USER_AGENT is still the placeholder; set it in .env")
    return httpx.Client(headers={"User-Agent": settings.sec_edgar_user_agent},
                        timeout=30.0)


def resolve_ciks(client: httpx.Client, tickers: List[str]) -> Dict[str, str]:
    data = client.get(TICKER_URL).json()
    by_ticker = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    return {t: by_ticker[t] for t in tickers if t in by_ticker}


def fetch_facts(client: httpx.Client, cik: str) -> Dict[str, Any]:
    r = client.get(FACTS_URL.format(cik=cik))
    r.raise_for_status()
    time.sleep(DELAY_S)
    return r.json()


def _unit_rows(payload: Dict[str, Any], concept: str) -> List[Tuple[str, Dict]]:
    units = payload.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})
    return [(unit, row) for unit, rows in units.items() for row in rows]


def _days(row: Dict) -> Optional[int]:
    if not row.get("start"):
        return None  # instant (balance sheet)
    return (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days


def primary_fact(payload: Dict, concept: str, fy: int, fp: str) -> Optional[Tuple[str, Dict]]:
    """The fact the company itself reported as its fiscal (fy, fp).

    Within a single filing (accession), the primary fact is the latest-ending
    row; earlier-ending rows in the same accession are comparatives carried
    in that filing. A `frame` key is NOT required: the SEC attaches a
    period's frame to the most recent filing to report it, so an older
    accession's own current-period fact can lose its frame once a later
    filing re-reports the same period as a comparative -- requiring `frame`
    would then wrongly prefer that later comparative over the filing's own
    primary fact. Group candidates by accession, take each accession's own
    latest-ending row, then the latest-ending of those.
    """
    lo, hi = ANNUAL_DAYS if fp == "FY" else QUARTER_DAYS
    forms = ("10-K", "10-K/A") if fp == "FY" else ("10-Q", "10-Q/A")
    cands = []
    for unit, row in _unit_rows(payload, concept):
        if row.get("fy") != fy or row.get("fp") != fp or row.get("form") not in forms:
            continue
        d = _days(row)
        if d is not None and not (lo <= d <= hi):
            continue
        cands.append((unit, row))
    if not cands:
        return None
    by_accn: Dict[str, Tuple[str, Dict]] = {}
    for unit, row in cands:
        accn = row["accn"]
        if accn not in by_accn or row["end"] > by_accn[accn][1]["end"]:
            by_accn[accn] = (unit, row)
    return max(by_accn.values(), key=lambda ur: ur[1]["end"])


def restated_values(payload: Dict, concept: str, years_back: int = 3) -> Dict[Tuple[Optional[str], str], List[Dict]]:
    """Annual periods whose filed value changed between filings.

    Keyed on (start, end), never on fy: a restated prior year is re-reported
    in the later 10-K as a comparative tagged with the LATER fy, so an
    fy-keyed check cannot see it. Returns {(start, end): [one row per
    distinct val]} for the most recent `years_back` period ends that carry
    more than one distinct value.
    """
    by_period: Dict[Tuple[Optional[str], str], Dict[float, Dict]] = {}
    for _unit, row in _unit_rows(payload, concept):
        if row.get("form") not in ("10-K", "10-K/A") or row.get("fp") != "FY":
            continue
        d = _days(row)
        if d is not None and not (ANNUAL_DAYS[0] <= d <= ANNUAL_DAYS[1]):
            continue
        key = (row.get("start"), row["end"])
        by_period.setdefault(key, {}).setdefault(row["val"], row)
    ends = sorted({k[1] for k in by_period}, reverse=True)[:years_back]
    return {k: list(v.values()) for k, v in by_period.items()
            if k[1] in ends and len(v) > 1}


def max_restatement_pct(payload: Dict, concept: str) -> float:
    """Largest relative change between two filed values of the same period, in %."""
    worst = 0.0
    for rows in restated_values(payload, concept).values():
        vals = [abs(r["val"]) for r in rows if r["val"]]
        if len(vals) > 1:
            worst = max(worst, (max(vals) - min(vals)) / max(vals) * 100)
    return worst


def latest_fy(payload: Dict) -> Optional[int]:
    years = [r.get("fy") for _u, r in _unit_rows(payload, "Assets")
             if r.get("fp") == "FY" and r.get("form") in ("10-K", "10-K/A") and r.get("fy")]
    return max(years) if years else None


def fye_month(payload: Dict, fy: int) -> Optional[int]:
    fact = primary_fact(payload, "Assets", fy, "FY")
    return int(fact[1]["end"][5:7]) if fact else None


def revenue_concept(payload: Dict, fy: int) -> Optional[str]:
    for concept in METRIC_TO_CONCEPTS["revenue"]:
        if primary_fact(payload, concept, fy, "FY"):
            return concept
    return None


METRIC_PHRASE = {
    "revenue": "total revenue", "net_income": "net income", "total_assets": "total assets",
    "diluted_eps": "diluted EPS", "eps": "basic EPS", "operating_income": "operating income",
    "operating_cash_flow": "operating cash flow", "shareholders_equity": "shareholders' equity",
    "total_liabilities": "total liabilities", "gross_profit": "gross profit",
    "research_and_development": "research and development expense",
    "cost_of_revenue": "cost of revenue", "capex": "capital expenditures",
    "interest_expense": "interest expense", "total_debt": "long-term debt",
}
EPS_METRICS = frozenset({"eps", "diluted_eps"})
# Metrics used per class. Financials have no gross profit / cost of revenue.
CLASS_METRICS = {
    "financials": ["net_income", "total_assets", "diluted_eps", "shareholders_equity"],
    "default": ["revenue", "net_income", "total_assets", "operating_cash_flow",
                "diluted_eps", "operating_income"],
}
OPERATOR_PHRASE = {"gt": "exceeded", "gte": "was at least", "lt": "was below",
                   "lte": "was at most", "approx": "was approximately"}


def tolerance_for(value: float) -> float:
    return (TOLERANCE_SEC_LARGE_VALUES if abs(value) >= SEC_LARGE_VALUE_THRESHOLD
            else TOLERANCE_SEC_SMALL_VALUES)


def magnitude_diff(claimed: float, filed: float) -> float:
    divisor = max(abs(claimed), abs(filed))
    return abs(claimed - filed) / divisor * 100 if divisor else 0.0


def format_amount(value: float, metric: str, sig: int = 3) -> Tuple[float, str]:
    """A readable dollar string and the exact number it denotes.

    Never emits a bare run of 9+ digits: the input guard's SSN regex would
    block the claim before any model ran.
    """
    sign = "-" if value < 0 else ""
    v = abs(value)
    if metric in EPS_METRICS:
        rounded = round(v, 2)
        return (-rounded if value < 0 else rounded), f"{sign}${rounded:.2f}"
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if v >= scale:
            digits = max(sig - 1 - int(math.floor(math.log10(v / scale))), 0)
            num = round(v / scale, digits)
            text = f"{sign}${num:.{digits}f} {word}"
            recon = round(num * scale)
            return (-recon if value < 0 else recon), text
    num = round(v)
    return (-num if value < 0 else num), f"{sign}${num:,}"


def claim_value(filed: float, metric: str, side: str, factor: float,
                sig: int = 3, band_multiplier: float = 1.0) -> Tuple[float, str]:
    """A claim number `factor` × tolerance away from the filed value.

    `side="in"` places it inside the band (SUPPORTS), `"out"` outside
    (REFUTES). The distance is taken below the filed value so the magnitude
    difference equals the intended percentage exactly. After rounding to
    `sig` significant figures the side is re-checked; if rounding crossed the
    line, one more digit is used.

    `band_multiplier` widens the accept band the "in"/"out" check is judged
    against (e.g. TOLERANCE_APPROX_MULTIPLIER for an "approx" operator claim,
    which SUPPORTS within 2×tol rather than 1×tol) without changing where
    the target itself is placed.
    """
    tol = tolerance_for(filed)
    band = tol * band_multiplier
    target = filed * (1 - factor * tol / 100)
    for s in (sig, sig + 1, sig + 2):
        value, text = format_amount(target, metric, sig=s)
        diff = magnitude_diff(value, filed)
        if (side == "in" and diff <= band * 0.98) or (side == "out" and diff > band * 1.02):
            return value, text
    raise ValueError(f"could not place {metric} {filed} {side} at {factor}×tol")


def period_text(fp: str, fy: int) -> str:
    return f"fiscal {fy}" if fp == "FY" else f"{fp} {fy}"


def render_claim(name: str, metric: str, text: str, fp: str, fy: int,
                 operator: str = "eq", value: float = 0.0) -> str:
    when = period_text(fp, fy)
    if value < 0:
        return f"{name} reported a net loss of {text.lstrip('-')} in {when}"
    if operator == "eq":
        return f"{name}'s {METRIC_PHRASE[metric]} was {text} in {when}"
    return f"{name}'s {METRIC_PHRASE[metric]} {OPERATOR_PHRASE[operator]} {text} in {when}"


def make_row(rid: int, filer: Dict, metric: str, concept: str, fact: Dict, unit: str,
             fp: str, fy: int, verdict: str, value: float, text: str,
             operator: str, seed: str, extra_truth: str = "") -> Dict[str, Any]:
    fact_year = int(fact["end"][:4])
    valid_years = (fy, fy + 1) if fp == "FY" else (fy - 1, fy, fy + 1)
    if fact_year not in valid_years:
        raise ValueError(f"{filer['ticker']} {metric} {fp} {fy}: fact ends {fact['end']}, "
                         f"inconsistent with the fiscal label")
    filed = fact["val"]
    diff = magnitude_diff(value, filed)
    tol = tolerance_for(filed)
    claim = render_claim(filer["name"], metric, text, fp, fy, operator, value=value)
    band = f"{diff:.4f}% from the filed value, {'inside' if verdict == 'SUPPORTS' else 'outside'} the {tol}% band"
    if operator in ("gt", "gte", "lt", "lte"):
        band = f"filed {filed:,} vs claimed bound {value:,} ({operator})"
    if operator == "approx":
        band = f"{diff:.4f}% from the filed value, inside the widened {tol * TOLERANCE_APPROX_MULTIPLIER}% band"
    return {
        "id": rid,
        "claim": claim,
        "category": "sec/tolerance" if seed.startswith("boundary") else
                    ("sec/operator" if operator != "eq" else "sec/xbrl"),
        "strength": "strict",
        "expected": {"verdict": verdict, "limitation": None, "sources": ["xbrl"]},
        "ground_truth": f"filed {filed:,} ({unit}) period_end {fact['end']}; {band}{extra_truth}",
        "source": f"{filer['ticker']} {period_text(fp, fy)} {fact['form']} accn {fact['accn']}, us-gaap:{concept}, frame {fact.get('frame')}",
        "gold_parse": {"claim_type": "sec", "ticker": filer["ticker"], "metric": metric,
                       "operator": operator, "value": value,
                       "period": period_text(fp, fy).replace("fiscal ", "FY"),
                       "reject_reason": None, "label_source": "derived_from_source"},
        "tags": {"class": filer["class"], "seed": seed},
    }


def _pick_metric(payload: Dict, filer: Dict, fy: int, avoid: set) -> Optional[Tuple[str, str, str, Dict]]:
    metrics = CLASS_METRICS["financials" if filer["class"] == "financials" else "default"]
    for metric in metrics:
        if metric in avoid:
            continue
        for concept in METRIC_TO_CONCEPTS[metric]:
            found = primary_fact(payload, concept, fy, "FY")
            if found and found[1]["val"]:
                return metric, concept, found[0], found[1]
    return None


def _pick_restated(payload: Dict, filer: Dict) -> Optional[Tuple[str, str, Tuple, List[Dict]]]:
    """The most recent restated period for a filer, over CLASS_METRICS['default'].

    Ruling 4: `restated_values` returns {(start, end): [rows]} keyed on the
    period, not the fy, so the "most recent" period is chosen by end date,
    not by fy. Returns (metric, concept, (start, end), rows) for the first
    metric/concept whose restated_values is non-empty.
    """
    for metric in CLASS_METRICS["default"]:
        for concept in METRIC_TO_CONCEPTS[metric]:
            periods = restated_values(payload, concept)
            if periods:
                (start, end), rows = max(periods.items(), key=lambda kv: kv[0][1])
                return metric, concept, (start, end), rows
    return None


def generate(out: Path) -> int:
    filers = json.loads(FILERS.read_text())
    rows: List[Dict[str, Any]] = []
    rid = 1001
    seeds_quarter = seeds_operator = 0
    operators = ["gt", "lt", "gte", "lte", "approx"]
    turned_profitable: List[str] = []
    quarter_fallback: List[str] = []
    with _client() as client:
        ciks = resolve_ciks(client, [f["ticker"] for f in filers])
        for f in filers:
            p = fetch_facts(client, ciks[f["ticker"]])
            fy = latest_fy(p)
            used: set = set()
            loss_fact = None
            if "loss-making" in f.get("note", ""):
                for concept in METRIC_TO_CONCEPTS["net_income"]:
                    ff = primary_fact(p, concept, fy, "FY")
                    if ff and ff[1]["val"] < 0:
                        loss_fact = ("net_income", concept, ff[0], ff[1])
                        break
                if loss_fact is None:
                    # The filer is tagged loss-making but its latest FY is
                    # profitable — keep the filer, generate it like any
                    # other mid_cap filer instead of forcing the "loss" seed.
                    turned_profitable.append(f["ticker"])
            is_loss = loss_fact is not None
            calendar = fye_month(p, fy) == 12

            if f["class"] == "restated":
                # Ruling 5: both rows are generated on the SAME most-recent
                # restated period, so the test exercises exactly the
                # hardness the class exists for.
                found = _pick_restated(p, f)
                assert found, f"{f['ticker']} is tagged restated but has no restated period"
                metric, concept, (start, end), period_rows = found
                rows_sorted = sorted(period_rows, key=lambda r: r.get("filed", ""))
                original, latest = rows_sorted[0], rows_sorted[-1]
                unit = "USD/shares" if metric in EPS_METRICS else "USD"
                fy_label = original["fy"]
                extra_truth = (
                    f"; restated: originally {original['val']:,} in {original['accn']} "
                    f"({original['form']}, filed {original.get('filed')}), restated to "
                    f"{latest['val']:,} in {latest['accn']} ({latest['form']}, filed "
                    f"{latest.get('filed')})"
                )
                v_in, t_in = claim_value(latest["val"], metric, "in", 0.3)
                rows.append(make_row(rid, f, metric, concept, latest, unit, "FY", fy_label,
                                     "SUPPORTS", v_in, t_in, "eq", "restated", extra_truth))
                rid += 1
                v_out, t_out = claim_value(latest["val"], metric, "out", 4.0)
                rows.append(make_row(rid, f, metric, concept, latest, unit, "FY", fy_label,
                                     "REFUTES", v_out, t_out, "eq", "restated", extra_truth))
                rid += 1
                continue

            # Row 1: SUPPORTS (plain, or loss / quarter seeding)
            if is_loss:
                metric, concept, unit, fact = loss_fact
                value, text = claim_value(fact["val"], metric, "in", 0.3)
                rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                     "SUPPORTS", value, text, "eq", "loss"))
            elif calendar and seeds_quarter < 30:
                fp = ["Q1", "Q2", "Q3"][seeds_quarter % 3]
                found = None
                for metric in ("revenue", "net_income", "operating_income"):
                    for concept in METRIC_TO_CONCEPTS[metric]:
                        for qfy in (fy, fy - 1):
                            ff = primary_fact(p, concept, qfy, fp)
                            if ff:
                                found = (metric, concept, ff[0], ff[1], qfy)
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    metric, concept, unit, fact, qfy = found
                    value, text = claim_value(fact["val"], metric, "in", 0.3)
                    rows.append(make_row(rid, f, metric, concept, fact, unit, fp, qfy,
                                         "SUPPORTS", value, text, "eq", "quarter"))
                    seeds_quarter += 1
                else:
                    # No quarterly primary fact for revenue/net_income/
                    # operating_income under this filer's usual concepts
                    # (e.g. a REIT with no standard quarterly revenue tag) —
                    # fall back to the plain annual path instead of crashing.
                    quarter_fallback.append(f["ticker"])
                    found2 = _pick_metric(p, f, fy, used)
                    assert found2, f"{f['ticker']} has no primary fact for any class metric"
                    metric, concept, unit, fact = found2
                    value, text = claim_value(fact["val"], metric, "in", 0.3)
                    rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                         "SUPPORTS", value, text, "eq", "plain"))
            else:
                found = _pick_metric(p, f, fy, used)
                assert found, f"{f['ticker']} has no primary fact for any class metric"
                metric, concept, unit, fact = found
                value, text = claim_value(fact["val"], metric, "in", 0.3)
                rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                     "SUPPORTS", value, text, "eq", "plain"))
            used.add(rows[-1]["gold_parse"]["metric"])
            rid += 1

            # Row 2: REFUTES (plain, or operator seeding)
            found = _pick_metric(p, f, fy, used)
            assert found, f"{f['ticker']} has no second primary fact for any class metric"
            metric, concept, unit, fact = found
            if seeds_operator < 20 and not is_loss:
                op = operators[seeds_operator % len(operators)]
                if op == "approx":
                    value, text = claim_value(fact["val"], metric, "in", 1.8,
                                              band_multiplier=TOLERANCE_APPROX_MULTIPLIER)
                    verdict = "SUPPORTS"
                elif op in ("gt", "gte"):
                    value, text = claim_value(fact["val"], metric, "out", 4.0)  # bound below filed → true
                    verdict = "SUPPORTS"
                else:  # lt, lte: bound below filed → false
                    value, text = claim_value(fact["val"], metric, "out", 4.0)
                    verdict = "REFUTES"
                rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                     verdict, value, text, op, "operator"))
                seeds_operator += 1
            else:
                value, text = claim_value(fact["val"], metric, "out", 4.0)
                rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                     "REFUTES", value, text, "eq", "plain"))
            rid += 1

        # Boundary: first 10 filers that are not loss-making and not restated
        # (restated filers take neither the quarter/operator nor the boundary
        # seeding; both of their rows are already generated above), one
        # metric each.
        boundary_pool = [x for x in filers
                         if "loss-making" not in x.get("note", "") and x["class"] != "restated"]
        for f in boundary_pool[:10]:
            p = fetch_facts(client, ciks[f["ticker"]])
            fy = latest_fy(p)
            found = _pick_metric(p, f, fy, set())
            assert found, f"{f['ticker']} has no primary fact for boundary seeding"
            metric, concept, unit, fact = found
            v_in, t_in = claim_value(fact["val"], metric, "in", 0.85, sig=4)
            rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                 "SUPPORTS", v_in, t_in, "eq", "boundary_in"))
            rid += 1
            v_out, t_out = claim_value(fact["val"], metric, "out", 1.15, sig=4)
            rows.append(make_row(rid, f, metric, concept, fact, unit, "FY", fy,
                                 "REFUTES", v_out, t_out, "eq", "boundary_out"))
            rid += 1

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    if turned_profitable:
        print(f"note: tagged loss-making but latest FY is profitable, generated as "
              f"non-loss instead: {', '.join(turned_profitable)}")
    if quarter_fallback:
        print(f"note: no quarterly primary fact under usual concepts, generated as "
              f"plain annual instead: {', '.join(quarter_fallback)}")
    return 0


def _recompute_verdict(gold_parse: Dict, refetched_val: float) -> str:
    """The verdict a fresh fact would earn against this row's own claim."""
    op = gold_parse["operator"]
    claimed = gold_parse["value"]
    if op in ("gt", "gte", "lt", "lte"):
        ok = {"gt": refetched_val > claimed, "gte": refetched_val >= claimed,
              "lt": refetched_val < claimed, "lte": refetched_val <= claimed}[op]
        return "SUPPORTS" if ok else "REFUTES"
    tol = tolerance_for(refetched_val) * (2.0 if op == "approx" else 1.0)
    return "SUPPORTS" if magnitude_diff(claimed, refetched_val) <= tol else "REFUTES"


def _check_row(r: Dict, refetched: Dict, period_end: str, accn_token: str,
              filed_label: float) -> List[str]:
    """Every mismatch between a row's recorded label and a fresh fact."""
    problems = []
    if abs(refetched["val"] - filed_label) > 1e-6:
        problems.append(f"value: label {filed_label} refetch {refetched['val']}")
    if refetched["end"] != period_end:
        problems.append(f"end: label {period_end} refetch {refetched['end']}")
    if refetched["accn"] != accn_token:
        problems.append(f"accn: label {accn_token} refetch {refetched['accn']}")
    recomputed = _recompute_verdict(r["gold_parse"], refetched["val"])
    if recomputed != r["expected"]["verdict"]:
        problems.append(f"verdict: label {r['expected']['verdict']} recomputed {recomputed}")
    return problems


def verify(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = 0
    with _client() as client:
        ciks = resolve_ciks(client, sorted({r["gold_parse"]["ticker"] for r in rows}))
        cache: Dict[str, Dict] = {}
        for r in rows:
            t = r["gold_parse"]["ticker"]
            cache.setdefault(t, fetch_facts(client, ciks[t]))
            src = r["source"]
            concept = src.split("us-gaap:")[1].split(",")[0]
            accn_token = src.split("accn ")[1].split(",")[0].strip()
            period_end = r["ground_truth"].split("period_end ")[1].split(";")[0].strip()
            filed = float(r["ground_truth"].split("filed ")[1].split(" ")[0].replace(",", ""))
            if r["tags"]["seed"] == "restated":
                periods = restated_values(cache[t], concept)
                match = None
                for (_start, end), period_rows in periods.items():
                    if end == period_end:
                        match = period_rows
                        break
                if match is None:
                    bad += 1
                    print(f"MISMATCH id {r['id']}: NO PERIOD MATCH ({period_end})")
                    continue
                latest = sorted(match, key=lambda row: row.get("filed", ""))[-1]
                problems = _check_row(r, latest, period_end, accn_token, filed)
                if problems:
                    bad += 1
                    print(f"MISMATCH id {r['id']}: " + "; ".join(problems))
                continue
            parts = src.split()
            fp = "FY" if parts[1] == "fiscal" else parts[1]
            fy = int(parts[2])
            found = primary_fact(cache[t], concept, fy, fp)
            if not found:
                bad += 1
                print(f"MISMATCH id {r['id']}: NO FACT FOUND")
                continue
            problems = _check_row(r, found[1], period_end, accn_token, filed)
            if problems:
                bad += 1
                print(f"MISMATCH id {r['id']}: " + "; ".join(problems))
    print(f"{len(rows) - bad}/{len(rows)} labels re-fetch identically")
    return 1 if bad else 0


def audit_filers() -> int:
    filers = json.loads(FILERS.read_text())
    with _client() as client:
        ciks = resolve_ciks(client, [f["ticker"] for f in filers])
        print(f"{'ticker':<6} {'class':<20} {'cik':<11} {'fy':>4} {'fye':>3} "
              f"{'revenue concept':<52} {'restate%':>8}")
        for f in filers:
            cik = ciks.get(f["ticker"])
            if not cik:
                print(f"{f['ticker']:<6} {f['class']:<20} {'NO CIK':<11}")
                continue
            p = fetch_facts(client, cik)
            fy = latest_fy(p)
            if not fy:
                print(f"{f['ticker']:<6} {f['class']:<20} {cik:<11} {'no 10-K':>4}")
                continue
            rc = revenue_concept(p, fy) or "-"
            restate_concept = rc if rc != "-" else "NetIncomeLoss"
            restate_pct = max_restatement_pct(p, restate_concept)
            print(f"{f['ticker']:<6} {f['class']:<20} {cik:<11} {fy:>4} "
                  f"{fye_month(p, fy) or 0:>3} {rc:<52} {restate_pct:>7.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-filers", action="store_true")
    ap.add_argument("--out", help="write derived numeric rows here (Task 4)")
    ap.add_argument("--verify", help="re-fetch every value in this JSONL and diff (Task 4)")
    args = ap.parse_args()
    if args.audit_filers:
        return audit_filers()
    if args.out:
        return generate(Path(args.out))
    if args.verify:
        return verify(Path(args.verify))
    ap.error("one of --audit-filers, --out, --verify is required")


if __name__ == "__main__":
    raise SystemExit(main())
