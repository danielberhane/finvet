"""Which filed number answers a claim about a fiscal period.

Pure functions over SEC companyfacts. No network, no model, and no filing
"under inspection": the agent chooses which filing to open, so a selection
that took that choice as an input would let the model decide the evidence.

Four rules, each from a defect found in SEC's own data:

1. Only financial statements supply a number. FedEx's proxy statement tags
   net income as 4,433 -- millions -- under the same concept as the 10-K's
   4,433,000,000, and MetLife's tags income available to common shareholders
   as net income.
2. The issuer names its own fiscal year. Home Depot's fiscal 2025 ends
   2026-02-01; Walmart's year ending 2026-01-31 is its fiscal 2026. No rule
   about dates can tell those apart. The filing labelled fiscal 2025 can.
3. When statements disagree on a period, the most recently filed is the value
   of record, and the earlier ones are kept. A restatement is the issuer
   telling the SEC the earlier figure no longer describes the period.
4. Anything that cannot be established is declined with a reason.
"""

from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from pydantic import BaseModel

# Forms that carry audited or reviewed financial statements. Everything else
# -- proxy statements, 8-Ks, registration statements -- may tag a us-gaap
# concept without being bound by its definition, its scale, or its period.
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "10-KT/A",
                          "20-F", "20-F/A", "40-F", "40-F/A"})
QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A", "10-QT", "10-QT/A"})
STATEMENT_FORMS = ANNUAL_FORMS | QUARTERLY_FORMS

# A fiscal year is 52 or 53 weeks; a fiscal quarter 12 to 14.
ANNUAL_DAYS = (330, 400)
QUARTER_DAYS = (75, 105)

# The flows whose reporting period defines the period a filing is about.
# Deliberately a short list of statement totals: a filing can carry
# undimensioned duration facts for other spans, and the fiscal year must not
# be read off one of those.
CALENDAR_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "NetIncomeLoss",
    "ProfitLoss",
    "OperatingIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "NetCashProvidedByUsedInOperatingActivities",
)


class FiscalPeriod(BaseModel):
    """The dates an issuer's own filing gives a fiscal year or quarter."""
    start: Optional[str] = None
    end: str


class Superseded(BaseModel):
    """A value an earlier statement reported for a period since restated."""
    value: float
    accession: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None


class Selection(BaseModel):
    """The value of record for one concept in one fiscal period."""
    value: float
    start: Optional[str] = None
    end: str
    accession: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None
    fiscal_year: int
    fiscal_period: str
    superseded: List[Superseded] = []


class Decline(BaseModel):
    """Why no number is offered. One of:

    concept_not_reported          the issuer files nothing under this concept
    fiscal_period_not_filed       no statement is labelled with this period
    fiscal_period_ambiguous       statements labelled with it disagree on its dates
    no_statement_fact_for_period  the concept has no statement fact for those dates
    """
    reason: str


# None marks a period whose filings disagree on its dates.
Calendar = Dict[Tuple[int, str], Optional[FiscalPeriod]]


def is_statement_fact(fact: Dict[str, Any]) -> bool:
    return fact.get("form") in STATEMENT_FORMS


def _facts(payload: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for unit in ((payload or {}).get("units") or {}).values():
        for fact in unit or []:
            if isinstance(fact, dict) and fact.get("val") is not None and fact.get("end"):
                yield fact


def _days(fact: Dict[str, Any]) -> Optional[int]:
    """Days a duration fact covers; None for an instant."""
    if not fact.get("start"):
        return None
    try:
        return (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    except (TypeError, ValueError):
        return None


def _span_for(fiscal_period: str) -> Tuple[int, int]:
    return ANNUAL_DAYS if fiscal_period == "FY" else QUARTER_DAYS


def _forms_for(fiscal_period: str) -> frozenset:
    return ANNUAL_FORMS if fiscal_period == "FY" else QUARTERLY_FORMS


def _filed_order(fact: Dict[str, Any]) -> Tuple[str, str]:
    """Recency by the date SEC received the filing. Accession prefixes belong
    to filing agents and do not order filings; they only break ties."""
    return (fact.get("filed") or "", fact.get("accn") or "")


def build_fiscal_calendar(us_gaap: Dict[str, Any]) -> Calendar:
    """The dates of every fiscal period the issuer has filed a statement for.

    SEC stamps each fact with the fiscal year and period of the FILING it came
    from, and a filing also carries its comparative periods under that same
    stamp. The period a filing is about is therefore its latest-ending flow of
    the right length -- a year for a 10-K, a quarter for a 10-Q -- that ended
    before the filing was made.
    """
    about: Dict[Tuple[int, str, str], List[Tuple[str, str]]] = defaultdict(list)
    for concept in CALENDAR_CONCEPTS:
        for fact in _facts(us_gaap.get(concept)):
            fy, fp = fact.get("fy"), fact.get("fp")
            if not isinstance(fy, int) or fp not in ("FY", "Q1", "Q2", "Q3"):
                continue
            if fact.get("form") not in _forms_for(fp):
                continue
            days = _days(fact)
            low, high = _span_for(fp)
            if days is None or not low <= days <= high:
                continue
            # A report cannot describe a period that ends after it was filed.
            if fact.get("filed") and fact["end"] > fact["filed"]:
                continue
            about[(fy, fp, fact.get("accn") or "")].append((fact["start"], fact["end"]))

    periods: Dict[Tuple[int, str], set] = defaultdict(set)
    for (fy, fp, _accn), spans in about.items():
        latest_end = max(end for _, end in spans)
        starts = Counter(start for start, end in spans if end == latest_end)
        periods[(fy, fp)].add((starts.most_common(1)[0][0], latest_end))

    calendar: Calendar = {}
    for key, found in periods.items():
        if key[1] == "FY":
            calendar[key] = _settle(found)

    # Filers mislabel. Target stamped two different quarters as fiscal 2023 Q3,
    # a year apart. A quarter lies inside its own fiscal year, so the year --
    # or, for a year still in progress, the end of the one before -- says
    # which filing to believe. Where neither is known the stamp stands alone.
    for key, found in periods.items():
        fy, fp = key
        if fp == "FY":
            continue
        bounds = _year_bounds(calendar, fy)
        if bounds is not None:
            found = {(start, end) for start, end in found
                     if bounds[0] < end <= bounds[1]}
            if not found:
                continue
        calendar[key] = _settle(found)
    return calendar


def _settle(found: set) -> Optional[FiscalPeriod]:
    """One period if the filings agree on when it ended, None if they do not."""
    ends = {end for _, end in found}
    if len(ends) != 1:
        return None
    return FiscalPeriod(start=min(start for start, _ in found), end=ends.pop())


def _year_bounds(calendar: Calendar, fiscal_year: int) -> Optional[Tuple[str, str]]:
    """(exclusive start, inclusive end) a quarter of this fiscal year ends within."""
    year = calendar.get((fiscal_year, "FY"))
    if year is not None and year.start:
        return year.start, year.end
    before = calendar.get((fiscal_year - 1, "FY"))
    if before is None:
        return None
    try:
        closing = date.fromisoformat(before.end)
    except ValueError:
        return None
    # 53 weeks, the longest a fiscal year runs.
    return before.end, (closing + timedelta(days=371)).isoformat()


def select_fact(payload: Optional[Dict[str, Any]], calendar: Calendar,
                fiscal_year: int, fiscal_period: str) -> Union[Selection, Decline]:
    """The value of record for a concept in a fiscal period, or why there is none."""
    if not payload:
        return Decline(reason="concept_not_reported")
    key = (fiscal_year, fiscal_period)
    if key not in calendar:
        return Decline(reason="fiscal_period_not_filed")
    period = calendar[key]
    if period is None:
        return Decline(reason="fiscal_period_ambiguous")

    low, high = _span_for(fiscal_period)
    candidates = []
    for fact in _facts(payload):
        if not is_statement_fact(fact) or fact["end"] != period.end:
            continue
        days = _days(fact)
        # An instant has no span to check: a balance-sheet figure at the
        # period's closing date. A duration must cover the period asked for,
        # or the fourth quarter -- which shares the year's end date -- would
        # stand in for the year.
        if days is not None and not low <= days <= high:
            continue
        candidates.append(fact)
    if not candidates:
        return Decline(reason="no_statement_fact_for_period")

    record = max(candidates, key=_filed_order)
    superseded: List[Superseded] = []
    for fact in sorted(candidates, key=_filed_order):
        if fact["val"] == record["val"]:
            continue
        if any(s.value == float(fact["val"]) for s in superseded):
            continue
        superseded.append(Superseded(value=float(fact["val"]), accession=fact.get("accn"),
                                     form=fact.get("form"), filed=fact.get("filed")))

    return Selection(
        value=float(record["val"]),
        start=record.get("start"),
        end=record["end"],
        accession=record.get("accn"),
        form=record.get("form"),
        filed=record.get("filed"),
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        superseded=superseded,
    )
