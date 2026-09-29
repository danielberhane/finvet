"""FRED client for the macro series behind the news agent's numeric metrics.

Eight macro metrics from the news whitelist map to FRED series IDs, taken from
the provenance URLs in the gold dataset and checked against its recorded source
values. fredgraph.csv needs no API key. Each series is fetched once per process
and cached.

FRED restates history, unlike SEC XBRL where a filed fact is frozen. A claim
that was true against the initial print can differ from the current series: GDP
Q1 2026 was 1.6% on the advance estimate and 2.1% on the third. This returns
today's value, and the tool passes the caveat to the agent.
"""

import csv
import datetime
import io
import re
from typing import Dict, Optional, Tuple

import httpx

from ..utils.logging import get_logger

logger = get_logger(__name__)

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# metric -> (FRED series id, transform). "direct": the observation is the
# figure. "yoy": the series is an index/level; the figure is the
# year-over-year percentage change, which is how these are quoted in claims.
FRED_SERIES: Dict[str, Tuple[str, str]] = {
    "unemployment_rate": ("UNRATE", "direct"),
    "federal_funds_rate": ("FEDFUNDS", "direct"),
    "consumer_confidence": ("UMCSENT", "direct"),
    "gdp_growth": ("A191RL1Q225SBEA", "direct"),   # already % change, SAAR
    "cpi_inflation": ("CPIAUCSL", "yoy"),
    "core_pce": ("PCEPILFE", "yoy"),
    "retail_sales_growth": ("RSAFS", "yoy"),
    "wage_growth": ("CES0500000003", "yoy"),
}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"))}

_series_cache: Dict[str, Dict[str, float]] = {}


def _fetch_csv(series_id: str) -> str:
    """Isolated for tests; one network call per series per process."""
    resp = httpx.get(FREDGRAPH_URL.format(series_id=series_id),
                     timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _observations(series_id: str) -> Dict[str, float]:
    if series_id not in _series_cache:
        rows = list(csv.reader(io.StringIO(_fetch_csv(series_id))))
        _series_cache[series_id] = {
            row[0]: float(row[1])
            for row in rows[1:]
            if len(row) == 2 and row[1] not in (".", "")
        }
    return _series_cache[series_id]


def period_to_observation_date(period: Optional[str]) -> Optional[str]:
    """'April 2026' -> 2026-04-01; 'Q4 2023' -> 2023-10-01 (FRED quarter
    observations are dated at the quarter's first day). None when the period
    does not name a month+year or quarter+year."""
    if not period:
        return None
    m = re.match(r"(\w+)\s+(\d{4})$", period.strip())
    if m and m.group(1) in _MONTHS:
        return f"{m.group(2)}-{_MONTHS[m.group(1)]:02d}-01"
    q = re.match(r"Q([1-4])\s+(\d{4})$", period.strip(), re.IGNORECASE)
    if q:
        return f"{q.group(2)}-{(int(q.group(1)) - 1) * 3 + 1:02d}-01"
    return None


def value_for(metric: str, period: Optional[str]) -> Optional[float]:
    """The metric's value for the period, from today's series. None when the
    metric is unmapped, the period unparseable, or the observation (or, for
    yoy, its prior-year base) is absent — never a guess."""
    entry = FRED_SERIES.get(metric)
    if entry is None:
        return None
    series_id, transform = entry
    date = period_to_observation_date(period)
    if date is None:
        return None
    obs = _observations(series_id)
    if date not in obs:
        return None
    if transform == "direct":
        return obs[date]
    prior = datetime.date.fromisoformat(date)
    prior = prior.replace(year=prior.year - 1).isoformat()
    if prior not in obs:
        return None
    return (obs[date] / obs[prior] - 1.0) * 100.0
