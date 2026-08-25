"""Macro-indicator tool for the news agent, backed by FRED.

Before this, the 13 macro metrics in the news whitelist were promises FinVet
could not keep: macro claims went to free-text news search and mostly died as
NOT_ENOUGH_INFO or false rejects. Eight of them now read the primary source.
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..mcp.fred import FRED_SERIES, period_to_observation_date, value_for
from ..utils.logging import get_logger

logger = get_logger(__name__)

_REVISION_NOTE = (
    "FRED restates history: CPI, retail sales and GDP are revised after first "
    "release, so this is today's series value, which can differ slightly from "
    "the figure as first reported."
)


class MacroIndicatorResult(BaseModel):
    """Result from get_macro_indicator."""
    success: bool = Field(..., description="Whether the lookup succeeded")
    metric: str = Field(..., description="Canonical metric name")
    value: Optional[float] = Field(None, description="Value for the period")
    series_id: Optional[str] = Field(None, description="FRED series ID")
    observation_date: Optional[str] = Field(None, description="Observation date (YYYY-MM-DD)")
    transform: Optional[str] = Field(None, description="'direct' or 'yoy' (year-over-year %)")
    units: Optional[str] = Field(None, description="Percent for rates/growth; index for confidence")
    note: Optional[str] = Field(None, description="Data caveats")
    error: Optional[str] = Field(None, description="Error message if failed")


@tool
def get_macro_indicator(metric: str, period: str) -> Dict[str, Any]:
    """Look up a US macroeconomic indicator from FRED (the primary source).

    USE THIS — not news search — for claims about:
    unemployment_rate, cpi_inflation, core_pce, federal_funds_rate,
    gdp_growth, wage_growth, retail_sales_growth, consumer_confidence.

    Args:
        metric: Canonical metric name from the list above.
        period: The claim's period — a month and year ("April 2026") or a
                quarter ("Q1 2026" — gdp_growth is quarterly).

    Returns:
        MacroIndicatorResult with the value from today's FRED series, the
        series ID and observation date for provenance, and a revision note.
    """
    entry = FRED_SERIES.get(metric)
    if entry is None:
        return MacroIndicatorResult(
            success=False, metric=metric,
            error=f"'{metric}' has no FRED series mapping; verifiable macro "
                  f"metrics are: {', '.join(sorted(FRED_SERIES))}").model_dump()

    date = period_to_observation_date(period)
    if date is None:
        return MacroIndicatorResult(
            success=False, metric=metric,
            error="period must name a month and year (e.g. 'April 2026') or "
                  "a quarter (e.g. 'Q1 2026')").model_dump()

    try:
        value = value_for(metric, period)
    except Exception as e:
        logger.error(f"get_macro_indicator failed for {metric} {period}: {e}")
        return MacroIndicatorResult(success=False, metric=metric, error=str(e)).model_dump()

    series_id, transform = entry
    if value is None:
        return MacroIndicatorResult(
            success=False, metric=metric, series_id=series_id,
            error=f"no observation for {date} in {series_id} (or no "
                  f"prior-year base for the year-over-year calculation)").model_dump()

    return MacroIndicatorResult(
        success=True, metric=metric, value=round(value, 2),
        series_id=series_id, observation_date=date, transform=transform,
        units="index" if metric == "consumer_confidence" else "percent",
        note=_REVISION_NOTE,
    ).model_dump()
