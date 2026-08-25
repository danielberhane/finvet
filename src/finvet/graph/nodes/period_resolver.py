"""Period resolver node for converting ambiguous time periods to canonical date ranges."""

import re
from datetime import datetime
from typing import Dict, Tuple
from ...models.state import VerificationState
from ...models.claim import CanonicalPeriod
from ...audit import get_audit_logger
from ...utils.logging import get_logger

logger = get_logger(__name__)

# Words that signal a period is relative to an external event rather than a
# fixed calendar span.  Requires BOTH a relative preposition AND an event noun
# to avoid false-positives like "after-hours" or "since Q3".
_RELATIVE_PREPOSITIONS = re.compile(
    r'\b(after|following|amid|in\s+the\s+wake\s+of|on\s+the\s+news\s+of|since\s+the)\b',
    re.IGNORECASE,
)
_EVENT_NOUNS = {
    'announcement', 'approval', 'acquisition', 'merger', 'rally', 'selloff',
    'sell-off', 'crash', 'surge', 'crisis', 'ruling', 'report', 'news',
    'earnings', 'incident', 'event', 'decision', 'outcome', 'vote',
}


def _is_event_relative(period_str: str) -> bool:
    """Return True when a period string references an external event."""
    if not _RELATIVE_PREPOSITIONS.search(period_str):
        return False
    tokens = set(period_str.lower().split())
    return bool(tokens & _EVENT_NOUNS)


def period_resolver(state: VerificationState) -> Dict:
    """
    Resolve ambiguous time period references to canonical date ranges.

    Handles all claim types:
    - SEC claims: full resolution with audit event
    - Market/News claims: same resolution logic; event-relative periods are
      flagged as "event_relative" so the agent knows the date is a placeholder

    Returns dictionary with:
    - canonical_period: Resolved period with exact dates
    Assumptions are carried on canonical_period.assumptions rather than as
    a parallel state field, so they travel with what they describe.
    - audit_events: Period resolution audit event (SEC only)
    """
    parsed_claim = state["parsed_claim"]
    request_id = state.get("request_id", "unknown")

    if not parsed_claim.period:
        logger.info(f"No period specified, using current (request: {request_id})")
        return _create_current_period(state)

    try:
        period_str = parsed_claim.period
        assumptions = []

        # Both notations put the digit on either side of the letter: "H1 2024"
        # and "1H 2024" are the same half.
        half_match = re.search(
            r'\b(?:H([12])|([12])H)\s*(\d{4})\b', period_str, re.IGNORECASE
        )
        qtr_trailing = re.search(
            r'\b([1-4])\s*Q\s*(?:FY)?\s*(\d{4})\b', period_str, re.IGNORECASE
        )
        qtr_leading = re.search(r'Q?(\d)\s*(?:FY)?(\d{4})', period_str, re.IGNORECASE)
        qtr_year_first = re.search(r'(\d{4})\s*Q(\d)', period_str, re.IGNORECASE)

        # --- Explicit ISO date (must precede every arm below: "2024-03-31"
        # contains "2024" and would otherwise be widened to the calendar year).
        # fullmatch, not match — start_date is an unvalidated str, so a partial
        # match would store the whole string in a date field.
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', period_str.strip()):
            canonical_period = CanonicalPeriod(
                period_type="date",
                start_date=period_str.strip(),
                end_date=period_str.strip(),
                fiscal_year=None,
                fiscal_quarter=None,
                is_assumption=False,
                assumptions=[],
                original_mention=period_str,
            )

        # --- H1 / H2 half-year (must precede annual: "H1 2024" contains "2024") ---
        elif half_match:
            half = int(half_match.group(1) or half_match.group(2))
            year = int(half_match.group(3))
            if half == 1:
                start_date, end_date = f"{year}-01-01", f"{year}-06-30"
            else:
                start_date, end_date = f"{year}-07-01", f"{year}-12-31"
            label = f"H{half}"
            assumptions.append(
                f"Interpreted '{period_str}' as {label} {year} (calendar half-year)"
            )
            canonical_period = CanonicalPeriod(
                period_type="half_year",
                start_date=start_date,
                end_date=end_date,
                fiscal_year=year,
                fiscal_quarter=None,
                is_assumption=True,
                assumptions=assumptions,
                original_mention=period_str,
            )

        # --- Quarter format: Q4 2024 / Q4 FY2024 / 4Q 2024 / 2024Q4 ---
        # Tried most specific first. The trailing-Q form ("4Q 2024") needs its
        # own pattern: the leading-Q one requires the digit after the Q, so this
        # notation used to fall through to the annual arm.
        elif qtr_trailing or qtr_leading or qtr_year_first:
            if qtr_trailing:
                quarter = int(qtr_trailing.group(1))
                year = int(qtr_trailing.group(2))
            elif qtr_leading:
                quarter = int(qtr_leading.group(1))
                year = int(qtr_leading.group(2))
            else:
                year = int(qtr_year_first.group(1))
                quarter = int(qtr_year_first.group(2))

            start_date, end_date = _get_calendar_quarter_dates(quarter, year)
            assumptions.append(f"Interpreted '{period_str}' as calendar Q{quarter} {year}")
            canonical_period = CanonicalPeriod(
                period_type="quarterly",
                start_date=start_date,
                end_date=end_date,
                fiscal_year=year,
                fiscal_quarter=f"Q{quarter}",
                is_assumption=True,
                assumptions=assumptions,
                original_mention=period_str,
            )

        # --- Annual format: FY2024 / 2024 / fiscal 2024 ---
        elif re.search(r'(?:FY|fiscal\s*)?(\d{4})', period_str, re.IGNORECASE):
            year_match = re.search(r'(\d{4})', period_str)
            year = int(year_match.group(1))
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            assumptions.append(f"Interpreted '{period_str}' as calendar year {year}")
            canonical_period = CanonicalPeriod(
                period_type="annual",
                start_date=start_date,
                end_date=end_date,
                fiscal_year=year,
                fiscal_quarter=None,
                is_assumption=True,
                assumptions=assumptions,
                original_mention=period_str,
            )

        # --- Event-relative period (market/news): cannot resolve to dates ---
        elif _is_event_relative(period_str):
            logger.warning(
                f"Event-relative period '{period_str}' cannot be resolved to exact dates "
                f"(request: {request_id})"
            )
            return _create_event_relative_period(state, period_str)

        else:
            logger.warning(f"Could not parse period '{period_str}', using current")
            return _create_current_period(state)

        # Audit event (all resolved claim types)
        # log_event() persists; state["audit_events"] did not. The period the
        # whole SEC retrieval path is scoped to, and the assumptions made to
        # reach it, were being discarded.
        get_audit_logger().log_event(
            event_type="period_resolved",
            request_id=request_id,
            data={
                "original_mention": period_str,
                "canonical_period": canonical_period.model_dump(),
                "assumptions_made": assumptions,
            },
        )

        logger.info(
            f"Period resolved (request: {request_id}): "
            f"{canonical_period.start_date} to {canonical_period.end_date}"
        )

        return {
            "canonical_period": canonical_period,
        }

    except Exception as e:
        logger.error(f"Period resolution failed: {str(e)}, using current period")
        return _create_current_period(state)


def _create_current_period(state: VerificationState) -> Dict:
    """Create a current period when no specific period is given."""
    today = datetime.utcnow().date().isoformat()
    canonical_period = CanonicalPeriod(
        period_type="current",
        start_date=today,
        end_date=today,
        fiscal_year=None,
        fiscal_quarter=None,
        is_assumption=True,
        assumptions=["No specific period provided, using current date"],
        original_mention=state["parsed_claim"].period,
    )
    return {
        "canonical_period": canonical_period,
    }


def _create_event_relative_period(state: VerificationState, period_str: str) -> Dict:
    """Create a placeholder period for event-relative strings that cannot be resolved."""
    today = datetime.utcnow().date().isoformat()
    assumption = (
        f"Period '{period_str}' references an external event and cannot be resolved "
        "to exact calendar dates. Using current date as placeholder — the agent "
        "should note that the verification date may not match the claimed event date."
    )
    canonical_period = CanonicalPeriod(
        period_type="event_relative",
        start_date=today,
        end_date=today,
        fiscal_year=None,
        fiscal_quarter=None,
        is_assumption=True,
        assumptions=[assumption],
        original_mention=period_str,
    )
    return {
        "canonical_period": canonical_period,
    }


def _get_calendar_quarter_dates(quarter: int, year: int) -> Tuple[str, str]:
    """Get start and end dates for a calendar quarter."""
    quarter_map = {
        1: ("01-01", "03-31"),
        2: ("04-01", "06-30"),
        3: ("07-01", "09-30"),
        4: ("10-01", "12-31"),
    }
    if quarter < 1 or quarter > 4:
        raise ValueError(f"Invalid quarter: {quarter}")
    start_md, end_md = quarter_map[quarter]
    return f"{year}-{start_md}", f"{year}-{end_md}"
