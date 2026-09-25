"""A read-only inventory of audit records nothing else surfaces.

Three ways a run stops being visible:

    orphan events      events under a request id with no execution row, from a
                       run that was never terminalized
    stuck reviews      rows left in REVIEWING because the process holding the
                       claim died. No reconcile call will take them, because
                       reconcile claims only REVIEW_FINALIZATION_FAILED
    awaiting reconcile rows in REVIEW_FINALIZATION_FAILED, where the graph ran
                       but the audit write did not land. Retryable through
                       POST /review/{id}/reconcile

A query for events without executions finds only the first; the other two have
execution rows. Separating them is what this module is for.

This module only reads. There is no repair path and no --apply flag. Building
an execution row from surviving events would mean inventing the parts the
events do not record, and a reconstructed row stored beside genuine ones cannot
be told apart from them.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..config.database import get_db_session
from ..utils.logging import get_logger
from .database import REVIEW_FINALIZATION_FAILED
from .models import AuditEvent, AuditExecution

logger = get_logger(__name__)

# How long a row may sit in REVIEWING before it is reported as stuck. Long
# enough that a reviewer reading a claim carefully is never flagged; short
# enough that an abandoned claim surfaces the same day. A row below this
# threshold is someone's work in progress, not a fault.
STALE_REVIEW_MINUTES = 60

# Bound on the identifiers listed per category. The counts are exact; the
# samples exist so an operator has somewhere to start.
DEFAULT_SAMPLE_LIMIT = 50


def stale_cutoff(minutes: int) -> str:
    """The ISO timestamp before which a REVIEWING row counts as stuck."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


@dataclass(frozen=True)
class OperatorReport:
    """What was found. Counts are exact; the lists are bounded samples."""

    orphan_event_request_ids: List[str] = field(default_factory=list)
    orphan_event_count: int = 0
    stuck_reviews: List[Dict[str, Any]] = field(default_factory=list)
    stuck_review_count: int = 0
    awaiting_reconciliation: List[Dict[str, Any]] = field(default_factory=list)
    awaiting_reconciliation_count: int = 0
    stale_after_minutes: int = STALE_REVIEW_MINUTES
    generated_at: str = ""
    error: Optional[str] = None

    @property
    def is_clean(self) -> bool:
        """Nothing to act on -- and the query actually ran.

        An unreachable database must never read as a clean system.
        """
        return (self.error is None
                and not self.orphan_event_count
                and not self.stuck_review_count
                and not self.awaiting_reconciliation_count)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "stale_after_minutes": self.stale_after_minutes,
            "orphan_event_count": self.orphan_event_count,
            "orphan_event_request_ids": list(self.orphan_event_request_ids),
            "stuck_review_count": self.stuck_review_count,
            "stuck_reviews": list(self.stuck_reviews),
            "awaiting_reconciliation_count": self.awaiting_reconciliation_count,
            "awaiting_reconciliation": list(self.awaiting_reconciliation),
            "is_clean": self.is_clean,
            "error": self.error,
        }


def _row_summary(row) -> Dict[str, Any]:
    return {
        "request_id": row.request_id,
        "timestamp": row.timestamp,
        "claim": row.claim_text,
    }


def build_operator_report(
    *,
    stale_after_minutes: int = STALE_REVIEW_MINUTES,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> OperatorReport:
    """Query the three categories. Reads only; writes nothing."""
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        with get_db_session() as session:
            known = session.query(AuditExecution.request_id)

            orphan_rows = (
                session.query(AuditEvent.request_id)
                .filter(~AuditEvent.request_id.in_(known))
                .limit(sample_limit)
                .all()
            )
            orphan_ids = sorted({row[0] for row in orphan_rows})

            cutoff = stale_cutoff(stale_after_minutes)
            stuck_rows = (
                session.query(AuditExecution)
                .filter(AuditExecution.verdict == "REVIEWING",
                        AuditExecution.timestamp < cutoff)
                .limit(sample_limit)
                .all()
            )

            awaiting_rows = (
                session.query(AuditExecution)
                .filter(AuditExecution.verdict == REVIEW_FINALIZATION_FAILED)
                .limit(sample_limit)
                .all()
            )

            return OperatorReport(
                orphan_event_request_ids=orphan_ids,
                orphan_event_count=len(orphan_ids),
                stuck_reviews=[_row_summary(r) for r in stuck_rows],
                stuck_review_count=len(stuck_rows),
                awaiting_reconciliation=[_row_summary(r) for r in awaiting_rows],
                awaiting_reconciliation_count=len(awaiting_rows),
                stale_after_minutes=stale_after_minutes,
                generated_at=generated_at,
            )
    except Exception as exc:
        logger.error(f"Operator report could not be built: {exc}")
        return OperatorReport(stale_after_minutes=stale_after_minutes,
                              generated_at=generated_at, error=str(exc))
