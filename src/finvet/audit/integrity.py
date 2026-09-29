"""Builds one canonical envelope for a finished run and checksums it.

The envelope carries the claim, the verdict, the confidence, the final response,
the data sources and the event list. The checksum covers exactly that envelope,
exactly what is stored, and verification recomputes it.

Scope of the guarantee. This is an integrity checksum over a stored snapshot. It
detects a value that changed without its checksum being recomputed: a partial
write, a manual edit of one column, a migration that rewrote a field. It does
not resist a privileged writer who updates the data and the checksum together,
because both live in the same database. That needs a key held outside the
database or an externally anchored hash chain, and neither is implemented here.
Do not describe this as tamper detection.
"""

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional

CHECKSUM_ALGORITHM = "sha256"

# The envelope layout the checksum is computed over. Stored with the row so a
# future reader knows which layout a historical digest was taken against;
# changing the fields below requires bumping this.
#
# 2: adds `llm_config` -- which models produced the run. A result that does not
#    say what produced it cannot be compared against a result from another
#    model, which is the comparison this system's central claim rests on.
ENVELOPE_SCHEMA_VERSION = 2

# What the checksum covers, reported by the API alongside the result so the
# claim is legible rather than implied.
CHECKSUM_SCOPE = "audit_execution.full_trace"


# The fields of an event that both copies carry. `created_at` is deliberately
# absent: it is storage metadata that get_events() returns and the envelope
# never held, so comparing it would manufacture a mismatch on correct data.
EVENT_PROJECTION_KEYS = (
    "event_id", "request_id", "parent_event_id", "event_type", "timestamp",
    "agent", "data",
)


def normalize_checksum_value(value: Any) -> Any:
    """The canonical form of a value for hashing, or a refusal.

    `default=str` used to stand here. It made the digest total by stringifying
    anything unrecognised, which is the opposite of what a checksum needs: two
    distinct objects could share a repr and hash identically, and an object
    whose repr is stable could change without changing the digest. Refusing is
    the honest response to a value this function cannot represent faithfully.

    NaN and the infinities are rejected for a related reason: JSON has no
    literal for them, `NaN != NaN` makes any later comparison meaningless, and
    a confidence that is not a number is a defect worth surfacing at the point
    it would be recorded.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(
                f"Non-finite value cannot be checksummed: {value!r}")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(
                f"Non-finite value cannot be checksummed: {value!r}")
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): normalize_checksum_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_checksum_value(item) for item in value]
    raise TypeError(f"Unsupported checksum value: {type(value).__name__}")


def project_events(events: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The comparable shape of an event list, from either side."""
    return [{key: event.get(key) for key in EVENT_PROJECTION_KEYS}
            for event in events or []]


def canonical_event_order(events: Optional[List[Dict[str, Any]]],
                          ) -> List[Dict[str, Any]]:
    """One total order over events, used by the envelope and by the query.

    Timestamp alone is not a total order -- two events can share one, and the
    two sides would then be free to disagree about their order while both are
    correct. event_id breaks the tie the same way `get_events` does.
    """
    return sorted(events or [],
                  key=lambda e: (e.get("timestamp") or "", e.get("event_id") or ""))


def build_execution_envelope(
    *,
    request_id: str,
    claim_text: str,
    terminal_status: Optional[str],
    verdict: Optional[str],
    confidence: Optional[float],
    agents_run: Optional[List[str]],
    events: Optional[List[Dict[str, Any]]],
    final_response: Optional[Dict[str, Any]],
    data_sources: Optional[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything about a finished run that the checksum protects.

    Every field the execution row persists appears here. A field stored beside
    the checksum but absent from it is a field the checksum does not cover,
    which is exactly the defect this replaces.

    `llm_config` names the models behind each role. It is not called
    `model_config`: that is Pydantic v2's reserved configuration attribute,
    already used by `config/settings.py`, `models/claim.py` and
    `api/models.py`, and a data field of that name would shadow it in any
    Pydantic model that later mirrors these rows.
    """
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "request_id": request_id,
        "claim": claim_text,
        "terminal_status": terminal_status,
        "verdict": verdict,
        "confidence": confidence,
        "agents_run": agents_run or [],
        # Stored in the same total order the event query returns, so the two
        # copies stay comparable even when timestamps collide.
        "events": canonical_event_order(events),
        "final_response": final_response,
        "data_sources": data_sources,
        "llm_config": llm_config,
    }


def compute_execution_checksum(envelope: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON encoding of the envelope.

    sort_keys makes the digest independent of dict ordering, so a round trip
    through JSONB that returns keys in a different order still verifies.
    Compact separators and ensure_ascii=False fix the encoding, since any
    variation there would change the digest without changing the data.
    """
    encoded = json.dumps(
        normalize_checksum_value(envelope),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_execution_projection(execution: Mapping[str, Any],
                                 events: Optional[List[Dict[str, Any]]],
                                 ) -> List[str]:
    """Field names where the stored row disagrees with the hashed envelope.

    The execution row keeps claim, verdict, confidence, agents_run and
    data_sources denormalized beside `full_trace`, and the events live in their
    own table. A checksum over the envelope alone leaves all of that free to
    drift while the API still reports a verified digest, so the displayed
    record and the hashed one are compared field by field.
    """
    envelope = execution.get("full_trace") or {}
    checks = {
        "request_id": execution.get("request_id"),
        "claim": execution.get("claim_text"),
        "verdict": execution.get("verdict"),
        "confidence": execution.get("confidence"),
        "agents_run": execution.get("agents_run") or [],
        "data_sources": execution.get("data_sources"),
        # Schema 1 envelopes have no such key; both sides read None and agree,
        # so legacy rows report no drift rather than a manufactured mismatch.
        "llm_config": execution.get("llm_config"),
    }
    mismatches = [
        key for key, value in checks.items()
        if normalize_checksum_value(value)
        != normalize_checksum_value(envelope.get(key))
    ]
    if (project_events(canonical_event_order(events))
            != project_events(envelope.get("events"))):
        mismatches.append("events")
    return mismatches


def verify_execution_checksum(envelope: Mapping[str, Any],
                              stored_checksum: Optional[str]) -> bool:
    """Recompute and compare.

    An absent checksum is not a pass. Rows written before this scheme have no
    verifiable digest, and reporting them as verified is the failure being
    corrected -- the UI used to do exactly that whenever the string was
    non-empty.
    """
    if not stored_checksum:
        return False
    return compute_execution_checksum(envelope) == stored_checksum
