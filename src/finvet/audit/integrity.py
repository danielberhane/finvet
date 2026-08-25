"""One canonical envelope for a finished run, and a checksum over all of it.

The previous scheme hashed the buffered event list alone. The row beside that
hash also stored the claim, the verdict, the confidence, the final response and
the data sources, so any of those could be edited without disturbing the digest
-- and the audit UI printed a green "Verified" badge whenever the hash string
was non-empty, having recomputed nothing.

What this module provides instead: build the envelope, hash exactly that
envelope, store exactly what was hashed, and verify by recomputation.

**Scope of the guarantee.** This is an integrity checksum over a stored
snapshot. It detects a value that changed without its checksum being recomputed
-- a partial write, a manual edit of one column, a migration that rewrote a
field. It does **not** resist a privileged writer who updates the data and the
checksum together, because both live in the same database. Tamper evidence
against that adversary needs a key held outside the database or an externally
anchored hash chain, and neither is implemented here. Do not describe this as
tamper detection.
"""

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

CHECKSUM_ALGORITHM = "sha256"

# The envelope layout the checksum is computed over. Stored with the row so a
# future reader knows which layout a historical digest was taken against;
# changing the fields below requires bumping this.
ENVELOPE_SCHEMA_VERSION = 1

# What the checksum covers, reported by the API alongside the result so the
# claim is legible rather than implied.
CHECKSUM_SCOPE = "audit_execution.full_trace"


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
) -> Dict[str, Any]:
    """Everything about a finished run that the checksum protects.

    Every field the execution row persists appears here. A field stored beside
    the checksum but absent from it is a field the checksum does not cover,
    which is exactly the defect this replaces.
    """
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "request_id": request_id,
        "claim": claim_text,
        "terminal_status": terminal_status,
        "verdict": verdict,
        "confidence": confidence,
        "agents_run": agents_run or [],
        "events": events or [],
        "final_response": final_response,
        "data_sources": data_sources,
    }


def compute_execution_checksum(envelope: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON encoding of the envelope.

    sort_keys makes the digest independent of dict ordering, so a round trip
    through JSONB that returns keys in a different order still verifies.
    Compact separators and ensure_ascii=False fix the encoding, since any
    variation there would change the digest without changing the data.
    """
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
