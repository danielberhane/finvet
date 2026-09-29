"""Extracts a penalty amount from filing text, in Python.

fine_amount and settlement_amount have no XBRL concept, so a numeric claim
about either has no structured value to compare and fails closed. This reads
the amount out of the filing instead.

The model is never asked for the number. XBRL is trusted because Python pulls
the value, not because the source is structured, and the same holds here: the
filing is the authoritative record of what a company disclosed, and the
untrusted step is a model reading a figure out of a paragraph.

Extraction runs only where exactly one penalty amount appears in the section
that discloses penalties. A section listing several amounts returns None:

    legal_proceedings  "...fined the Company EUR 500 million in the Article
                        5(4) Investigation..."          -> one amount, extracted
    mda                $44.1B, $43.8B, EUR 14.2B, ...   -> declines

None leaves the existing fail-closed path unchanged, so this can only turn a
decline into a verdict, never one verdict into another.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Where an issuer discloses a fine. The same sentence in MD&A is discussion,
# not the disclosure, and MD&A is dense with unrelated figures.
PENALTY_SECTIONS = frozenset({"legal_proceedings"})

# The amount must be doing penalty work in the sentence. A number that merely
# shares a paragraph with the word "fine" is not a fine.
_PENALTY_LANGUAGE = re.compile(
    r"\b(?:fined|fine\s+of|penalt(?:y|ies)\s+of|civil\s+penalty|"
    r"settle(?:d|ment)\s+(?:of|for)|agreed\s+to\s+pay)\b",
    re.IGNORECASE,
)

# How far an amount may sit from that language and still be its object. Long
# enough for "fined the Company EUR 500 million", short enough that the next
# unrelated figure in the paragraph does not qualify.
PENALTY_PROXIMITY_CHARS = 120

_CURRENCY_BY_SYMBOL = {"€": "EUR", "$": "USD", "£": "GBP"}
_SCALE = {"thousand": 1_000, "million": 1_000_000,
          "billion": 1_000_000_000, "trillion": 1_000_000_000_000}

_AMOUNT = re.compile(
    r"(?P<symbol>[€$£])\s?(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion))?",
    re.IGNORECASE,
)

# The tool wraps stored text before the model sees it; extraction reads the
# same passage either way.
_DELIMITERS = re.compile(r"</?filing_excerpt>")


_CURRENCY_WORDS = {
    "EUR": re.compile(r"€|\beuros?\b|\bEUR\b", re.IGNORECASE),
    "USD": re.compile(r"\$|\bdollars?\b|\bUSD\b", re.IGNORECASE),
    "GBP": re.compile(r"£|\bpounds?\b|\bGBP\b", re.IGNORECASE),
}


def currency_in_text(text: Optional[str]) -> Optional[str]:
    """The one currency a passage names, or None.

    `ParsedClaim` has no currency field, so without this a claim stating
    "500 million euros" was indistinguishable from one naming no currency at
    all, and a EUR filing figure could not be compared against either. Two
    currencies in one passage is ambiguity, not information -- a claim that
    says "EUR 500 million, about $570 million" has not named its unit.
    """
    if not text:
        return None
    found = {code for code, pattern in _CURRENCY_WORDS.items()
             if pattern.search(text)}
    return found.pop() if len(found) == 1 else None


@dataclass(frozen=True)
class FilingAmount:
    """A figure Python lifted from filing prose, and where to find it again.

    `extraction_method` is fixed rather than defaulted: this type exists to be
    distinguishable from an XBRL fact and from a model's reading, and a field
    that could be set to something else would not distinguish it.
    """

    value: float
    currency: str
    evidence_id: Optional[str]
    content_sha256: Optional[str]
    section: str
    matched_text: str
    extraction_method: str = "deterministic"


def _to_value(match: re.Match) -> Optional[float]:
    try:
        number = float(match.group("number").replace(",", ""))
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    scale = (match.group("scale") or "").lower()
    return number * _SCALE.get(scale, 1)


def extract_amount(chunk: Dict[str, Any]) -> Optional[FilingAmount]:
    """One penalty amount from one chunk, or None.

    None whenever extraction would be a guess: the wrong section, no penalty
    language, no amount near it, or more than one candidate. The caller keeps
    its existing fail-closed behaviour in every one of those cases.
    """
    if not isinstance(chunk, dict):
        return None
    if (chunk.get("section") or "") not in PENALTY_SECTIONS:
        return None

    text = _DELIMITERS.sub("", str(chunk.get("chunk_text") or "")).strip()
    if not text:
        return None

    penalties = [m.span() for m in _PENALTY_LANGUAGE.finditer(text)]
    if not penalties:
        return None

    candidates = []
    for match in _AMOUNT.finditer(text):
        value = _to_value(match)
        if value is None:
            continue
        start, end = match.span()
        near = any(start - p_end <= PENALTY_PROXIMITY_CHARS and p_start - end <= 0
                   for p_start, p_end in penalties)
        if near:
            candidates.append((value, match))

    # Exactly one, or nothing. Two penalty figures in one passage is the case
    # where picking either would be a coin toss, and a wrong number here would
    # decide a verdict.
    if len(candidates) != 1:
        return None

    value, match = candidates[0]
    return FilingAmount(
        value=value,
        currency=_CURRENCY_BY_SYMBOL.get(match.group("symbol"), ""),
        evidence_id=chunk.get("evidence_id"),
        content_sha256=chunk.get("content_sha256"),
        section=chunk.get("section") or "",
        matched_text=match.group(0),
    )
