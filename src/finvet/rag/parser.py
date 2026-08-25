"""SEC filing HTML parser and section-aware chunker.

Parses 10-K and 10-Q filings from SEC EDGAR into structured sections,
then splits each section into overlapping chunks suitable for RAG.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from typing import Literal

from bs4 import BeautifulSoup, NavigableString
import tiktoken


# 10-K sections are keyed by item number alone: the form uses one continuous
# item sequence, so "Item 1" is unambiguous.
TEN_K_SECTION_MAP = {
    "1": "business",
    "1A": "risk_factors",
    "1B": "unresolved_staff_comments",
    "1C": "cybersecurity",
    "2": "properties",
    "3": "legal_proceedings",
    "4": "mine_safety",
    "5": "market_for_equity",
    "6": "reserved",
    "7": "mda",
    "7A": "market_risk",
    "8": "financial_statements_and_notes",
    "9": "changes_in_disagreements",
    "9A": "controls_and_procedures",
    "10": "directors_and_officers",
    "11": "executive_compensation",
    "12": "security_ownership",
    "13": "related_transactions",
    "14": "accountant_fees",
    "15": "exhibits",
}

# A 10-Q restarts its numbering in each part, so "Item 1" means Financial
# Statements in Part I and Legal Proceedings in Part II. Keying on the item
# number alone applied the 10-K map, which labelled Financial Statements as
# `business`, mapped MD&A to `properties` (dropping it as non-indexable),
# filed Market Risk under `legal_proceedings`, and discarded Part II's real
# Legal Proceedings heading as a duplicate of Item 1.
TEN_Q_SECTION_MAP = {
    ("I", "1"): "financial_statements_and_notes",
    ("I", "2"): "mda",
    ("I", "3"): "market_risk",
    ("I", "4"): "controls_and_procedures",
    ("II", "1"): "legal_proceedings",
    ("II", "1A"): "risk_factors",
}

# Backwards-compatible alias; the 10-K map was the only one that existed.
SECTION_MAP = TEN_K_SECTION_MAP

# "PART I" / "PART II" headings, which reset a 10-Q's item numbering.
_PART_PATTERN = re.compile(r"^PART\s+(I{1,3}|IV)\b", re.IGNORECASE)

# Sections worth indexing for RAG (skip boilerplate)
INDEXABLE_SECTIONS = {
    "business",
    "risk_factors",
    "legal_proceedings",
    "mda",
    "market_risk",
    "financial_statements_and_notes",
    "cybersecurity",
    "controls_and_procedures",
    "executive_compensation",
}

# Tokenizer for chunk sizing
_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class Section:
    """A named section extracted from an SEC filing."""
    item_number: str          # e.g., "1A"
    name: str                 # e.g., "risk_factors"
    title: str                # e.g., "Risk Factors"
    text: str                 # Plain text content
    part: str | None = None   # "I"/"II" for a 10-Q; None for a 10-K


@dataclass
class Chunk:
    """A sized text chunk from a filing section."""
    text: str
    section_name: str         # e.g., "risk_factors"
    section_title: str        # e.g., "Risk Factors"
    chunk_index: int
    token_count: int


def _count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base tokenizer."""
    return len(_tokenizer.encode(text))


# Regex to match Item headings in SEC filings.
# Matches patterns like: "Item 1.", "ITEM 1A.", "Item 7A.", "ITEM 1. BUSINESS"
_ITEM_PATTERN = re.compile(
    r"^(?:ITEM|Item)\s+(\d+[A-Ca-c]?)\.?\s+(.*)",
    re.IGNORECASE,
)

# Fallback for filers (Workiva/iXBRL, e.g. AMZN) that put the item number and
# its title in separate table cells: once concatenated there is no space after
# the period, so _ITEM_PATTERN's "\s+" never matches. Only used when the strict
# pattern finds nothing, because this one also matches table-of-contents rows
# ("Item 1.Business4") that would otherwise win the first-occurrence dedup.
_ITEM_PATTERN_NO_SPACE = re.compile(
    r"^(?:ITEM|Item)\s+(\d+[A-Ca-c]?)\.\s*(\D.*)",
    re.IGNORECASE,
)


def parse_filing_html(
    filepath: str | Path,
    *,
    filing_type: Literal["10-K", "10-Q"],
) -> list[Section]:
    """Parse an SEC filing into structured sections.

    Args:
        filepath: Path to the filing HTML.
        filing_type: "10-K" or "10-Q". Required rather than inferred: the two
            forms number their items differently, and guessing from a filename
            inside the parser hides the assumption from the caller that
            already knows. ingest_filing passes the type it parsed from the
            filename.

    Returns:
        Sections in document order, one per indexable item.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    is_10q = str(filing_type).upper().replace("_", "-") == "10-Q"

    # Find span/div/p/b elements holding Item or Part headings. <a> tags are
    # table-of-contents links and are excluded by not being searched.
    candidate_tags = soup.find_all(["span", "div", "p", "b"])

    def _collect(pattern):
        """Headings in document order, each tagged with the part in force."""
        found = []
        current_part = None
        for tag in candidate_tags:
            text = tag.get_text(strip=True)
            if len(text) >= 100:
                continue

            part_match = _PART_PATTERN.match(text)
            if part_match:
                current_part = part_match.group(1).upper()
                continue

            match = pattern.match(text)
            if match:
                found.append((tag, current_part, match.group(1).upper(),
                              match.group(2).strip().rstrip(".")))
        return found

    heading_elements = _collect(_ITEM_PATTERN) or _collect(_ITEM_PATTERN_NO_SPACE)
    if not heading_elements:
        return []

    # Deduplicate on (part, item). A 10-Q repeats item numbers across parts, so
    # keying on the number alone silently discarded the second occurrence --
    # which is where Legal Proceedings lives.
    #
    # Which duplicate wins is decided by span, not document order. A table of
    # contents repeats every heading before the body, and taking the first
    # occurrence handed each section the few nodes between two adjacent
    # contents rows: the body text then attached to whichever contents entry
    # happened to be last. Choosing the candidate with the most content
    # between it and the next candidate picks the body heading, because that
    # is what a contents row lacks.
    # Substantive text, not node count: a contents row and a body heading are
    # separated by a similar number of nodes, but only one of them is followed
    # by prose. One pass builds a prefix sum of text length in document order,
    # after which each span costs a subtraction.
    order = list(soup.descendants)
    position = {id(node): i for i, node in enumerate(order)}
    text_before = [0] * (len(order) + 1)
    for i, node in enumerate(order):
        length = len(node.strip()) if isinstance(node, NavigableString) else 0
        text_before[i + 1] = text_before[i] + length
    span_end = len(order)

    def _span(index: int) -> int:
        """Characters of text between this heading and the next candidate."""
        start = position.get(id(heading_elements[index][0]), 0)
        if index + 1 < len(heading_elements):
            stop = position.get(id(heading_elements[index + 1][0]), span_end)
        else:
            stop = span_end
        return max(text_before[stop] - text_before[start], 0)

    best: dict = {}
    for index, (tag, part, item_num, title) in enumerate(heading_elements):
        key = (part, item_num) if is_10q else item_num
        span = _span(index)
        if key not in best or span > best[key][0]:
            best[key] = (span, index, tag, part, item_num, title)

    unique_headings = [
        (tag, part, item_num, title)
        for _, _, tag, part, item_num, title in sorted(best.values(),
                                                       key=lambda b: b[1])
    ]

    sections = []
    for i, (tag, part, item_num, title) in enumerate(unique_headings):
        if is_10q:
            section_name = TEN_Q_SECTION_MAP.get(
                (part, item_num), f"item_{item_num.lower()}")
        else:
            section_name = TEN_K_SECTION_MAP.get(
                item_num, f"item_{item_num.lower()}")

        if section_name not in INDEXABLE_SECTIONS:
            continue

        texts = []
        next_tag = unique_headings[i + 1][0] if i + 1 < len(unique_headings) else None
        for sibling in _iter_after(tag, soup, next_tag):
            if hasattr(sibling, "get_text"):
                s = sibling.get_text(separator=" ", strip=True)
                if s and len(s) > 2:
                    texts.append(s)

        full_text = "\n".join(texts)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"[ \t]{2,}", " ", full_text)

        if len(full_text) > 100:
            sections.append(Section(
                item_number=item_num,
                name=section_name,
                title=title or section_name.replace("_", " ").title(),
                text=full_text,
                part=part,
            ))

    return sections


def _iter_after(start_tag, soup, stop_tag=None):
    """Iterate over all elements between start_tag and stop_tag in document order.

    This walks the DOM tree to collect content between two Item headings.
    The stop check runs against every descendant, not only the leaf elements
    that get yielded — a heading tag is usually a container, so testing it
    against the yielded leaves alone would never match and every section
    would run to the end of the document.
    """
    found_start = False
    for element in soup.descendants:
        if element is start_tag:
            found_start = True
            continue
        if stop_tag is not None and element is stop_tag:
            return
        if found_start:
            # Only yield leaf text-bearing elements to avoid duplication
            if hasattr(element, "children"):
                children = list(element.children)
                if not children or (len(children) == 1 and isinstance(children[0], str)):
                    yield element


def _token_windows(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Slice text that exceeds the ceiling into overlapping token windows.

    The sentence splitter is the last structural boundary available; below it
    there is none, so a single run of text longer than the ceiling used to
    become one oversized chunk. A 3000-token sentence produced a 3000-token
    chunk against a stated 500 limit, which degrades embedding quality
    silently -- the model truncates and nothing raises.
    """
    ids = _tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    windows = []
    for start in range(0, len(ids), step):
        window = ids[start:start + max_tokens]
        if not window:
            break
        windows.append(_tokenizer.decode(window))
        if start + max_tokens >= len(ids):
            break
    return windows


def _atoms(section_text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a section into units that each fit inside the ceiling.

    Paragraph first, then sentence, then raw tokens. Every atom returned is at
    most max_tokens, so packing them can never exceed it.
    """
    atoms = []
    for para in (p.strip() for p in section_text.split("\n")):
        if not para:
            continue
        if _count_tokens(para) <= max_tokens:
            atoms.append(para)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _count_tokens(sentence) <= max_tokens:
                atoms.append(sentence)
            else:
                atoms.extend(_token_windows(sentence, max_tokens, overlap_tokens))
    return atoms


def chunk_sections(
    sections: list[Section],
    max_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Split sections into overlapping chunks for embedding.

    Each section is split into chunks of approximately max_tokens, with
    overlap_tokens of overlap between consecutive chunks. Splitting is
    done at paragraph boundaries when possible.

    Args:
        sections: List of Section objects from parse_filing_html().
        max_tokens: Target maximum tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            f"overlap_tokens must satisfy 0 <= overlap < max_tokens "
            f"(got overlap={overlap_tokens}, max={max_tokens})")

    all_chunks: list[Chunk] = []

    for section in sections:
        atoms = _atoms(section.text, max_tokens, overlap_tokens)
        if not atoms:
            continue

        buffer: list[str] = []
        chunk_index = 0

        def _emit(parts: list[str], index: int) -> Chunk:
            text = "\n".join(parts)
            return Chunk(
                text=text,
                section_name=section.name,
                section_title=section.title,
                chunk_index=index,
                token_count=_count_tokens(text),
            )

        for atom in atoms:
            atom_tokens = _count_tokens(atom)
            # Measure the joined text, not the sum of its parts: _emit inserts
            # a newline between atoms, and summing atoms alone under-counts by
            # one token per separator -- enough to breach an exact ceiling.
            # Chunking runs at ingest time, so the extra encodes are free.
            candidate_tokens = (_count_tokens("\n".join(buffer + [atom]))
                                if buffer else atom_tokens)
            if buffer and candidate_tokens > max_tokens:
                chunk = _emit(buffer, chunk_index)
                all_chunks.append(chunk)
                chunk_index += 1
                # Overlap is measured in tokens, not whole paragraphs. Keeping
                # only complete trailing parts meant a final part larger than
                # the budget yielded no overlap at all: two 300-token
                # paragraphs shared nothing despite overlap_tokens=100.
                carry = _trailing_tokens(chunk.text, overlap_tokens)
                # Overlap is best effort; the ceiling is a guarantee. An atom
                # may itself be max_tokens long (a token-sliced window is
                # exactly that), and carrying context in front of it would
                # push the chunk over. When they cannot both fit, the overlap
                # yields.
                # Measure the join, not the sum: the separator costs a token,
                # and summing here would repeat the same off-by-one the
                # candidate check above exists to avoid.
                if carry and _count_tokens(carry + "\n" + atom) <= max_tokens:
                    buffer = [carry]
                else:
                    buffer = []

            buffer.append(atom)

        if buffer:
            chunk = _emit(buffer, chunk_index)
            if chunk.token_count > 20:  # skip tiny trailing fragments
                all_chunks.append(chunk)

    return all_chunks


def _trailing_tokens(text: str, count: int) -> str:
    """The last `count` tokens of text, decoded."""
    if count <= 0:
        return ""
    ids = _tokenizer.encode(text)
    return _tokenizer.decode(ids[-count:]) if ids else ""


def _get_overlap(parts: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Get the trailing portion of parts that fits within overlap_tokens.

    Returns the overlap parts and their total token count.
    """
    if not parts or overlap_tokens <= 0:
        return [], 0

    overlap_parts = []
    overlap_count = 0

    for part in reversed(parts):
        t = _count_tokens(part)
        if overlap_count + t > overlap_tokens:
            break
        overlap_parts.insert(0, part)
        overlap_count += t

    return overlap_parts, overlap_count
