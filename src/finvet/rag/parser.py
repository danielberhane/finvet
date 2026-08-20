"""SEC filing HTML parser and section-aware chunker.

Parses 10-K and 10-Q filings from SEC EDGAR into structured sections,
then splits each section into overlapping chunks suitable for RAG.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
import tiktoken


# Section mapping: Item number → semantic name
SECTION_MAP = {
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


def parse_filing_html(filepath: str | Path) -> list[Section]:
    """Parse an SEC filing HTML file into structured sections.

    Finds Item headings (Item 1, Item 1A, Item 7, etc.) in <span> tags
    and extracts the text content between them.

    Args:
        filepath: Path to the SEC filing HTML file.

    Returns:
        List of Section objects, one per identified Item.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Find all span/div elements containing Item headings.
    # SEC filings use <span> tags for section headings in the body.
    # The <a> tags are table-of-contents links — skip those.
    heading_elements = []

    for tag in soup.find_all(["span", "div", "p", "b"]):
        text = tag.get_text(strip=True)
        match = _ITEM_PATTERN.match(text)
        if match and len(text) < 100:
            item_num = match.group(1).upper()
            title = match.group(2).strip().rstrip(".")
            heading_elements.append((tag, item_num, title))

    if not heading_elements:
        return []

    # Deduplicate: keep only the first occurrence of each Item number.
    # SEC filings often have a table of contents that duplicates headings.
    seen = set()
    unique_headings = []
    for tag, item_num, title in heading_elements:
        if item_num not in seen:
            seen.add(item_num)
            unique_headings.append((tag, item_num, title))

    # Extract text between consecutive headings.
    sections = []
    for i, (tag, item_num, title) in enumerate(unique_headings):
        section_name = SECTION_MAP.get(item_num, f"item_{item_num.lower()}")

        # Skip sections not worth indexing
        if section_name not in INDEXABLE_SECTIONS:
            continue

        # Collect all text from this heading to the next heading
        texts = []

        # Find the next heading's tag (or end of document)
        next_tag = unique_headings[i + 1][0] if i + 1 < len(unique_headings) else None

        # Walk siblings and descendants after the heading tag
        for sibling in _iter_after(tag, soup):
            if next_tag and sibling is next_tag:
                break
            if hasattr(sibling, "get_text"):
                t = sibling.get_text(separator=" ", strip=True)
                if t and len(t) > 2:
                    texts.append(t)

        full_text = "\n".join(texts)
        # Clean up excessive whitespace
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"[ \t]{2,}", " ", full_text)

        if len(full_text) > 100:  # Skip near-empty sections
            sections.append(Section(
                item_number=item_num,
                name=section_name,
                title=title or section_name.replace("_", " ").title(),
                text=full_text,
            ))

    return sections


def _iter_after(start_tag, soup):
    """Iterate over all elements after start_tag in document order.

    This walks the DOM tree to collect content between two Item headings.
    Uses next_elements to traverse in document order.
    """
    found_start = False
    for element in soup.descendants:
        if element is start_tag:
            found_start = True
            continue
        if found_start:
            # Only yield leaf text-bearing elements to avoid duplication
            if hasattr(element, "children"):
                children = list(element.children)
                if not children or (len(children) == 1 and isinstance(children[0], str)):
                    yield element


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
    all_chunks = []

    for section in sections:
        # Split section text into paragraphs
        paragraphs = [p.strip() for p in section.text.split("\n") if p.strip()]

        # Build chunks by accumulating paragraphs
        current_parts: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            # If a single paragraph exceeds max_tokens, split it by sentences
            if para_tokens > max_tokens:
                # Flush current buffer first
                if current_parts:
                    chunk_text = "\n".join(current_parts)
                    all_chunks.append(Chunk(
                        text=chunk_text,
                        section_name=section.name,
                        section_title=section.title,
                        chunk_index=chunk_index,
                        token_count=current_tokens,
                    ))
                    chunk_index += 1
                    # Keep overlap from the end of current buffer
                    current_parts, current_tokens = _get_overlap(
                        current_parts, overlap_tokens
                    )

                # Split the long paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    s_tokens = _count_tokens(sentence)
                    if current_tokens + s_tokens > max_tokens and current_parts:
                        chunk_text = " ".join(current_parts)
                        all_chunks.append(Chunk(
                            text=chunk_text,
                            section_name=section.name,
                            section_title=section.title,
                            chunk_index=chunk_index,
                            token_count=current_tokens,
                        ))
                        chunk_index += 1
                        current_parts, current_tokens = _get_overlap(
                            current_parts, overlap_tokens
                        )
                    current_parts.append(sentence)
                    current_tokens += s_tokens
                continue

            # Normal case: accumulate paragraphs
            if current_tokens + para_tokens > max_tokens and current_parts:
                chunk_text = "\n".join(current_parts)
                all_chunks.append(Chunk(
                    text=chunk_text,
                    section_name=section.name,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                ))
                chunk_index += 1
                # Keep overlap
                current_parts, current_tokens = _get_overlap(
                    current_parts, overlap_tokens
                )

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining content
        if current_parts:
            chunk_text = "\n".join(current_parts)
            tokens = _count_tokens(chunk_text)
            if tokens > 20:  # Skip tiny trailing chunks
                all_chunks.append(Chunk(
                    text=chunk_text,
                    section_name=section.name,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    token_count=tokens,
                ))

    return all_chunks


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
