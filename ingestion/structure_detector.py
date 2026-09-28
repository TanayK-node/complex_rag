"""
Stage 2 of ingestion: decide which lines are headings (and at what depth) vs.
ordinary paragraph/clause text, and pull out cross-references like
"Section 4.2" so downstream stages can wire up the cross-reference graph.

Heuristic, not ML-based, by design: legal/enterprise documents are extremely
regular in their numbering conventions, and a transparent rule-based detector
is both cheaper and easier to debug/explain than a learned layout model. This
module is the seam where a learned layout model (e.g. LayoutLM) could later
be swapped in without touching the rest of the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ingestion.pdf_parser import ParsedDocument, ParsedLine

# Matches "1.", "1.1", "4.2.1", optionally followed by a title on the same line.
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s*[-–—]?\s*(.*)$")

# Matches references like "Section 4.2", "Section 6.2(a)", "Sections 3, 5, 7, and 9"
CROSS_REF_RE = re.compile(r"[Ss]ections?\s+(\d+(?:\.\d+){0,3}(?:\([a-z]\))?(?:\s*,\s*\d+(?:\.\d+){0,3})*(?:\s*,?\s*(?:and|&)\s*\d+(?:\.\d+){0,3})?)")
NUMBER_TOKEN_RE = re.compile(r"\d+(?:\.\d+){0,3}")

# Matches a standalone "Article 4" / "Article 17" style heading line (EU/legislative
# convention). Added after testing against a real GDPR excerpt, where none of the
# existing patterns (decimal numbering, ALL-CAPS) recognize this convention at all —
# a third real-world numbering style, distinct from both the decimal-numbered UK
# contract and the "Item 1A" 10-K.
ARTICLE_HEADING_RE = re.compile(r"^Article\s+(\d+)\s*\.?\s*$")


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE_ANCHOR = "table_anchor"


@dataclass
class StructuredBlock:
    block_type: BlockType
    text: str
    page: int
    y0: float = 0.0                 # vertical position on the page, for reading-order interleaving
    depth: int = 0                 # 1 = "1", 2 = "1.1", 3 = "1.1.1", 0 = not numbered
    section_number: str | None = None
    title: str | None = None       # only for headings
    referenced_sections: list[str] | None = None
    table_id: str | None = None    # only for TABLE_ANCHOR blocks


def _extract_cross_references(text: str) -> list[str]:
    refs: set[str] = set()
    for match in CROSS_REF_RE.finditer(text):
        for tok in NUMBER_TOKEN_RE.findall(match.group(1)):
            refs.add(tok)
    return sorted(refs)


def _looks_like_heading(line: ParsedLine, body_font_size: float) -> tuple[bool, str | None, str | None, int | None]:
    """Returns (is_heading, section_number_or_None, title_text, depth_override).
    depth_override is None for the normal decimal-numbering case (depth is
    derived from how many dots are in the section number); it's set explicitly
    for heading conventions where that derivation doesn't apply — currently
    just "Article N", which should always nest one level under whatever
    Chapter is open, not be computed from its own (dot-free) number."""
    text = line.text.strip()
    if not text:
        return False, None, None, None

    m = SECTION_NUMBER_RE.match(text)
    if m and m.group(1):
        number, title = m.group(1), m.group(2).strip()

        # Guards added after testing against a real 117-page 10-K annual report,
        # where bold financial-table column headers ("2025 2024"), pull-quote
        # figures ("$3.75B"), and headcounts ("9,400") were all being matched by
        # the digit-leading regex and misclassified as numbered section headings
        # (they're bold/larger font, same signal a real heading uses). None of
        # that is possible in a document built for testing on clean legal-style
        # numbering, which is why this only showed up on real financial filings.
        if not re.search(r"[a-zA-Z]", title):
            # a numbered "heading" whose title has no letters at all is a bare
            # number/figure (a year, a dollar amount, a table column header),
            # not a real section title.
            return False, None, None, None
        if number.isdigit() and not title and len(number) >= 4:
            return False, None, None, None
        if len(text.split()) > 20:
            # a real numbered heading is short; a bold multi-sentence paragraph
            # that happens to start with a digit is not a heading.
            return False, None, None, None

        # Numbered heading: trust font size OR boldness OR a bigger-than-body size
        # to avoid misclassifying inline references like "as set out in 4.2" as headings
        # (those never start the line with the number).
        if line.font_size > body_font_size + 0.4 or line.is_bold:
            return True, number, title or None, None

    # Un-numbered top-level heading (e.g. document title, "DEFINITIONS" alone).
    # Guard against pull-quote figures on report cover pages ("$3.75B", "9,400")
    # that are bold/large-font and technically "uppercase" (no lowercase letters
    # to contradict) but are numbers/currency, not section titles.
    if (text.isupper() and len(text.split()) <= 8 and line.font_size > body_font_size + 0.4
            and not re.search(r"\d", text) and not text.startswith(("$", "€", "£"))):
        return True, None, text, None

    # "Article N" standalone heading (EU/legislative convention) — depth 2 so it
    # nests under whatever Chapter (depth 1, caught by the ALL-CAPS branch above)
    # is currently open.
    am = ARTICLE_HEADING_RE.match(text)
    if am and (line.font_size > body_font_size + 0.2 or line.is_bold):
        return True, am.group(1), None, 2

    return False, None, None, None


def detect_structure(parsed: ParsedDocument) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []

    for line in parsed.lines:
        if not line.text.strip():
            continue

        is_heading, section_number, title, depth_override = _looks_like_heading(line, parsed.body_font_size)

        if is_heading:
            depth = depth_override if depth_override is not None else (
                section_number.count(".") + 1 if section_number else 1)
            blocks.append(StructuredBlock(
                block_type=BlockType.HEADING,
                text=line.text,
                page=line.page,
                y0=line.y0,
                depth=depth,
                section_number=section_number,
                title=title,
            ))
        else:
            blocks.append(StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text=line.text,
                page=line.page,
                y0=line.y0,
                referenced_sections=_extract_cross_references(line.text),
            ))

    # Interleave table anchors into the same reading-order stream (by page, then
    # vertical position) so tree_builder attaches each table to whichever section
    # is actually open at that point in the document, instead of guessing from
    # page ranges after the fact.
    for table in parsed.tables:
        anchor = StructuredBlock(
            block_type=BlockType.TABLE_ANCHOR,
            text=f"[table:{table.table_id}]",
            page=table.page,
            y0=table.y0,
            table_id=table.table_id,
        )
        insert_at = len(blocks)
        for idx, b in enumerate(blocks):
            if (b.page, b.y0) > (table.page, table.y0):
                insert_at = idx
                break
        blocks.insert(insert_at, anchor)

    return blocks
