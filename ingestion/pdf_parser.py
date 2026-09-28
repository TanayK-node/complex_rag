"""
Stage 1 of ingestion: turn raw PDF bytes into layout-aware lines + tables.

We deliberately do NOT just call `page.extract_text()` and throw away layout,
because heading detection downstream needs font size / boldness, and table
detection needs pdfplumber's table geometry. This stage is the only place
that touches the PDF library directly, so swapping PyMuPDF/pdfplumber for
another parser later only touches this file.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import pdfplumber

from schemas.models import BoundingSpan, TableBlock


@dataclass
class ParsedLine:
    text: str
    page: int
    font_size: float
    is_bold: bool
    y0: float


@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    total_pages: int
    lines: list[ParsedLine] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    body_font_size: float = 10.0  # the most common font size -> treated as "normal paragraph text"


# Repeated running headers/footers (e.g. "CONFIDENTIAL", "Page 3") should not be
# treated as content. Detected as lines that repeat on >=40% of pages AND sit at
# consistent, narrow-band vertical positions across those occurrences — a real
# header/footer is glued to the same spot on every page. Checking recurrence alone
# is not enough: tested against a real GDPR excerpt where every "Article N" heading
# normalizes to the same masked key ("Article #") and recurs on nearly every page,
# but at very different y-positions each time (it's a heading, not a footer) — the
# position-variance check is what tells the two apart.
def _strip_headers_footers(raw_lines: list[ParsedLine], total_pages: int) -> list[ParsedLine]:
    if total_pages < 3:
        return raw_lines
    seen: dict[str, set[int]] = {}
    positions: dict[str, list[float]] = {}
    for ln in raw_lines:
        key = re.sub(r"\d+", "#", ln.text.strip())
        seen.setdefault(key, set()).add(ln.page)
        positions.setdefault(key, []).append(ln.y0)

    repeated_keys = set()
    for k, pages in seen.items():
        if len(pages) < max(3, int(total_pages * 0.4)):
            continue
        ys = positions[k]
        if max(ys) - min(ys) <= 15:  # narrow band = glued to the same spot every page
            repeated_keys.add(k)

    return [ln for ln in raw_lines if re.sub(r"\d+", "#", ln.text.strip()) not in repeated_keys]


def parse_pdf(file_path: str, document_id: str, document_name: str) -> ParsedDocument:
    raw_lines: list[ParsedLine] = []
    tables: list[TableBlock] = []
    table_counter = 0

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for page_idx, page in enumerate(pdf.pages, start=1):
            # --- tables first, so we can exclude their bounding boxes from text lines ---
            page_tables = page.find_tables()
            table_bboxes = []
            for t in page_tables:
                extracted = t.extract()
                if not extracted or not any(any(cell for cell in row) for row in extracted):
                    continue
                table_counter += 1
                headers = [c.strip() if c else "" for c in extracted[0]]
                rows = [[c.strip() if c else "" for c in row] for row in extracted[1:]]
                tables.append(TableBlock(
                    document_id=document_id,
                    table_id=f"{document_id}-tbl-{table_counter}",
                    page=page_idx,
                    y0=t.bbox[1],
                    headers=headers,
                    rows=rows,
                ))
                table_bboxes.append(t.bbox)

            def in_table(word) -> bool:
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 - 2 <= word["x0"] and word["x1"] <= x1 + 2 and top - 2 <= word["top"] <= bottom + 2:
                        return True
                return False

            words = [w for w in page.extract_words(extra_attrs=["size", "fontname"]) if not in_table(w)]

            # group words into lines by their vertical position
            words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
            current_line: list[dict] = []
            current_top = None
            for w in words:
                top = round(w["top"], 1)
                if current_top is None or abs(top - current_top) <= 2.5:
                    current_line.append(w)
                    current_top = top if current_top is None else current_top
                else:
                    raw_lines.append(_line_from_words(current_line, page_idx))
                    current_line = [w]
                    current_top = top
            if current_line:
                raw_lines.append(_line_from_words(current_line, page_idx))

    lines = _strip_headers_footers(raw_lines, total_pages)
    body_font_size = _estimate_body_font_size(lines)

    return ParsedDocument(
        document_id=document_id,
        document_name=document_name,
        total_pages=total_pages,
        lines=lines,
        tables=tables,
        body_font_size=body_font_size,
    )


def _line_from_words(words: list[dict], page: int) -> ParsedLine:
    text = " ".join(w["text"] for w in words).strip()
    sizes = [w.get("size", 10.0) for w in words]
    fontnames = [w.get("fontname", "") for w in words]
    is_bold = any("bold" in fn.lower() for fn in fontnames)
    return ParsedLine(
        text=text,
        page=page,
        font_size=round(statistics.mean(sizes), 1) if sizes else 10.0,
        is_bold=is_bold,
        y0=words[0]["top"] if words else 0.0,
    )


def _estimate_body_font_size(lines: list[ParsedLine]) -> float:
    sizes = [ln.font_size for ln in lines if ln.text]
    if not sizes:
        return 10.0
    return statistics.mode(sizes)
