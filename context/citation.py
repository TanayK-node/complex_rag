"""
Spec section 14: every factual claim must be traceable to document/page/section.
One formatting function so context building and (later) generation produce
identical citation strings.
"""
from __future__ import annotations

from schemas.models import Chunk


def format_citation(chunk: Chunk) -> str:
    pages = f"Page {chunk.page_start}" if chunk.page_start == chunk.page_end else f"Pages {chunk.page_start}-{chunk.page_end}"
    if chunk.section_number:
        return f"[Source: {chunk.document_name}, Section {chunk.section_number}, {pages}]"
    return f"[Source: {chunk.document_name}, {pages}]"
