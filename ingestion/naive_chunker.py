"""
Baseline chunker used ONLY by the evaluation framework (spec sections 21/22):
splits the raw parsed text into fixed-size, overlapping windows with zero
awareness of headings, clauses, or tables. This is the "naive PDF -> chunks
-> embeddings -> FAISS" approach the spec explicitly contrasts the real
system against — it exists so Phase 6 can prove (or disprove) that the
structure-aware pipeline actually helps, rather than asserting it.
"""
from __future__ import annotations

from ingestion.pdf_parser import ParsedDocument
from schemas.models import Chunk, ChunkType


def fixed_size_chunks(parsed: ParsedDocument, document_name: str,
                       chunk_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    # flatten to (word, page) pairs so we can still report a page range,
    # even though section/clause boundaries are completely ignored
    word_pages: list[tuple[str, int]] = []
    for line in parsed.lines:
        for w in line.text.split():
            word_pages.append((w, line.page))

    chunks: list[Chunk] = []
    step = max(1, chunk_words - overlap_words)
    i = 0
    idx = 0
    while i < len(word_pages):
        window = word_pages[i:i + chunk_words]
        if not window:
            break
        text = " ".join(w for w, _ in window)
        pages = [p for _, p in window]
        idx += 1
        chunks.append(Chunk(
            chunk_id=f"{parsed.document_id}-fixed-{idx}",
            document_id=parsed.document_id,
            document_name=document_name,
            chunk_type=ChunkType.PARAGRAPH,
            node_id="flat",              # no tree — this baseline has no hierarchy
            section_number=None,
            section_title=None,
            parent_section=None,
            page_start=min(pages),
            page_end=max(pages),
            source_text=text,
            display_text=text,
            token_count=int(len(window) * 1.3),
        ))
        i += step

    return chunks
