"""
Context building (spec section 11). Takes reranked candidates, not the raw
retrieval pool, and turns them into the actual prompt context: grouped by
section (so two chunks from the same clause don't repeat the breadcrumb),
ordered by natural reading order (not rerank score — a reader/LLM should see
Section 3 before Section 4, even if 4 scored higher), and trimmed to a hard
token budget by dropping the lowest-relevance chunks first.

This deliberately does NOT just concatenate top-K chunks (spec explicitly
calls that out as insufficient): grouping + budget-aware trimming + explicit
citations are the three things a naive concatenation gets wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import settings
from context.citation import format_citation
from ingestion.chunker import approx_tokens
from reranking.base import RerankedCandidate
from schemas.models import Chunk


@dataclass
class ContextBlock:
    document_name: str
    section_number: str | None
    section_title: str | None
    parent_section: str | None
    page_start: int
    page_end: int
    text: str          # possibly multiple chunks' text, joined
    citation: str
    chunk_ids: list[str] = field(default_factory=list)
    max_rerank_score: float = 0.0


@dataclass
class ContextBundle:
    blocks: list[ContextBlock]
    context_text: str
    total_tokens: int
    included_chunk_ids: list[str]
    dropped_for_budget: list[str]  # chunk_ids that made reranking but were cut for token budget


def _group_key(chunk: Chunk) -> tuple:
    return (chunk.document_id, chunk.node_id)


def build_context(reranked: list[RerankedCandidate], max_tokens: int | None = None) -> ContextBundle:
    max_tokens = max_tokens or settings.max_context_tokens

    # 1) group ALL reranked chunks that share the same document node first —
    # budget trimming happens per *group* below, since header overhead is
    # per-group, not per-chunk, and trimming at the chunk level would
    # underestimate the true rendered cost (breadcrumb + citation text).
    groups: dict[tuple, list[RerankedCandidate]] = {}
    order: list[tuple] = []
    for rc in reranked:
        key = _group_key(rc.candidate.chunk)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rc)

    group_best_score = {key: max(rc.rerank_score for rc in groups[key]) for key in order}

    def _build_block(key) -> ContextBlock:
        rcs = groups[key]
        first_chunk = rcs[0].candidate.chunk
        seen_text: set[str] = set()
        texts = []
        for rc in rcs:
            t = rc.candidate.chunk.source_text.strip()
            fingerprint = t[:60]
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            texts.append(t)
        return ContextBlock(
            document_name=first_chunk.document_name,
            section_number=first_chunk.section_number,
            section_title=first_chunk.section_title,
            parent_section=first_chunk.parent_section,
            page_start=min(rc.candidate.chunk.page_start for rc in rcs),
            page_end=max(rc.candidate.chunk.page_end for rc in rcs),
            text="\n\n".join(texts),
            citation=format_citation(first_chunk),
            chunk_ids=[rc.candidate.chunk_id for rc in rcs],
            max_rerank_score=max(rc.rerank_score for rc in rcs),
        )

    # 2) natural reading order: by page, then by section number — NOT by rerank
    # score, so the assembled context reads like the document
    def _sort_key(key):
        c = groups[key][0].candidate.chunk
        sec_parts = tuple(int(p) for p in c.section_number.split(".")) if c.section_number else (9999,)
        return (c.page_start, sec_parts)

    order.sort(key=_sort_key)
    active_keys = list(order)
    dropped: list[str] = []

    # 3) iteratively drop the lowest-scoring group (not the last-in-reading-order
    # one) until the actually-rendered context fits the token budget
    while True:
        blocks = [_build_block(k) for k in active_keys]
        context_text = _render(blocks)
        total = approx_tokens(context_text)
        if total <= max_tokens or len(active_keys) <= 1:
            break
        worst_key = min(active_keys, key=lambda k: group_best_score[k])
        active_keys.remove(worst_key)
        dropped.extend(rc.candidate.chunk_id for rc in groups[worst_key])

    return ContextBundle(
        blocks=blocks,
        context_text=context_text,
        total_tokens=total,
        included_chunk_ids=[cid for b in blocks for cid in b.chunk_ids],
        dropped_for_budget=dropped,
    )


def _render(blocks: list[ContextBlock]) -> str:
    parts = []
    for b in blocks:
        header_bits = [f"Document: {b.document_name}"]
        if b.section_number:
            header_bits.append(f"Section {b.section_number}" + (f" ({b.section_title})" if b.section_title else ""))
        elif b.section_title:
            header_bits.append(b.section_title)
        if b.parent_section:
            header_bits.append(f"Parent: {b.parent_section}")
        pages = f"Page {b.page_start}" if b.page_start == b.page_end else f"Pages {b.page_start}-{b.page_end}"
        header_bits.append(pages)
        header = "[" + " | ".join(header_bits) + "]"
        parts.append(f"{header}\n{b.text}")
    return "\n\n---\n\n".join(parts)
