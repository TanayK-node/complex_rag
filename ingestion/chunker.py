"""
Stage 4 of ingestion: turn each tree node's raw_text into one or more Chunks.

Design choice: we chunk *within* each already-built tree node rather than
chunking the raw document and assigning nodes afterward. That guarantees a
chunk can never straddle a section boundary, and it means every chunk is
produced already knowing its section_number / section_title / parent_section
metadata (spec section 3/5) instead of that being inferred later.

Clause boundaries are detected the same way headings are (a leading
"4.2.1"-style number), even when the clause is not its own tree node, so a
clause is never split across two chunks unless it alone exceeds max_tokens.
"""
from __future__ import annotations

import re

from config.settings import settings
from ingestion.structure_detector import SECTION_NUMBER_RE, CROSS_REF_RE, NUMBER_TOKEN_RE
from schemas.models import Chunk, ChunkType, DocumentNode, DocumentTree, TableBlock

_CLAUSE_START_RE = re.compile(r"^(\d+(?:\.\d+){1,4})\.?\s+")


def approx_tokens(text: str) -> int:
    # Cheap, dependency-free approximation (~0.75 words/token for English legal
    # prose). Good enough for chunk-size budgeting; swap for a real tokenizer
    # (tiktoken) if exact provider token counts are needed later.
    return max(1, int(len(text.split()) * 1.3))


_approx_tokens = approx_tokens  # backward-compatible alias used elsewhere in this module


def _split_into_clauses(raw_text: str) -> list[str]:
    lines = [l for l in raw_text.split("\n") if l.strip()]
    if not lines:
        return []
    clauses: list[str] = []
    current: list[str] = []
    for line in lines:
        if _CLAUSE_START_RE.match(line) and current:
            clauses.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        clauses.append(" ".join(current))
    return clauses


def _extract_refs(text: str) -> list[str]:
    refs: set[str] = set()
    for match in CROSS_REF_RE.finditer(text):
        refs.update(NUMBER_TOKEN_RE.findall(match.group(1)))
    return sorted(refs)


def _breadcrumb(tree: DocumentTree, node: DocumentNode) -> str:
    ancestors = list(reversed(tree.get_ancestors(node.node_id)))
    parts = [a.title for a in ancestors if a.node_id != "root"]
    parts.append(node.title)
    return " > ".join(parts)


def chunk_node(tree: DocumentTree, node: DocumentNode, document_name: str) -> list[Chunk]:
    if not node.raw_text.strip():
        return []

    clauses = _split_into_clauses(node.raw_text)
    breadcrumb = _breadcrumb(tree, node)
    parent = tree.nodes.get(node.parent_id) if node.parent_id else None
    parent_title = parent.title if parent and parent.node_id != "root" else None

    max_t = settings.chunk_max_tokens
    min_t = settings.chunk_min_tokens
    overlap_t = settings.chunk_overlap_tokens

    # group clauses greedily up to max_tokens, never splitting a clause unless
    # it alone exceeds max_tokens (then, and only then, fall back to sentence split)
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tokens = 0

    for clause in clauses:
        clause_tokens = _approx_tokens(clause)
        if clause_tokens > max_t:
            if current_group:
                groups.append(current_group)
                current_group, current_tokens = [], 0
            sentences = re.split(r"(?<=[.;])\s+", clause)
            sub_group, sub_tokens = [], 0
            for sent in sentences:
                st = _approx_tokens(sent)
                if sub_tokens + st > max_t and sub_group:
                    groups.append(sub_group)
                    sub_group, sub_tokens = [], 0
                sub_group.append(sent)
                sub_tokens += st
            if sub_group:
                groups.append(sub_group)
            continue

        if current_tokens + clause_tokens > max_t and current_tokens >= min_t:
            groups.append(current_group)
            current_group, current_tokens = [], 0
        current_group.append(clause)
        current_tokens += clause_tokens

    if current_group:
        groups.append(current_group)

    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        source_text = " ".join(group).strip()

        # word-level overlap with the tail of the previous chunk, so a reader/embedder
        # never loses context right at a chunk boundary
        if overlap_t > 0 and i > 0:
            prev_words = " ".join(groups[i - 1]).split()
            overlap_words = prev_words[-int(overlap_t / 1.3):] if prev_words else []
            if overlap_words:
                source_text = " ".join(overlap_words) + " " + source_text

        display_text = source_text
        if settings.chunk_include_section_title:
            display_text = f"[{breadcrumb}]\n{source_text}"

        chunk_type = ChunkType.CLAUSE if len(group) == 1 and _CLAUSE_START_RE.match(group[0]) else ChunkType.PARAGRAPH

        chunks.append(Chunk(
            chunk_id=f"{node.node_id}-chunk-{i + 1}",
            document_id=node.document_id,
            document_name=document_name,
            chunk_type=chunk_type,
            node_id=node.node_id,
            section_number=node.section_number,
            section_title=node.title,
            parent_section=parent_title,
            page_start=node.page_start or 0,
            page_end=node.page_end or 0,
            source_text=source_text,
            display_text=display_text,
            token_count=_approx_tokens(source_text),
            referenced_sections=_extract_refs(source_text),
        ))

    return chunks


def chunk_table(table: TableBlock, node: DocumentNode | None, document_name: str) -> Chunk:
    """Tables get their own chunk type so retrieval can special-case them (spec section 18)."""
    md = table.to_markdown()
    return Chunk(
        chunk_id=f"{table.table_id}-chunk",
        document_id=table.document_id,
        document_name=document_name,
        chunk_type=ChunkType.TABLE,
        node_id=node.node_id if node else "root",
        section_number=node.section_number if node else None,
        section_title=node.title if node else None,
        page_start=table.page,
        page_end=table.page,
        source_text=md,
        display_text=md,
        token_count=_approx_tokens(md),
        table_id=table.table_id,
    )


def chunk_document(tree: DocumentTree, tables: list[TableBlock], document_name: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for node_id, node in tree.nodes.items():
        if node_id == "root":
            continue
        node_chunks = chunk_node(tree, node, document_name)
        node.chunk_ids = [c.chunk_id for c in node_chunks]
        all_chunks.extend(node_chunks)

    # Table -> node ownership was already decided in reading order by tree_builder
    # (via TABLE_ANCHOR blocks), so we just look it up rather than re-guessing here.
    table_owner: dict[str, DocumentNode] = {}
    for node in tree.nodes.values():
        for tid in node.table_ids:
            table_owner[tid] = node

    for table in tables:
        owner = table_owner.get(table.table_id)
        table_chunk = chunk_table(table, owner, document_name)
        all_chunks.append(table_chunk)

    return all_chunks
