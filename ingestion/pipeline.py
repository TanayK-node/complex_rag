"""
Ties together stages 1-4 of ingestion (spec section 3) into a single
entry point: PDF path in, (DocumentTree, list[Chunk]) out.

Deliberately a thin orchestrator with no logic of its own, so each stage
stays independently testable and swappable.
"""
from __future__ import annotations

import uuid

from ingestion.pdf_parser import parse_pdf
from ingestion.structure_detector import detect_structure
from ingestion.chunker import chunk_document
from hierarchy.tree_builder import build_tree
from hierarchy.summarizer import summarize_tree
from schemas.models import Chunk, DocumentTree


def ingest_pdf(file_path: str, document_name: str | None = None) -> tuple[DocumentTree, list[Chunk]]:
    document_id = str(uuid.uuid4())[:8]
    document_name = document_name or file_path.split("/")[-1]

    parsed = parse_pdf(file_path, document_id, document_name)
    blocks = detect_structure(parsed)
    tree = build_tree(document_id, document_name, parsed.total_pages, blocks)
    chunks = chunk_document(tree, parsed.tables, document_name)
    summarize_tree(tree)  # populates node.summary for higher-level nodes

    return tree, chunks
