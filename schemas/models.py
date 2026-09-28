"""
Core data contracts used across ingestion, hierarchy, retrieval, and generation.
Keeping these in one module lets every other component agree on the same shapes
and lets us evolve one stage without breaking the others.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    TABLE = "table"
    LIST_ITEM = "list_item"
    MIXED = "mixed"


class BoundingSpan(BaseModel):
    """Raw text span extracted from the PDF, before structure is assigned."""
    text: str
    page: int
    font_size: float
    is_bold: bool = False
    x0: float = 0.0
    y0: float = 0.0


class TableBlock(BaseModel):
    document_id: str
    table_id: str
    page: int
    y0: float = 0.0  # vertical position on the page, used to place the table in reading order
    section_number: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """LLM-readable representation that preserves row/column structure."""
        if not self.headers:
            return "\n".join(" | ".join(r) for r in self.rows)
        md = ["| " + " | ".join(self.headers) + " |",
              "| " + " | ".join(["---"] * len(self.headers)) + " |"]
        for row in self.rows:
            md.append("| " + " | ".join(row) + " |")
        return "\n".join(md)


class DocumentNode(BaseModel):
    """One node in the hierarchical document tree (document/section/subsection/clause)."""
    node_id: str
    document_id: str
    title: str
    section_number: Optional[str] = None   # e.g. "4.2.1"
    depth: int = 0                          # 0 = document root, 1 = top-level section, ...
    parent_id: Optional[str] = None
    child_ids: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    raw_text: str = ""                      # full text belonging to this node (own paragraphs only)
    summary: Optional[str] = None           # populated for higher-level nodes
    chunk_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    referenced_sections: list[str] = Field(default_factory=list)  # cross-references found in text


class Chunk(BaseModel):
    """Retrieval unit. Always knows exactly where it sits in the document tree."""
    chunk_id: str
    document_id: str
    document_name: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    node_id: str                            # the DocumentNode this chunk belongs to
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    source_text: str = ""                   # exact text used for embedding/BM25
    display_text: Optional[str] = None      # source_text optionally prefixed with title/breadcrumb
    token_count: int = 0
    referenced_sections: list[str] = Field(default_factory=list)
    table_id: Optional[str] = None


class DocumentTree(BaseModel):
    document_id: str
    document_name: str
    total_pages: int = 0
    nodes: dict[str, DocumentNode] = Field(default_factory=dict)
    root_id: str = "root"

    def get_children(self, node_id: str) -> list[DocumentNode]:
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.child_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> list[DocumentNode]:
        chain = []
        current = self.nodes.get(node_id)
        while current and current.parent_id and current.parent_id in self.nodes:
            current = self.nodes[current.parent_id]
            chain.append(current)
        return chain

    def find_by_section_number(self, section_number: str) -> Optional[DocumentNode]:
        for node in self.nodes.values():
            if node.section_number == section_number:
                return node
        return None
