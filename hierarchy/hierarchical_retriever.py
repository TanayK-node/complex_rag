"""
Hierarchical retrieval (spec section 6C / 8): use the document tree's titles
and summaries to figure out *which sections* are relevant before falling back
to chunk-level scoring. This is what lets "when can the buyer terminate"
jump straight to Section 4 and its children instead of scoring all 28 nodes
identically the way flat chunk retrieval would.

Scoring is deliberately simple (keyword overlap against title+summary, plus a
big boost for an explicit "Section 4.2" reference in the query) — the point of
this stage isn't semantic subtlety, it's cheaply pruning the tree so the
candidate pool downstream doesn't get overwhelmed with off-topic chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

from retrieval.query_analysis import QueryAnalysis
from schemas.models import DocumentNode, DocumentTree


@dataclass
class HierarchyHit:
    node_id: str
    section_number: str | None
    title: str
    score: float
    chunk_ids: list[str]


def _node_text(node: DocumentNode) -> str:
    return f"{node.title} {node.summary or ''}".lower()


def _collect_descendant_chunk_ids(tree: DocumentTree, node_id: str) -> list[str]:
    node = tree.nodes[node_id]
    ids = list(node.chunk_ids)
    for child_id in node.child_ids:
        ids.extend(_collect_descendant_chunk_ids(tree, child_id))
    return ids


def hierarchical_search(tree: DocumentTree, analysis: QueryAnalysis, top_k: int) -> list[HierarchyHit]:
    scored: list[tuple[float, DocumentNode]] = []

    for node in tree.nodes.values():
        if node.node_id == "root":
            continue
        text = _node_text(node)
        score = 0.0

        # explicit "Section 4.2" reference in the query is a near-certain match
        if node.section_number and node.section_number in analysis.explicit_section_refs:
            score += 10.0

        # candidate topics derived from query intent (e.g. "termination for cause")
        for topic in analysis.candidate_topics:
            if topic in text:
                score += 3.0

        # raw keyword overlap against title/summary
        for kw in analysis.keywords:
            if kw in text:
                score += 1.0

        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_nodes = scored[:top_k]

    hits = []
    for score, node in top_nodes:
        chunk_ids = _collect_descendant_chunk_ids(tree, node.node_id)
        hits.append(HierarchyHit(
            node_id=node.node_id,
            section_number=node.section_number,
            title=node.title,
            score=score,
            chunk_ids=chunk_ids,
        ))
    return hits
