"""
Generates the `summary` field on higher-level DocumentNodes (spec section 4),
which hierarchical retrieval uses to decide "does this section look relevant"
before descending into its children/chunks.

Phase 1 implementation is deliberately extractive (no LLM call) so ingestion
has no external dependency and stays cheap/fast/deterministic to test. This
is the seam to swap in an LLM-based abstractive summarizer later without
touching tree_builder or the retrieval layer, which only consume `node.summary`.
"""
from __future__ import annotations

from schemas.models import DocumentNode, DocumentTree

_MAX_SUMMARY_CHARS = 240


def _collect_text(tree: DocumentTree, node: DocumentNode) -> str:
    parts = [node.raw_text] if node.raw_text else []
    for child in tree.get_children(node.node_id):
        if child.raw_text:
            parts.append(child.raw_text)
        else:
            parts.append(child.title)
    return " ".join(parts)


def summarize_tree(tree: DocumentTree) -> None:
    for node in tree.nodes.values():
        if node.node_id == "root":
            child_titles = [tree.nodes[c].title for c in node.child_ids]
            node.summary = f"{node.title}. Covers: {', '.join(child_titles)}." if child_titles else node.title
            continue

        # leaf nodes with their own body text: summary is just a lead-in snippet
        if not node.child_ids:
            text = node.raw_text.strip()
            node.summary = (text[:_MAX_SUMMARY_CHARS] + "…") if len(text) > _MAX_SUMMARY_CHARS else text
            continue

        # parent nodes: summarize by listing child section titles (a cheap but
        # effective signal for "does the query's topic live under this branch")
        child_titles = [tree.nodes[c].title for c in node.child_ids]
        base = f"{node.title}. Includes: {', '.join(child_titles)}."
        node.summary = base[:_MAX_SUMMARY_CHARS]
