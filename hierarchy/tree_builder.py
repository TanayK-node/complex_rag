"""
Stage 3 of ingestion: turn the flat, ordered list of StructuredBlocks into a
DocumentTree. A stack keyed by depth tracks the currently "open" ancestor at
each level; when a new heading arrives at depth D, we close everything deeper
than D and attach the new node under whatever is open at depth D-1.

This is the structure the retrieval layer later walks top-down
(document -> section -> subsection -> clause) instead of treating every
chunk as an independent, context-free blob.
"""
from __future__ import annotations

from ingestion.structure_detector import BlockType, StructuredBlock
from schemas.models import DocumentNode, DocumentTree


def build_tree(document_id: str, document_name: str, total_pages: int,
                blocks: list[StructuredBlock]) -> DocumentTree:
    root = DocumentNode(
        node_id="root",
        document_id=document_id,
        title=document_name,
        depth=0,
    )
    tree = DocumentTree(document_id=document_id, document_name=document_name,
                         total_pages=total_pages, root_id="root")
    tree.nodes["root"] = root

    # stack[d] = node_id of the currently open node at depth d
    stack: dict[int, str] = {0: "root"}
    node_counter = 0

    def current_parent_for_depth(depth: int) -> str:
        # find the nearest open ancestor at depth-1, depth-2, ...
        for d in range(depth - 1, -1, -1):
            if d in stack:
                return stack[d]
        return "root"

    current_leaf_id = "root"  # node that in-flight paragraph text attaches to

    for block in blocks:
        if block.block_type == BlockType.HEADING:
            node_counter += 1
            node_id = f"{document_id}-node-{node_counter}"
            parent_id = current_parent_for_depth(block.depth)

            # close deeper levels that are no longer open
            stack = {d: nid for d, nid in stack.items() if d < block.depth}
            stack[block.depth] = node_id

            node = DocumentNode(
                node_id=node_id,
                document_id=document_id,
                title=block.title or block.text,
                section_number=block.section_number,
                depth=block.depth,
                parent_id=parent_id,
                page_start=block.page,
                page_end=block.page,
            )
            tree.nodes[node_id] = node
            tree.nodes[parent_id].child_ids.append(node_id)
            current_leaf_id = node_id

        elif block.block_type == BlockType.TABLE_ANCHOR:
            tree.nodes[current_leaf_id].table_ids.append(block.table_id)

        else:  # paragraph -> belongs to whatever node is currently open (deepest)
            leaf = tree.nodes[current_leaf_id]
            leaf.raw_text = (leaf.raw_text + "\n" + block.text).strip()
            leaf.page_end = max(leaf.page_end or block.page, block.page)
            if block.referenced_sections:
                for ref in block.referenced_sections:
                    if ref not in leaf.referenced_sections:
                        leaf.referenced_sections.append(ref)

    _backfill_page_ranges(tree, "root")
    return tree


def _backfill_page_ranges(tree: DocumentTree, node_id: str) -> tuple[int, int]:
    """Ensure parent nodes' page ranges span all of their descendants."""
    node = tree.nodes[node_id]
    starts = [node.page_start] if node.page_start else []
    ends = [node.page_end] if node.page_end else []
    for child_id in node.child_ids:
        c_start, c_end = _backfill_page_ranges(tree, child_id)
        starts.append(c_start)
        ends.append(c_end)
    if starts:
        node.page_start = min(s for s in starts if s)
    if ends:
        node.page_end = max(e for e in ends if e)
    return node.page_start or 0, node.page_end or 0
