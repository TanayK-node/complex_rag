"""
Phase 1 smoke test. Not a unit test with mocks — an end-to-end run against
the synthetic sample contract, with printed output so the tree/chunk quality
can actually be eyeballed (per spec: "do not claim something works unless it
has actually been tested").
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import ingest_pdf

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "master_service_agreement.pdf")


def print_tree(tree, node_id="root", indent=0):
    node = tree.nodes[node_id]
    label = f"{node.section_number + ' ' if node.section_number else ''}{node.title}"
    pages = f"(p{node.page_start}-{node.page_end})" if node.page_start else ""
    print("  " * indent + f"- {label} {pages} [{len(node.chunk_ids)} chunks, {len(node.table_ids)} tables]")
    for child_id in node.child_ids:
        print_tree(tree, child_id, indent + 1)


def main():
    tree, chunks = ingest_pdf(SAMPLE_PDF, document_name="Master Service Agreement")

    print("=" * 70)
    print(f"DOCUMENT TREE  ({len(tree.nodes)} nodes, {tree.total_pages} pages)")
    print("=" * 70)
    print_tree(tree)

    print()
    print("=" * 70)
    print(f"CHUNKS  ({len(chunks)} total)")
    print("=" * 70)
    for c in chunks:
        refs = f" refs={c.referenced_sections}" if c.referenced_sections else ""
        print(f"[{c.chunk_id}] type={c.chunk_type.value} sec={c.section_number} "
              f"page={c.page_start}-{c.page_end} tokens={c.token_count}{refs}")
        preview = c.source_text[:110].replace("\n", " ")
        print(f"    \"{preview}...\"" if len(c.source_text) > 110 else f"    \"{preview}\"")

    print()
    print("=" * 70)
    print("CROSS-REFERENCE CHECK (spec section 12)")
    print("=" * 70)
    node_4_2 = tree.find_by_section_number("4.2")
    if node_4_2:
        print(f"Section 4.2 references detected: {node_4_2.referenced_sections}")

    print()
    print("=" * 70)
    print("SAMPLE NODE SUMMARY (spec section 4)")
    print("=" * 70)
    for sec in ["4", "3"]:
        node = tree.find_by_section_number(sec)
        if node:
            print(f"[{sec}] {node.title} -> {node.summary}")

    # basic assertions so failures are loud, not silent
    assert len(tree.nodes) > 5, "tree looks too flat — structure detection likely failed"
    assert any(c.chunk_type.value == "table" for c in chunks), "fee table was not captured as a chunk"
    assert node_4_2 and "6.2" in node_4_2.referenced_sections, "cross-reference extraction failed for Section 4.2"
    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    main()
