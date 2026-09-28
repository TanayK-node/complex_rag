#!/usr/bin/env python3
"""
Usage:
    python3 scripts/index_document.py path/to/document.pdf [--name "Display Name"] [--out data/processed/my-doc]

Ingests the PDF (Phase 1), builds the dense + lexical indices (Phase 2), and
persists everything to --out so scripts/ask.py and scripts/evaluate.py can
load it without re-ingesting. Run this once per document.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import ingest_pdf
from ingestion.persistence import save_index
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore
from lexical_search.bm25_index import LexicalIndex


def main():
    parser = argparse.ArgumentParser(description="Ingest and index a PDF for the legal RAG pipeline.")
    parser.add_argument("pdf_path", help="Path to the PDF to ingest")
    parser.add_argument("--name", default=None, help="Display name for the document (default: filename)")
    parser.add_argument("--out", default=None, help="Output directory for the persisted index "
                                                      "(default: data/processed/<slugified filename>)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"ERROR: file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    document_name = args.name or os.path.splitext(os.path.basename(args.pdf_path))[0]
    out_dir = args.out or os.path.join("data", "processed", _slug(document_name))

    print(f"Ingesting '{args.pdf_path}' as '{document_name}' ...")
    tree, chunks = ingest_pdf(args.pdf_path, document_name=document_name)
    print(f"  -> {len(tree.nodes)} tree nodes, {len(chunks)} chunks, {tree.total_pages} pages")

    print("Building dense (TF-IDF/SVD + FAISS) and lexical (BM25) indices ...")
    vector_store = VectorStore(get_embedder())
    vector_store.build(chunks)
    lexical_index = LexicalIndex()
    lexical_index.build(chunks)

    save_index(out_dir, tree, chunks, vector_store, lexical_index)
    print(f"Saved index to: {out_dir}")
    print(f"\nNext steps:\n"
          f"  python3 scripts/ask.py {out_dir}\n"
          f"  python3 scripts/evaluate.py {out_dir} <questions.json>")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


if __name__ == "__main__":
    main()
