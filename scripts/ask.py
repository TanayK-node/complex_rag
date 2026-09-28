#!/usr/bin/env python3
"""
Usage:
    python3 scripts/ask.py data/processed/my-doc [--verbose]

Interactive REPL: type a question, get a grounded answer with citations.
Type 'exit' or 'quit' (or Ctrl-D) to stop. Uses a real Anthropic call if
ANTHROPIC_API_KEY / RAG_LLM_API_KEY is set, otherwise falls back to the
deterministic mock provider (see generation/providers.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.persistence import load_index, index_exists
from reranking.heuristic_reranker import get_reranker
from generation.providers import get_llm_provider, describe_provider
from generation.pipeline import answer_query


def main():
    parser = argparse.ArgumentParser(description="Ask questions against an indexed document.")
    parser.add_argument("index_dir", help="Directory produced by scripts/index_document.py")
    parser.add_argument("--verbose", action="store_true", help="Show retrieval trace + groundedness detail")
    args = parser.parse_args()

    if not index_exists(args.index_dir):
        print(f"ERROR: no index found at {args.index_dir}. Run scripts/index_document.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading index from {args.index_dir} ...")
    tree, chunks, chunks_by_id, vector_store, lexical_index = load_index(args.index_dir)
    reranker = get_reranker()
    llm = get_llm_provider()
    print(f"Loaded '{tree.document_name}' ({len(chunks)} chunks). "
          f"LLM provider: {describe_provider(llm)}")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        result = answer_query(question, tree, chunks_by_id, vector_store, lexical_index, reranker, llm)

        print(f"\n{result.answer}\n")
        if args.verbose:
            print(f"[groundedness: passed={result.groundedness.passed} "
                  f"insufficient_evidence={result.groundedness.is_insufficient_evidence} "
                  f"retries={result.retries_used}]")
            print(f"[context: {len(result.context.blocks)} section(s), {result.context.total_tokens} tokens]")
            for b in result.context.blocks:
                print(f"    - {b.citation}")
            print()


if __name__ == "__main__":
    main()
