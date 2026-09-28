import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import ingest_pdf
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore
from lexical_search.bm25_index import LexicalIndex
from retrieval.hybrid_retriever import hybrid_retrieve
from reranking.heuristic_reranker import get_reranker
from context.context_builder import build_context
from config.settings import settings

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "master_service_agreement.pdf")

QUERIES = [
    "When can the Client terminate the agreement for cause, and is there a cure period?",
    "What is the monthly fee for the Enterprise tier and what response SLA does it come with?",
]


def main():
    tree, chunks = ingest_pdf(SAMPLE_PDF, document_name="Master Service Agreement")
    chunks_by_id = {c.chunk_id: c for c in chunks}

    vector_store = VectorStore(get_embedder())
    vector_store.build(chunks)
    lexical_index = LexicalIndex()
    lexical_index.build(chunks)
    reranker = get_reranker()

    for query in QUERIES:
        print("=" * 78)
        print(f"QUERY: {query}")
        print("=" * 78)

        candidates, analysis = hybrid_retrieve(query, tree, chunks_by_id, vector_store, lexical_index)
        reranked = reranker.rerank(query, candidates, top_k=settings.rerank_top_k)

        print("-- reranked candidates --")
        for rc in reranked:
            print(f"  [{rc.rerank_score:.3f}] sec={rc.candidate.chunk.section_number or '-':6} "
                  f"({rc.rationale})")

        # deliberately tight budget so we actually exercise the trimming path
        bundle = build_context(reranked, max_tokens=180)

        print(f"\n-- assembled context ({bundle.total_tokens} tokens, "
              f"{len(bundle.included_chunk_ids)} chunks included, "
              f"{len(bundle.dropped_for_budget)} dropped for budget) --")
        print(bundle.context_text)
        print()

    # --- assertions ---
    candidates, _ = hybrid_retrieve(QUERIES[0], tree, chunks_by_id, vector_store, lexical_index)
    reranked = reranker.rerank(QUERIES[0], candidates, top_k=settings.rerank_top_k)
    assert reranked[0].candidate.chunk.section_number in ("4.2", "4.2.1"), \
        f"termination-for-cause query should rerank Section 4.2/4.2.1 to the top, got {reranked[0].candidate.chunk.section_number}"

    bundle = build_context(reranked, max_tokens=180)
    assert bundle.total_tokens <= 220, f"context exceeded budget by more than rendering overhead: {bundle.total_tokens}"
    assert bundle.dropped_for_budget, "tight budget should have forced at least one chunk to be dropped"
    # no duplicated citation header for chunks sharing the same section
    section_numbers_in_output = [b.section_number for b in bundle.blocks]
    assert len(section_numbers_in_output) == len(set(section_numbers_in_output)), \
        "context should not repeat a section header — grouping failed"

    candidates2, _ = hybrid_retrieve(QUERIES[1], tree, chunks_by_id, vector_store, lexical_index)
    reranked2 = reranker.rerank(QUERIES[1], candidates2, top_k=settings.rerank_top_k)
    bundle2 = build_context(reranked2, max_tokens=settings.max_context_tokens)
    assert any(b.section_number == "3.2" for b in bundle2.blocks), "fee/SLA query should include the Section 3.2 fee table"

    print("All Phase 3 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
