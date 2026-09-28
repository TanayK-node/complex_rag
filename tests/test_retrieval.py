import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import ingest_pdf
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore
from lexical_search.bm25_index import LexicalIndex
from retrieval.hybrid_retriever import hybrid_retrieve

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "master_service_agreement.pdf")

TEST_QUERIES = [
    "When can the Client terminate the agreement for cause?",
    "What are the vendor's termination rights if the client fails to make payments, and what notice period is required?",
    "Compare termination for cause and termination for convenience.",
    "What does Section 4.2 say about termination?",
    "What is the monthly fee for the Enterprise tier?",
    "What is the maximum liability cap under this agreement?",
]


def main():
    tree, chunks = ingest_pdf(SAMPLE_PDF, document_name="Master Service Agreement")
    chunks_by_id = {c.chunk_id: c for c in chunks}

    vector_store = VectorStore(get_embedder())
    vector_store.build(chunks)

    lexical_index = LexicalIndex()
    lexical_index.build(chunks)

    for query in TEST_QUERIES:
        print("=" * 78)
        print(f"QUERY: {query}")
        print("=" * 78)
        candidates, analysis = hybrid_retrieve(query, tree, chunks_by_id, vector_store, lexical_index)

        print(f"  query analysis: keywords={analysis.keywords[:6]} "
              f"explicit_refs={analysis.explicit_section_refs} "
              f"topics={analysis.candidate_topics} "
              f"comparison={analysis.is_comparison} multi_hop={analysis.is_multi_hop}")
        print()

        for cand in candidates[:6]:
            c = cand.chunk
            print(f"  [{cand.fused_score:.3f}] sec={c.section_number or '-':6} "
                  f"type={c.chunk_type.value:9} sources={cand.provenance}")
            preview = c.source_text[:90].replace("\n", " ")
            print(f"           \"{preview}\"")
        print()

    # --- assertions: loud failures over silent hope ---
    cands, _ = hybrid_retrieve(TEST_QUERIES[0], tree, chunks_by_id, vector_store, lexical_index)
    top_sections = {c.chunk.section_number for c in cands[:5]}
    assert "4.2" in top_sections or "4.2.1" in top_sections, \
        f"termination-for-cause query should surface Section 4.2, got {top_sections}"

    cands, analysis = hybrid_retrieve(TEST_QUERIES[3], tree, chunks_by_id, vector_store, lexical_index)
    assert "4.2" in analysis.explicit_section_refs, "explicit 'Section 4.2' reference was not extracted"
    assert cands[0].chunk.section_number == "4.2", "explicit section reference should force it to the top"

    cands, _ = hybrid_retrieve(TEST_QUERIES[4], tree, chunks_by_id, vector_store, lexical_index)
    assert any(c.chunk.chunk_type.value == "table" for c in cands[:5]), \
        "fee-schedule query should retrieve the table chunk"

    cands, analysis = hybrid_retrieve(TEST_QUERIES[1], tree, chunks_by_id, vector_store, lexical_index)
    assert analysis.is_multi_hop, "payment+notice-period query should be flagged multi-hop"
    top_sections = {c.chunk.section_number for c in cands[:8]}
    assert {"3.1", "3.4"} & top_sections, "multi-hop query should surface payment sections"

    print("All Phase 2 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
