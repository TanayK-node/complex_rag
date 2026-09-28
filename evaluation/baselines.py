"""
Spec section 21: build the 5 baselines and let evaluate.py run the same
question set through each, so the comparison actually measures what each
added component (structure, hybrid, hierarchy, reranking) contributes,
rather than five independently-coded pipelines that might differ in ways
that aren't the thing being tested.

BASELINE 1: fixed-size chunking + dense-only retrieval
BASELINE 2: structure-aware chunking + dense-only retrieval
BASELINE 3: structure-aware chunking + hybrid (dense+BM25), no hierarchy signal
BASELINE 4: + hierarchical retrieval signal (full hybrid_retrieve)
BASELINE 5: + reranking on top of baseline 4
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ingestion.pdf_parser import parse_pdf
from ingestion.naive_chunker import fixed_size_chunks
from ingestion.pipeline import ingest_pdf
from lexical_search.bm25_index import LexicalIndex
from reranking.heuristic_reranker import get_reranker
from retrieval.hybrid_retriever import hybrid_retrieve
from retrieval.query_analysis import analyze_query
from schemas.models import Chunk, DocumentTree
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore

BASELINE_NAMES = {
    1: "Fixed-size chunking + dense-only",
    2: "Structure-aware chunking + dense-only",
    3: "Hybrid (dense+BM25), no hierarchy",
    4: "Hybrid + hierarchical retrieval",
    5: "Hybrid + hierarchical + reranking",
}


@dataclass
class BaselineArtifacts:
    name: str
    chunks: list[Chunk]
    chunks_by_id: dict[str, Chunk]
    vector_store: VectorStore
    lexical_index: LexicalIndex | None
    tree: DocumentTree | None


def build_all_baselines(pdf_path: str, document_name: str) -> dict[int, BaselineArtifacts]:
    """Builds all 5 baselines from one PDF. Parses/ingests only as many times
    as structurally necessary (once for the naive path, once for the
    structure-aware path — baselines 2-5 share the same tree/chunks)."""
    baselines: dict[int, BaselineArtifacts] = {}

    # --- Baseline 1: fixed-size chunking, dense only ---
    parsed = parse_pdf(pdf_path, document_id="baseline1", document_name=document_name)
    fixed_chunks = fixed_size_chunks(parsed, document_name)
    vs1 = VectorStore(get_embedder())
    vs1.build(fixed_chunks)
    baselines[1] = BaselineArtifacts(
        name=BASELINE_NAMES[1], chunks=fixed_chunks,
        chunks_by_id={c.chunk_id: c for c in fixed_chunks},
        vector_store=vs1, lexical_index=None, tree=None,
    )

    # --- shared structure-aware ingestion for baselines 2-5 ---
    tree, chunks = ingest_pdf(pdf_path, document_name=document_name)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    vs2 = VectorStore(get_embedder())
    vs2.build(chunks)
    baselines[2] = BaselineArtifacts(
        name=BASELINE_NAMES[2], chunks=chunks, chunks_by_id=chunks_by_id,
        vector_store=vs2, lexical_index=None, tree=None,  # dense-only: no tree used
    )

    lexical_index = LexicalIndex()
    lexical_index.build(chunks)

    for n in (3, 4, 5):
        baselines[n] = BaselineArtifacts(
            name=BASELINE_NAMES[n], chunks=chunks, chunks_by_id=chunks_by_id,
            vector_store=vs2, lexical_index=lexical_index, tree=tree,
        )

    return baselines


def retrieve_with_baseline(baseline_id: int, artifacts: BaselineArtifacts, query: str, top_k: int):
    """Returns (ranked_chunks: list[Chunk], latency_seconds: float)."""
    start = time.time()

    if baseline_id == 1:
        hits = artifacts.vector_store.search(query, top_k)
        ranked = [artifacts.chunks_by_id[h.chunk_id] for h in hits]

    elif baseline_id == 2:
        hits = artifacts.vector_store.search(query, top_k)
        ranked = [artifacts.chunks_by_id[h.chunk_id] for h in hits]

    elif baseline_id == 3:
        candidates, _ = hybrid_retrieve(
            query, artifacts.tree, artifacts.chunks_by_id, artifacts.vector_store,
            artifacts.lexical_index, use_hierarchy=False, use_cross_reference=False,
        )
        ranked = [c.chunk for c in candidates[:top_k]]

    elif baseline_id == 4:
        candidates, _ = hybrid_retrieve(
            query, artifacts.tree, artifacts.chunks_by_id, artifacts.vector_store,
            artifacts.lexical_index, use_hierarchy=True, use_cross_reference=True,
        )
        ranked = [c.chunk for c in candidates[:top_k]]

    elif baseline_id == 5:
        candidates, _ = hybrid_retrieve(
            query, artifacts.tree, artifacts.chunks_by_id, artifacts.vector_store,
            artifacts.lexical_index, use_hierarchy=True, use_cross_reference=True,
        )
        reranked = get_reranker().rerank(query, candidates, top_k=top_k)
        ranked = [rc.candidate.chunk for rc in reranked]

    else:
        raise ValueError(f"unknown baseline id: {baseline_id}")

    latency = time.time() - start
    return ranked, latency
