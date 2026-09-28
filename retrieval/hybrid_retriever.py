"""
Combines the three retrieval mechanisms (spec sections 6, 9) into one ranked
candidate pool, then does one hop of cross-reference expansion (spec section
12): if a top candidate's text says "subject to Section 6.2", pull Section
6.2's chunks in too, tagged distinctly so the reranker/context builder can
tell "the query matched this" apart from "this was pulled in because it's
referenced by something that matched."

Score fusion is deliberately simple min-max normalization + weighted sum
per source, not because a fancier fusion (e.g. reciprocal rank fusion) wouldn't
help, but because with three very differently-scaled scorers (cosine sim,
BM25, additive keyword-overlap), min-max is the safe default. Swappable in
one function (`_fuse`) if RRF proves better under the eval framework in
Phase 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import settings
from hierarchy.hierarchical_retriever import hierarchical_search
from lexical_search.bm25_index import LexicalIndex
from retrieval.query_analysis import QueryAnalysis, analyze_query
from schemas.models import Chunk, DocumentTree
from vector_store.faiss_index import VectorStore

SOURCE_VECTOR = "vector"
SOURCE_LEXICAL = "lexical"
SOURCE_HIERARCHY = "hierarchy"
SOURCE_CROSS_REF = "cross_reference"


@dataclass
class RetrievalCandidate:
    chunk_id: str
    chunk: Chunk
    scores: dict[str, float] = field(default_factory=dict)  # per-source raw score
    provenance: list[str] = field(default_factory=list)      # which sources contributed
    fused_score: float = 0.0


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def hybrid_retrieve(
    query: str,
    tree: DocumentTree,
    chunks_by_id: dict[str, Chunk],
    vector_store: VectorStore,
    lexical_index: LexicalIndex,
    analysis: QueryAnalysis | None = None,
    use_hierarchy: bool = True,
    use_cross_reference: bool = True,
) -> tuple[list[RetrievalCandidate], QueryAnalysis]:
    """
    use_hierarchy / use_cross_reference default to True (full pipeline). Both
    exist mainly so evaluation/baselines.py can turn pieces off one at a time
    to reproduce spec section 21's baseline ablation (hybrid-without-hierarchy
    vs. hybrid-with-hierarchy) using this exact same fusion code, rather than
    a separately-maintained copy that could silently drift from production.
    """
    analysis = analysis or analyze_query(query)

    vector_hits = vector_store.search(query, settings.vector_top_k)
    lexical_hits = lexical_index.search(query, settings.bm25_top_k) if lexical_index else []
    hierarchy_hits = hierarchical_search(tree, analysis, settings.hierarchy_top_k) if (use_hierarchy and tree) else []

    vector_scores = _minmax({h.chunk_id: h.score for h in vector_hits})
    lexical_scores = _minmax({h.chunk_id: h.score for h in lexical_hits})
    hierarchy_chunk_scores: dict[str, float] = {}
    for hit in hierarchy_hits:
        for cid in hit.chunk_ids:
            # a chunk can belong to multiple matched sections; keep the max
            hierarchy_chunk_scores[cid] = max(hierarchy_chunk_scores.get(cid, 0.0), hit.score)
    hierarchy_scores = _minmax(hierarchy_chunk_scores)

    pool: dict[str, RetrievalCandidate] = {}

    def _add(chunk_id: str, source: str, score: float):
        if chunk_id not in chunks_by_id:
            return
        cand = pool.setdefault(chunk_id, RetrievalCandidate(chunk_id=chunk_id, chunk=chunks_by_id[chunk_id]))
        cand.scores[source] = score
        if source not in cand.provenance:
            cand.provenance.append(source)

    for cid, s in vector_scores.items():
        _add(cid, SOURCE_VECTOR, s)
    for cid, s in lexical_scores.items():
        _add(cid, SOURCE_LEXICAL, s)
    for cid, s in hierarchy_scores.items():
        _add(cid, SOURCE_HIERARCHY, s)

    # fuse: average across whichever sources actually fired for this chunk
    # (a chunk found by all three sources should outrank one found by only one)
    for cand in pool.values():
        cand.fused_score = sum(cand.scores.values()) / max(1, len(cand.scores))
        cand.fused_score *= (1 + 0.15 * (len(cand.scores) - 1))  # small agreement bonus

    # an explicit "Section 4.2" in the query is a near-deterministic signal that
    # should outrank chunks that merely share vocabulary with the query (e.g. a
    # different section that happens to also use the word "termination"). Fusion
    # alone under-weights this because it only has one contributing source.
    if analysis.explicit_section_refs:
        for cand in pool.values():
            if cand.chunk.section_number in analysis.explicit_section_refs:
                cand.fused_score += 5.0
                if "explicit_reference" not in cand.provenance:
                    cand.provenance.append("explicit_reference")

    # --- one hop of cross-reference expansion (spec section 12) ---
    if use_cross_reference and tree is not None:
        seed_ids = list(pool.keys())
        for cid in seed_ids:
            chunk = chunks_by_id[cid]
            for ref_section in chunk.referenced_sections[: settings.max_cross_reference_hops * 5]:
                ref_node = tree.find_by_section_number(ref_section)
                if not ref_node:
                    continue
                for ref_chunk_id in ref_node.chunk_ids:
                    if ref_chunk_id in pool:
                        if SOURCE_CROSS_REF not in pool[ref_chunk_id].provenance:
                            pool[ref_chunk_id].provenance.append(SOURCE_CROSS_REF)
                        continue
                    if ref_chunk_id not in chunks_by_id:
                        continue
                    cand = RetrievalCandidate(
                        chunk_id=ref_chunk_id,
                        chunk=chunks_by_id[ref_chunk_id],
                        scores={SOURCE_CROSS_REF: 0.3},  # modest fixed score; reranker has final say
                        provenance=[SOURCE_CROSS_REF],
                        fused_score=0.3,
                    )
                    pool[ref_chunk_id] = cand

    ranked = sorted(pool.values(), key=lambda c: c.fused_score, reverse=True)
    return ranked, analysis
