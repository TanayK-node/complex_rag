"""
Retrieval metrics, computed on a ranked list of Chunks against an
EvalQuestion's `expected_keywords` ground truth (spec section 20: "evaluate
retrieval separately from generation").

Relevance ground truth note: since chunks (Wanting to compare across baselines
with completely different chunking schemes — fixed-size vs. structure-aware)
can't be matched by section_number alone, relevance is defined by whether a
chunk's text contains every expected keyword/phrase. This is a real
methodology choice, not a shortcut: it's the only definition of "relevant"
that is fair across chunkers that don't share a common unit of retrieval.

Recall@K here means "hit@K" (at least one relevant chunk appears in the top
K) rather than "what fraction of all relevant chunks in the corpus were
found" — the latter would require exhaustively labeling every chunk in the
document as relevant/not, which isn't done here. This is documented rather
than silently assumed; it's the standard practical simplification for
single-supporting-passage QA evaluation.
"""
from __future__ import annotations

from schemas.models import Chunk


def is_relevant(chunk: Chunk, expected_keyword_groups: list[list[str]]) -> bool:
    if not expected_keyword_groups:
        return False
    text = chunk.source_text.lower()
    return any(all(kw.lower() in text for kw in group) for group in expected_keyword_groups)


def precision_at_k(ranked_chunks: list[Chunk], expected_keyword_groups: list[list[str]], k: int) -> float:
    top_k = ranked_chunks[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if is_relevant(c, expected_keyword_groups))
    return relevant / len(top_k)


def hit_at_k(ranked_chunks: list[Chunk], expected_keyword_groups: list[list[str]], k: int) -> float:
    """"Recall@K" in the single-supporting-passage sense — see module docstring."""
    return 1.0 if any(is_relevant(c, expected_keyword_groups) for c in ranked_chunks[:k]) else 0.0


def reciprocal_rank(ranked_chunks: list[Chunk], expected_keyword_groups: list[list[str]]) -> float:
    for i, c in enumerate(ranked_chunks, start=1):
        if is_relevant(c, expected_keyword_groups):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_chunks: list[Chunk], expected_keyword_groups: list[list[str]], k: int) -> float:
    """Binary-relevance NDCG@K. Ideal DCG accounts for however many relevant
    chunks actually exist in the ranked list (capped at k) — not just one —
    since some questions (comparison, some multi-hop) genuinely have more
    than one valid supporting chunk via distinct keyword groups."""
    import math
    dcg = 0.0
    for i, c in enumerate(ranked_chunks[:k], start=1):
        if is_relevant(c, expected_keyword_groups):
            dcg += 1.0 / math.log2(i + 1)
    total_relevant = sum(1 for c in ranked_chunks if is_relevant(c, expected_keyword_groups))
    r = min(total_relevant, k)
    if r == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, r + 1))
    return dcg / idcg if idcg > 0 else 0.0
