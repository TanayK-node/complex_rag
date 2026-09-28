"""
Concrete Reranker used in this environment.

A real cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, per
config.settings.reranker_model) jointly encodes (query, chunk) pairs and is
strictly better at judging fine-grained relevance than anything query-side
retrieval can do — but it's another transformer download, and this sandbox
already hit a disk wall installing sentence-transformers for the embedder.

So HeuristicReranker recomputes relevance from features the fusion stage
does NOT use: exact phrase overlap, query-term recall/precision against the
*whole* chunk (not just top-k membership), and whether the chunk's own
section title matches the query. This is a genuinely different signal from
the retrieval-stage fused_score, not a rename of it — which is the point of
having a separate reranking stage at all (spec: "should consider semantic
relevance rather than blindly trusting initial ... similarity").

`CrossEncoderReranker` and `LLMReranker` below are the swap targets once
running somewhere with the disk/API budget for them; nothing outside this
file needs to change to switch (see reranking/README_SWAP.md).
"""
from __future__ import annotations

import re

from reranking.base import Reranker, RerankedCandidate
from retrieval.hybrid_retriever import RetrievalCandidate

_TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or",
              "for", "on", "what", "does", "do", "under", "this", "that"}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)} - _STOPWORDS


def _phrase_hits(query: str, text: str) -> int:
    """Count query bigrams/trigrams that appear verbatim in the chunk — a much
    stronger relevance signal than unordered token overlap."""
    q_tokens = [t.lower() for t in _TOKEN_RE.findall(query) if t.lower() not in _STOPWORDS]
    text_lower = text.lower()
    hits = 0
    for n in (3, 2):
        for i in range(len(q_tokens) - n + 1):
            phrase = " ".join(q_tokens[i:i + n])
            if len(phrase) > 5 and phrase in text_lower:
                hits += 1
    return hits


class HeuristicReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RerankedCandidate]:
        q_tokens = _tokens(query)
        scored: list[RerankedCandidate] = []

        for cand in candidates:
            chunk = cand.chunk
            c_tokens = _tokens(chunk.source_text)
            title_tokens = _tokens(chunk.section_title or "")

            overlap = q_tokens & c_tokens
            recall = len(overlap) / max(1, len(q_tokens))          # does the chunk cover what was asked?
            precision = len(overlap) / max(1, len(c_tokens)) if c_tokens else 0.0
            phrase_score = _phrase_hits(query, chunk.source_text)
            title_match = len(q_tokens & title_tokens) / max(1, len(q_tokens))

            # cross-reference-sourced candidates get a small penalty by default:
            # they were pulled in because something *else* referenced them, not
            # because they matched the query, so they need real signal to rank high
            provenance_prior = -0.15 if cand.provenance == ["cross_reference"] else 0.0

            score = (
                0.40 * recall
                + 0.15 * precision
                + 0.25 * min(1.0, phrase_score / 2)
                + 0.20 * title_match
                + provenance_prior
                + 0.10 * cand.fused_score  # retain some weight from retrieval-stage agreement
            )

            rationale_bits = []
            if title_match > 0:
                rationale_bits.append("section title matches query")
            if phrase_score:
                rationale_bits.append(f"{phrase_score} exact phrase match(es)")
            if recall > 0.5:
                rationale_bits.append("covers most query terms")
            if cand.provenance == ["cross_reference"]:
                rationale_bits.append("pulled in via cross-reference only")
            rationale = "; ".join(rationale_bits) or "weak lexical overlap"

            scored.append(RerankedCandidate(candidate=cand, rerank_score=score, rationale=rationale))

        scored.sort(key=lambda r: r.rerank_score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker(Reranker):
    """Not active here (no disk budget for the model download in this sandbox).
    Swap target: set RAG_RERANKER_MODEL and flip get_reranker() to this class."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder  # deferred import
        from config.settings import settings
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RerankedCandidate]:
        pairs = [(query, c.chunk.source_text) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [RerankedCandidate(candidate=c, rerank_score=float(s)) for c, s in ranked[:top_k]]


class LLMReranker(Reranker):
    """Swap target for an LLM-based reranker (e.g. asking Claude to score each
    chunk's relevance 0-10). Not wired to a live API call in this environment;
    left as the documented seam per spec section 10 ("later replaced by ...
    LLM-based reranker")."""

    def __init__(self, llm_call_fn):
        self.llm_call_fn = llm_call_fn  # injected callable: (query, chunk_text) -> float

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RerankedCandidate]:
        scored = [
            RerankedCandidate(candidate=c, rerank_score=self.llm_call_fn(query, c.chunk.source_text))
            for c in candidates
        ]
        scored.sort(key=lambda r: r.rerank_score, reverse=True)
        return scored[:top_k]


def get_reranker() -> Reranker:
    """Single place that decides which reranker backend is active."""
    return HeuristicReranker()
