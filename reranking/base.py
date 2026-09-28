"""
Reranking stage (spec section 10). Interface-first so the concrete scorer can
be swapped without touching hybrid_retriever or context_builder, both of which
only depend on this ABC's `rerank()` contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from retrieval.hybrid_retriever import RetrievalCandidate


@dataclass
class RerankedCandidate:
    candidate: RetrievalCandidate
    rerank_score: float
    rationale: str = ""  # short human-readable reason, useful for the UI's "why this chunk" view


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RerankedCandidate]:
        ...
