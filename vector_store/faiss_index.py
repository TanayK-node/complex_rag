"""
Thin wrapper around a FAISS index, keyed by chunk_id rather than raw index
position so callers never have to reason about FAISS's internal integer ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from schemas.models import Chunk
from vector_store.embedder import Embedder


@dataclass
class VectorHit:
    chunk_id: str
    score: float  # cosine similarity (embeddings are L2-normalized), higher = better


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        texts = [c.source_text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vecs = cosine
        self.index.add(vectors)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before search()")
        qvec = self.embedder.encode([query])
        scores, idxs = self.index.search(qvec, min(top_k, len(self.chunk_ids)))
        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hits.append(VectorHit(chunk_id=self.chunk_ids[idx], score=float(score)))
        return hits
