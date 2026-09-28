"""
Keyword/lexical retrieval (spec section 6B). BM25 catches exact-term matches
that dense retrieval can miss — defined terms, party names, specific figures
like "$2,000,000" — which matter a lot in legal text where wording is precise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from schemas.models import Chunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class LexicalHit:
    chunk_id: str
    score: float


class LexicalIndex:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [_tokenize(c.source_text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int) -> list[LexicalHit]:
        if self.bm25 is None:
            raise RuntimeError("LexicalIndex.build() must be called before search()")
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [LexicalHit(chunk_id=cid, score=float(s)) for cid, s in ranked[:top_k] if s > 0]
