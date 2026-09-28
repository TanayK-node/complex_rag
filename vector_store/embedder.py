"""
Embedding backend for dense retrieval.

Interface-first, per spec section 26 ("modular enough that ... other retriever
can later be added"): `Embedder` is the contract every retrieval-layer module
talks to; `TfidfEmbedder` is the concrete implementation actually running in
this environment.

Why TF-IDF+SVD instead of sentence-transformers here: a transformer embedding
model requires downloading multi-hundred-MB weights, and this sandbox does not
have the disk headroom for a torch install alongside everything else. TF-IDF+SVD
(a.k.a. classic LSA) is a real, well-understood dense-embedding technique — it
just captures lexical/co-occurrence semantics rather than deep semantic
similarity, so it will under-perform a transformer encoder on paraphrase-heavy
queries. `SentenceTransformerEmbedder` below is a drop-in replacement: swap the
`get_embedder()` factory and nothing else in the retrieval stack changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config.settings import settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Fit on the document corpus (no-op for embedders that need no fitting)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""


class TfidfEmbedder(Embedder):
    """Default embedder for this environment: TF-IDF -> TruncatedSVD (LSA)."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
        )
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(corpus)
        # SVD components can't exceed min(n_samples, n_features) - 1
        n_components = max(2, min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_components
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms  # normalize so FAISS inner-product == cosine similarity


class SentenceTransformerEmbedder(Embedder):
    """
    Production embedder — not wired up in this environment (no disk headroom to
    install torch + download weights here). Kept so the swap is a one-line
    change in get_embedder() once running somewhere with network/disk for it.
    """

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # deferred import
        self.model = SentenceTransformer(model_name or settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, corpus: list[str]) -> None:
        pass  # pretrained model, nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


def get_embedder() -> Embedder:
    """Single place that decides which embedder backend is active."""
    return TfidfEmbedder(dim=128)
