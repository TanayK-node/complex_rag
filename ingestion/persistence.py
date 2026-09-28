"""
Persists a document's full index (tree, chunks, dense index, lexical index)
to disk so `scripts/ask.py` and `scripts/evaluate.py` can load an
already-ingested document instead of re-running ingestion (and re-fitting
TF-IDF/SVD) on every invocation. Not a database — just one directory per
document under data/processed/, which is enough for a prototype and mirrors
where a real metadata DB (spec section 3's "persistent storage") would plug
in later.
"""
from __future__ import annotations

import json
import os
import pickle

import faiss

from lexical_search.bm25_index import LexicalIndex
from schemas.models import Chunk, DocumentTree
from vector_store.faiss_index import VectorStore


def save_index(out_dir: str, tree: DocumentTree, chunks: list[Chunk],
                vector_store: VectorStore, lexical_index: LexicalIndex) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "tree.json"), "w") as f:
        f.write(tree.model_dump_json())

    with open(os.path.join(out_dir, "chunks.json"), "w") as f:
        json.dump([c.model_dump() for c in chunks], f)

    faiss.write_index(vector_store.index, os.path.join(out_dir, "vector.index"))
    with open(os.path.join(out_dir, "embedder.pkl"), "wb") as f:
        pickle.dump(vector_store.embedder, f)
    with open(os.path.join(out_dir, "vector_chunk_ids.json"), "w") as f:
        json.dump(vector_store.chunk_ids, f)

    with open(os.path.join(out_dir, "lexical.pkl"), "wb") as f:
        pickle.dump({"bm25": lexical_index.bm25, "chunk_ids": lexical_index.chunk_ids}, f)


def load_index(out_dir: str) -> tuple[DocumentTree, list[Chunk], dict[str, Chunk], VectorStore, LexicalIndex]:
    with open(os.path.join(out_dir, "tree.json")) as f:
        tree = DocumentTree.model_validate_json(f.read())

    with open(os.path.join(out_dir, "chunks.json")) as f:
        chunks = [Chunk.model_validate(d) for d in json.load(f)]
    chunks_by_id = {c.chunk_id: c for c in chunks}

    with open(os.path.join(out_dir, "embedder.pkl"), "rb") as f:
        embedder = pickle.load(f)
    vector_store = VectorStore(embedder)
    vector_store.index = faiss.read_index(os.path.join(out_dir, "vector.index"))
    with open(os.path.join(out_dir, "vector_chunk_ids.json")) as f:
        vector_store.chunk_ids = json.load(f)

    with open(os.path.join(out_dir, "lexical.pkl"), "rb") as f:
        lex_data = pickle.load(f)
    lexical_index = LexicalIndex()
    lexical_index.bm25 = lex_data["bm25"]
    lexical_index.chunk_ids = lex_data["chunk_ids"]

    return tree, chunks, chunks_by_id, vector_store, lexical_index


def index_exists(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "tree.json"))
