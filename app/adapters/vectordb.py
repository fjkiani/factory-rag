"""VectorStore protocol + two backends:
- NumpyVectorStore (MVP default): zero-dependency, no native compilation,
  no external DB. Embeddings live as float32 arrays in-memory + on-disk
  via np.savez. Sparse search is BM25 (rank_bm25) per collection.
- QdrantVectorStore: placeholder that raises NotImplementedError. Same
  protocol surface so we can swap by flipping VECTOR_BACKEND.

Strict collection isolation: each domain is its own collection.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class Point:
    chunk_id: str
    vector: list[float]
    payload: dict


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    def upsert(self, collection: str, points: list[Point]) -> None: ...
    def search_dense(self, collection: str, vector: list[float], top_k: int) -> list[Hit]: ...
    def search_sparse(self, collection: str, query: str, top_k: int) -> list[Hit]: ...
    def has_collection(self, collection: str) -> bool: ...
    def collection_size(self, collection: str) -> int: ...
    def persist(self) -> None: ...
    def load(self) -> None: ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@dataclass
class _Collection:
    chunk_ids: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None  # (N, D) float32
    payloads: list[dict] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)


class NumpyVectorStore:
    """In-memory dense (cosine via L2-normalized dot product) + BM25 sparse.

    Persistence: a single pickle file at `path` (atomic write). Vectors
    stored as float32 numpy arrays inside the pickle. No external engines.
    """

    def __init__(self, path: Path):
        self.path = path
        self._cols: dict[str, _Collection] = {}

    # ---- protocol --------------------------------------------------------
    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._cols.setdefault(collection, _Collection())
        # Build/extend matrix
        new_vecs = np.asarray([p.vector for p in points], dtype=np.float32)
        new_vecs = _l2_normalize(new_vecs)
        if col.vectors is None or col.vectors.size == 0:
            col.vectors = new_vecs
            col.chunk_ids = [p.chunk_id for p in points]
            col.payloads = [p.payload for p in points]
            col.tokenized_corpus = [_tokenize(p.payload.get("body", "")) for p in points]
        else:
            col.vectors = np.vstack([col.vectors, new_vecs]).astype(np.float32)
            col.chunk_ids.extend([p.chunk_id for p in points])
            col.payloads.extend([p.payload for p in points])
            col.tokenized_corpus.extend([_tokenize(p.payload.get("body", "")) for p in points])
        # Rebuild BM25 (cheap for small corpora)
        col.bm25 = BM25Okapi(col.tokenized_corpus)

    def search_dense(self, collection: str, vector: list[float], top_k: int) -> list[Hit]:
        col = self._cols.get(collection)
        if col is None or col.vectors is None or col.vectors.size == 0:
            return []
        q = _l2_normalize(np.asarray([vector], dtype=np.float32))[0]
        sims = col.vectors @ q  # cosine since both sides are L2-normalized
        idx = np.argsort(-sims)[:top_k]
        return [
            Hit(chunk_id=col.chunk_ids[i], score=float(sims[i]), payload=col.payloads[i])
            for i in idx
        ]

    def search_sparse(self, collection: str, query: str, top_k: int) -> list[Hit]:
        col = self._cols.get(collection)
        if col is None or col.bm25 is None or not col.chunk_ids:
            return []
        scores = col.bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:top_k]
        return [
            Hit(chunk_id=col.chunk_ids[i], score=float(scores[i]), payload=col.payloads[i])
            for i in idx
        ]

    def has_collection(self, collection: str) -> bool:
        col = self._cols.get(collection)
        return col is not None and col.vectors is not None and col.vectors.size > 0

    def collection_size(self, collection: str) -> int:
        col = self._cols.get(collection)
        if col is None or col.vectors is None:
            return 0
        return int(col.vectors.shape[0])

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    name: {
                        "chunk_ids": c.chunk_ids,
                        "vectors": c.vectors,
                        "payloads": c.payloads,
                        "tokenized_corpus": c.tokenized_corpus,
                    }
                    for name, c in self._cols.items()
                },
                f,
            )
        tmp.replace(self.path)

    def load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self._cols = {}
        for name, d in data.items():
            col = _Collection(
                chunk_ids=d["chunk_ids"],
                vectors=d["vectors"],
                payloads=d["payloads"],
                tokenized_corpus=d["tokenized_corpus"],
            )
            if col.tokenized_corpus:
                col.bm25 = BM25Okapi(col.tokenized_corpus)
            self._cols[name] = col


class QdrantVectorStore:
    """Placeholder. Same protocol so we can flip VECTOR_BACKEND=qdrant later
    without touching the pipeline."""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    def upsert(self, collection, points):  # pragma: no cover
        raise NotImplementedError("Qdrant backend is a placeholder for the MVP")

    def search_dense(self, collection, vector, top_k):  # pragma: no cover
        raise NotImplementedError("Qdrant backend is a placeholder for the MVP")

    def search_sparse(self, collection, query, top_k):  # pragma: no cover
        raise NotImplementedError("Qdrant backend is a placeholder for the MVP")

    def has_collection(self, collection):  # pragma: no cover
        return False

    def collection_size(self, collection):  # pragma: no cover
        return 0

    def persist(self):  # pragma: no cover
        return None

    def load(self):  # pragma: no cover
        return None


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Returns the same dtype (float32 preferred)."""
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x if n == 0 else (x / n).astype(x.dtype)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(x.dtype)


def get_vector_store(backend: str, *, numpy_path: Path, qdrant_url: str = "", qdrant_api_key: str = "") -> VectorStore:
    if backend == "numpy":
        store = NumpyVectorStore(numpy_path)
        store.load()
        return store
    if backend == "qdrant":
        return QdrantVectorStore(qdrant_url, qdrant_api_key)
    raise ValueError(f"Unknown VECTOR_BACKEND={backend!r}; expected 'numpy' or 'qdrant'")
