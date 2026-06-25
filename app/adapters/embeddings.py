"""Embedding adapter over OpenRouter (default: Nemotron Embed VL 1B V2).

Includes a deterministic, dependency-free `HashingEmbedder` for offline tests
and CI. The dim is fixed so vectors are comparable across runs.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Protocol

import httpx
import numpy as np


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class OpenRouterEmbeddings:
    """Calls the OpenRouter `/embeddings` endpoint. Some free embedding models
    are not exposed there yet; on 404/400 we surface a clear error so the
    caller can swap models via env."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float = 60.0, expected_dim: int = 1024):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._dim = expected_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://rag-mvp.local",
            "X-Title": "rag-mvp",
        }
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter embeddings failed {resp.status_code}: {resp.text[:300]} "
                f"(model={self.model}). Swap EMBED_MODEL or set EMBED_BACKEND=hashing for offline use."
            )
        data = resp.json()
        vecs = [row["embedding"] for row in data["data"]]
        if vecs:
            self._dim = len(vecs[0])
        return vecs


class HashingEmbedder:
    """Deterministic offline embedder. Token hash -> bucket counts -> L2 unit.

    Not a semantic embedder, but stable, free, and good enough to exercise
    the full pipeline (incl. BM25 + dense + RRF) deterministically in tests
    and local demos without network access.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (1, 2)):
        self._dim = dim
        self._ng = ngram_range

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            tokens = [tok.lower() for tok in _TOKEN_RE.findall(t)]
            v = np.zeros(self._dim, dtype=np.float32)
            lo, hi = self._ng
            for n in range(lo, hi + 1):
                for i in range(0, len(tokens) - n + 1):
                    gram = " ".join(tokens[i : i + n])
                    h = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16)
                    idx = h % self._dim
                    sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
                    v[idx] += sign
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out.append(v.tolist())
        return out


def get_embedder(
    backend: str, *, api_key: str, base_url: str, model: str, hashing_dim: int = 256
) -> EmbeddingClient:
    backend = (backend or "openrouter").lower()
    if backend == "hashing":
        return HashingEmbedder(dim=hashing_dim)
    return OpenRouterEmbeddings(api_key=api_key, base_url=base_url, model=model)
