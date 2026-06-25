"""Shared fakes for tests. No network. No real LLM. Deterministic embedder."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from app.adapters.embeddings import HashingEmbedder
from app.adapters.llm import LLMResponse
from app.adapters.vectordb import NumpyVectorStore, Point
from app.corpus.ingest import DEFAULT_CORPUS


@dataclass
class FakeLLM:
    """LLM stub that maps (system_marker, user_substring) to canned outputs.
    Tests register handlers; default raises so tests are loud about misses.
    """
    handlers: list = field(default_factory=list)  # list[(predicate, response_factory)]
    calls: list[dict] = field(default_factory=list)

    def register(self, predicate, response):
        self.handlers.append((predicate, response))

    def complete(self, system, user, *, temperature=0.0, max_tokens=800, model=None, response_format_json=False):
        self.calls.append({"system": system, "user": user, "model": model})
        for predicate, response in self.handlers:
            if predicate(system, user):
                text = response(system, user) if callable(response) else response
                return LLMResponse(text=text, model=model or "fake", prompt_tokens=10, completion_tokens=20, raw={})
        raise AssertionError(f"No FakeLLM handler matched. system={system[:80]!r} user={user[:120]!r}")


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def ingested_store(tmp_data_dir: Path, hashing_embedder: HashingEmbedder) -> NumpyVectorStore:
    """A NumpyVectorStore populated from the real seed corpus using the
    deterministic hashing embedder. Used by integration-style tests."""
    store = NumpyVectorStore(tmp_data_dir / "index.npz")
    with open(DEFAULT_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    texts, keys = [], []
    for doc in corpus["documents"]:
        for sec in doc["sections"]:
            chunk_id = f"{doc['doc_id']}#{sec['section_id']}"
            payload = {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "domain": doc["domain"],
                "section_id": sec["section_id"],
                "heading": sec["heading"],
                "body": sec["body"],
            }
            texts.append(f"{sec['heading']}\n\n{sec['body']}")
            keys.append((f"kb_{doc['domain']}", chunk_id, payload))
    vecs = hashing_embedder.embed(texts)
    grouped: dict[str, list[Point]] = {}
    for (collection, cid, payload), v in zip(keys, vecs):
        grouped.setdefault(collection, []).append(Point(chunk_id=cid, vector=v, payload=payload))
    for collection, points in grouped.items():
        store.upsert(collection, points)
    return store
