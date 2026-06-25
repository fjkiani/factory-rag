"""Ingest the seed JSON corpus into the NumPy vector store.

One chunk per section. Chunk id is `DOC-ID#section_id`. Per-domain collections.
Idempotent: re-running with the same corpus rebuilds the index in place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..adapters.embeddings import get_embedder
from ..adapters.vectordb import NumpyVectorStore, Point, get_vector_store
from ..config import load_config

DEFAULT_CORPUS = Path(__file__).resolve().parent / "seed_docs.json"


def ingest_path(corpus_path: Path | None = None) -> dict:
    cfg = load_config()
    corpus_path = corpus_path or DEFAULT_CORPUS
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    embed_backend = os.getenv("EMBED_BACKEND", "openrouter")
    embedder = get_embedder(
        embed_backend,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        model=cfg.embed_model,
    )

    # Build a fresh NumpyVectorStore so ingest is idempotent.
    store = NumpyVectorStore(cfg.index_path)

    grouped: dict[str, list[Point]] = {}
    counts = {"safety": 0, "maintenance": 0, "quality": 0}
    texts: list[str] = []
    keys: list[tuple[str, str, dict]] = []
    for doc in corpus["documents"]:
        domain = doc["domain"]
        for sec in doc["sections"]:
            chunk_id = f"{doc['doc_id']}#{sec['section_id']}"
            text = f"{sec['heading']}\n\n{sec['body']}"
            payload = {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "domain": domain,
                "section_id": sec["section_id"],
                "heading": sec["heading"],
                "body": sec["body"],
            }
            keys.append((f"kb_{domain}", chunk_id, payload))
            texts.append(text)
            counts[domain] = counts.get(domain, 0) + 1

    vectors = embedder.embed(texts)
    for (collection, chunk_id, payload), vec in zip(keys, vectors):
        grouped.setdefault(collection, []).append(Point(chunk_id=chunk_id, vector=vec, payload=payload))

    for collection, points in grouped.items():
        store.upsert(collection, points)
    store.persist()
    return {
        "index_path": str(cfg.index_path),
        "counts": counts,
        "collections": {c: store.collection_size(c) for c in grouped},
        "embed_model": cfg.embed_model if embed_backend == "openrouter" else f"hashing:{embedder.dim}",
    }


if __name__ == "__main__":
    out = ingest_path()
    print(json.dumps(out, indent=2))
