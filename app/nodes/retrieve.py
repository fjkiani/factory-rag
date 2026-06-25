"""Retrieve node: BM25 sparse + dense + RRF fusion against the routed collection only.

Strict collection isolation per the planning session: classify -> one collection -> retrieve.
"""
from __future__ import annotations

import time
from typing import Callable

from ..adapters.embeddings import EmbeddingClient
from ..adapters.vectordb import Hit, VectorStore
from ..state import AgentState, RetrievedChunk


def _domain_collection(domain: str) -> str:
    return f"kb_{domain}"


def _rrf_fuse(
    dense_hits: list[Hit], sparse_hits: list[Hit], *, k: int = 60, top_n: int = 5
) -> list[tuple[str, float, dict, float, float]]:
    """Reciprocal Rank Fusion. Returns list of (chunk_id, rrf_score, payload, dense_score, sparse_score)."""
    rrf: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    for rank, h in enumerate(dense_hits, start=1):
        rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
        payloads[h.chunk_id] = h.payload
        dense_scores[h.chunk_id] = max(dense_scores.get(h.chunk_id, -1e9), h.score)
    for rank, h in enumerate(sparse_hits, start=1):
        rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
        payloads.setdefault(h.chunk_id, h.payload)
        sparse_scores[h.chunk_id] = max(sparse_scores.get(h.chunk_id, -1e9), h.score)
    ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_n]
    out = []
    for cid, score in ranked:
        out.append(
            (
                cid,
                float(score),
                payloads.get(cid, {}),
                float(dense_scores.get(cid, 0.0)),
                float(sparse_scores.get(cid, 0.0)),
            )
        )
    return out


def make_retrieve_node(
    store: VectorStore,
    embedder: EmbeddingClient,
    *,
    top_k_each: int = 20,
    top_n_fused: int = 5,
    rrf_k: int = 60,
) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        # If classify already refused (route none / low conf), skip retrieval cleanly.
        route = state.get("route", "none")
        if route == "none":
            state["retrieved"] = []
            state["retrieval_confidence"] = 0.0
            state["dense_top"] = []
            state["sparse_top"] = []
            state.setdefault("latency_ms", {})["retrieve"] = 0
            return state

        collection = _domain_collection(route)
        # Dense
        qvec = embedder.embed([state["query"]])[0]
        dense_hits = store.search_dense(collection, qvec, top_k=top_k_each)
        # Sparse
        sparse_hits = store.search_sparse(collection, state["query"], top_k=top_k_each)
        # Fuse
        fused = _rrf_fuse(dense_hits, sparse_hits, k=rrf_k, top_n=top_n_fused)

        retrieved: list[RetrievedChunk] = []
        for cid, rrf_score, payload, dscore, sscore in fused:
            retrieved.append(
                {
                    "chunk_id": cid,
                    "doc_id": payload.get("doc_id", ""),
                    "doc_title": payload.get("doc_title", ""),
                    "domain": payload.get("domain", route),
                    "section_id": payload.get("section_id", ""),
                    "heading": payload.get("heading", ""),
                    "body": payload.get("body", ""),
                    "dense_score": dscore,
                    "sparse_score": sscore,
                    "rrf_score": rrf_score,
                }
            )

        # Confidence: normalize top RRF score against a soft ceiling 2/(k+1).
        # For k=60 the max single-list RRF is ~1/61=0.0164; fused max ~2*0.0164=0.0328.
        max_rrf = retrieved[0]["rrf_score"] if retrieved else 0.0
        ceiling = 2.0 / (rrf_k + 1)
        confidence = min(1.0, max_rrf / ceiling) if ceiling > 0 else 0.0

        state["retrieved"] = retrieved
        state["retrieval_confidence"] = float(confidence)
        state["dense_top"] = [{"chunk_id": h.chunk_id, "score": float(h.score)} for h in dense_hits[:5]]
        state["sparse_top"] = [{"chunk_id": h.chunk_id, "score": float(h.score)} for h in sparse_hits[:5]]
        state.setdefault("latency_ms", {})["retrieve"] = int((time.perf_counter() - t0) * 1000)
        return state

    return retrieve
