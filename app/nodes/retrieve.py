"""Retrieve node: BM25 sparse + dense + RRF fusion.

Multi-route fanout:
- Reads `state["route_candidates"]` (ranked list).
- Queries the top-N domains (default top 2) whose confidence clears
  `min_secondary_conf`. Each collection contributes its own dense + sparse
  hits. We RRF-fuse across ALL of them so a single ranked list goes to
  generate. Each retrieved chunk carries its `domain` so generate can
  show the user which doc each citation came from.
- If only the primary route applies (single-domain question), this
  collapses to the original behavior.
- If route is 'none' or no candidates, skip retrieval cleanly.
"""
from __future__ import annotations

import time
from typing import Callable

from ..adapters.embeddings import EmbeddingClient
from ..adapters.vectordb import Hit, VectorStore
from ..state import AgentState, RetrievedChunk


def _domain_collection(domain: str) -> str:
    return f"kb_{domain}"


def _rrf_fuse_multi(
    per_collection_hits: dict[str, tuple[list[Hit], list[Hit]]],
    *,
    k: int = 60,
    top_n: int = 5,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """RRF-fuse dense+sparse hits ACROSS multiple collections.

    Returns:
        fused: ordered list of dicts ready to become RetrievedChunk
        per_collection_diag: {collection: [{chunk_id, rrf}]} for telemetry
    """
    rrf: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    home_collection: dict[str, str] = {}

    for collection, (dense_hits, sparse_hits) in per_collection_hits.items():
        for rank, h in enumerate(dense_hits, start=1):
            rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            payloads[h.chunk_id] = h.payload
            dense_scores[h.chunk_id] = max(dense_scores.get(h.chunk_id, -1e9), h.score)
            home_collection.setdefault(h.chunk_id, collection)
        for rank, h in enumerate(sparse_hits, start=1):
            rrf[h.chunk_id] = rrf.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(h.chunk_id, h.payload)
            sparse_scores[h.chunk_id] = max(sparse_scores.get(h.chunk_id, -1e9), h.score)
            home_collection.setdefault(h.chunk_id, collection)

    ranked = sorted(rrf.items(), key=lambda kv: -kv[1])
    fused: list[dict] = []
    per_collection_diag: dict[str, list[dict]] = {c: [] for c in per_collection_hits}
    for cid, score in ranked[:top_n]:
        payload = payloads.get(cid, {})
        fused.append(
            {
                "chunk_id": cid,
                "payload": payload,
                "dense_score": float(dense_scores.get(cid, 0.0)),
                "sparse_score": float(sparse_scores.get(cid, 0.0)),
                "rrf_score": float(score),
                "collection": home_collection.get(cid, ""),
            }
        )
    # Per-collection diagnostics: take each collection's local top hits
    for collection, (dense_hits, sparse_hits) in per_collection_hits.items():
        seen: set[str] = set()
        for h in dense_hits[:5] + sparse_hits[:5]:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            per_collection_diag[collection].append(
                {"chunk_id": h.chunk_id, "rrf": rrf.get(h.chunk_id, 0.0)}
            )
    return fused, per_collection_diag


def make_retrieve_node(
    store: VectorStore,
    embedder: EmbeddingClient,
    *,
    top_k_each: int = 20,
    top_n_fused: int = 5,
    rrf_k: int = 60,
    max_collections: int = 2,
    min_secondary_conf: float = 0.30,
) -> Callable[[AgentState], AgentState]:
    """Build retrieve node.

    Args:
        max_collections: maximum number of domains to query per request.
            Default 2 = primary + at most one alternate.
        min_secondary_conf: alternates with confidence below this are dropped
            (we only fan out when a secondary domain has real signal).
    """

    def retrieve(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        candidates = state.get("route_candidates") or []
        # Back-compat: if state has `route` but no `route_candidates` (e.g. a
        # caller wired classify -> retrieve directly with the old shape, or a
        # test injects route by hand), synthesize a single-candidate list.
        if not candidates and state.get("route") in {"safety", "maintenance", "quality"}:
            candidates = [
                {
                    "route": state["route"],
                    "confidence": float(state.get("route_confidence", 1.0)),
                    "reason": state.get("route_reason", ""),
                    "source": state.get("route_source", "compat"),
                }
            ]
        # Filter to real domains, deduped, ordered by confidence
        seen: set[str] = set()
        domains: list[tuple[str, float]] = []
        for c in candidates:
            r = c.get("route")
            if r in {"safety", "maintenance", "quality"} and r not in seen:
                seen.add(r)
                domains.append((r, float(c.get("confidence", 0.0))))

        # Keep primary always; keep alternates only if they clear the secondary threshold.
        if not domains:
            state["retrieved"] = []
            state["retrieval_confidence"] = 0.0
            state["dense_top"] = []
            state["sparse_top"] = []
            state["retrieved_per_collection"] = {}
            state.setdefault("latency_ms", {})["retrieve"] = 0
            return state

        primary_route, primary_conf = domains[0]
        keep = [(primary_route, primary_conf)]
        for r, c in domains[1:]:
            if c >= min_secondary_conf and len(keep) < max_collections:
                keep.append((r, c))

        # Embed the query ONCE; reuse across collections.
        qvec = embedder.embed([state["query"]])[0]

        per_collection_hits: dict[str, tuple[list[Hit], list[Hit]]] = {}
        for route, _ in keep:
            collection = _domain_collection(route)
            dense_hits = store.search_dense(collection, qvec, top_k=top_k_each)
            sparse_hits = store.search_sparse(collection, state["query"], top_k=top_k_each)
            per_collection_hits[collection] = (dense_hits, sparse_hits)

        fused, per_collection_diag = _rrf_fuse_multi(
            per_collection_hits, k=rrf_k, top_n=top_n_fused
        )

        retrieved: list[RetrievedChunk] = []
        for f in fused:
            p = f["payload"]
            retrieved.append(
                {
                    "chunk_id": f["chunk_id"],
                    "doc_id": p.get("doc_id", ""),
                    "doc_title": p.get("doc_title", ""),
                    "domain": p.get("domain", ""),
                    "section_id": p.get("section_id", ""),
                    "heading": p.get("heading", ""),
                    "body": p.get("body", ""),
                    "dense_score": f["dense_score"],
                    "sparse_score": f["sparse_score"],
                    "rrf_score": f["rrf_score"],
                }
            )

        max_rrf = retrieved[0]["rrf_score"] if retrieved else 0.0
        # For multi-collection, ceiling is N*2/(k+1) where N=number of collections
        ceiling = (2.0 * len(keep)) / (rrf_k + 1)
        confidence = min(1.0, max_rrf / ceiling) if ceiling > 0 else 0.0

        # Flatten per-collection top hits for the existing dense_top/sparse_top fields.
        dense_top: list[dict] = []
        sparse_top: list[dict] = []
        for collection, (dh, sh) in per_collection_hits.items():
            for h in dh[:5]:
                dense_top.append({"chunk_id": h.chunk_id, "score": float(h.score), "collection": collection})
            for h in sh[:5]:
                sparse_top.append({"chunk_id": h.chunk_id, "score": float(h.score), "collection": collection})

        state["retrieved"] = retrieved
        state["retrieval_confidence"] = float(confidence)
        state["dense_top"] = dense_top
        state["sparse_top"] = sparse_top
        state["retrieved_per_collection"] = per_collection_diag
        state.setdefault("latency_ms", {})["retrieve"] = int((time.perf_counter() - t0) * 1000)
        return state

    return retrieve
