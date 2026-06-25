"""AgentState + shared types. LangGraph-shaped TypedDict so nodes can be
imported into a StateGraph later without changes."""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

Domain = Literal["safety", "maintenance", "quality", "none"]


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    doc_title: str
    domain: str
    section_id: str
    heading: str
    body: str
    dense_score: float
    sparse_score: float
    rrf_score: float


class JudgeVerdict(TypedDict):
    grounded: bool
    routing_ok: bool
    score: float
    reasons: list[str]
    model: str


class AgentState(TypedDict, total=False):
    # input
    trace_id: str
    query: str
    session_id: Optional[str]

    # classify
    route: Domain
    route_confidence: float
    route_reason: str

    # retrieve
    retrieved: list[RetrievedChunk]
    retrieval_confidence: float
    dense_top: list[dict]
    sparse_top: list[dict]

    # guard
    refused: bool
    refusal_reason: Optional[str]

    # generate
    answer: str
    citations: list[str]
    generation_meta: dict[str, Any]

    # judge
    judge: Optional[JudgeVerdict]

    # latency
    latency_ms: dict[str, int]
