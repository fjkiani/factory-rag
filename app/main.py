"""FastAPI app. Swagger /docs is the demo surface."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .adapters.embeddings import get_embedder
from .adapters.llm import OpenRouterLLM
from .adapters.vectordb import get_vector_store
from .config import load_config
from .corpus.ingest import ingest_path
from .nodes.classify import make_classify_node
from .nodes.generate import make_generate_node
from .nodes.guard import make_guard_node
from .nodes.judge import make_judge_node
from .nodes.retrieve import make_retrieve_node
from .nodes.telemetry import make_telemetry_node
from .pipeline import make_pipeline

cfg = load_config()

app = FastAPI(
    title="Manufacturing Floor RAG MVP",
    description=(
        "Floor-supervisor RAG over Safety / Maintenance / Quality docs. "
        "Strict cite-or-refuse. Online + offline LLM-as-judge. JSONL telemetry. "
        "Zero-dependency NumPy vector store (Qdrant placeholder kept for future swap)."
    ),
    version="0.1.0",
)


# ---- Pydantic models -------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_heading: str
    score: float


class JudgeVerdictModel(BaseModel):
    grounded: bool
    routing_ok: bool
    score: float
    reasons: list[str]
    model: str


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    route: Literal["safety", "maintenance", "quality", "none"]
    route_confidence: float
    retrieval_confidence: float
    citations: list[Citation]
    judge: Optional[JudgeVerdictModel] = None
    latency_ms: dict[str, int]


# ---- Wiring ----------------------------------------------------------------
def _build_llm() -> OpenRouterLLM:
    return OpenRouterLLM(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        default_model=cfg.llm_model,
    )


def _build_pipeline_runner():
    llm = _build_llm()
    embed_backend = os.getenv("EMBED_BACKEND", "openrouter")
    embedder = get_embedder(
        embed_backend,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        model=cfg.embed_model,
    )
    store = get_vector_store(
        cfg.vector_backend,
        numpy_path=cfg.index_path,
        qdrant_url=cfg.qdrant_url,
        qdrant_api_key=cfg.qdrant_api_key,
    )
    nodes = [
        make_classify_node(llm, model=cfg.llm_model),
        make_retrieve_node(store, embedder),
        make_guard_node(
            route_conf_threshold=cfg.route_conf_threshold,
            retrieval_conf_threshold=cfg.retrieval_conf_threshold,
        ),
        make_generate_node(llm, model=cfg.llm_model),
        make_judge_node(llm, model=cfg.judge_model),
        make_telemetry_node(cfg.telemetry_path),
    ]
    return make_pipeline(nodes), store


_pipeline_runner = None
_store = None


def get_pipeline():
    global _pipeline_runner, _store
    if _pipeline_runner is None:
        _pipeline_runner, _store = _build_pipeline_runner()
    return _pipeline_runner


def get_store():
    if _store is None:
        get_pipeline()
    return _store


# ---- Admin auth ------------------------------------------------------------
bearer = HTTPBearer(auto_error=False)


def require_admin(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if not creds or creds.credentials != cfg.admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad admin token")


# ---- Routes ----------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    store = get_store()
    return {
        "ok": True,
        "vector_backend": cfg.vector_backend,
        "collections": {
            "kb_safety": store.collection_size("kb_safety"),
            "kb_maintenance": store.collection_size("kb_maintenance"),
            "kb_quality": store.collection_size("kb_quality"),
        },
        "llm_model": cfg.llm_model,
        "judge_model": cfg.judge_model,
        "embed_model": cfg.embed_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    run = get_pipeline()
    state = run(req.query, session_id=req.session_id)

    citations: list[Citation] = []
    # Build citation list from retrieved chunks that were actually cited.
    retrieved_by_id = {r["chunk_id"]: r for r in (state.get("retrieved") or [])}
    for cid in state.get("citations") or []:
        r = retrieved_by_id.get(cid)
        if not r:
            continue
        citations.append(
            Citation(
                chunk_id=cid,
                doc_id=r["doc_id"],
                doc_title=r["doc_title"],
                section_heading=r["heading"],
                score=r["rrf_score"],
            )
        )

    judge = state.get("judge")
    return ChatResponse(
        trace_id=state["trace_id"],
        answer=state.get("answer", ""),
        refused=bool(state.get("refused")),
        refusal_reason=state.get("refusal_reason"),
        route=state.get("route", "none"),
        route_confidence=float(state.get("route_confidence", 0.0)),
        retrieval_confidence=float(state.get("retrieval_confidence", 0.0)),
        citations=citations,
        judge=JudgeVerdictModel(**judge) if judge else None,
        latency_ms=state.get("latency_ms", {}),
    )


@app.get("/telemetry")
def telemetry(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(500, limit))
    p = cfg.telemetry_path
    if not p.exists():
        return {"count": 0, "records": []}
    # Tail without loading the whole file.
    lines: list[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line)
            if len(lines) > limit:
                lines.pop(0)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return {"count": len(records), "records": records}


@app.get("/eval/scorecard")
def get_scorecard() -> dict[str, Any]:
    p = cfg.scorecard_path
    if not p.exists():
        raise HTTPException(status_code=404, detail="scorecard not generated yet")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/admin/ingest", dependencies=[Depends(require_admin)])
def admin_ingest() -> dict[str, Any]:
    out = ingest_path()
    # Reset store cache so the new index loads.
    global _pipeline_runner, _store
    _pipeline_runner = None
    _store = None
    return out


@app.post("/admin/eval", dependencies=[Depends(require_admin)])
def admin_eval() -> dict[str, Any]:
    from .eval.run_offline import run_offline_eval

    return run_offline_eval()
