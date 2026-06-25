"""FastAPI app. Swagger /docs is the demo surface; HTML chat UI at /."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, ValidationInfo

from .adapters.embeddings import get_embedder
from .adapters.llm import GroqLLM, LLMClient, OpenRouterLLM, ProviderError
from .adapters.providers import GLOBAL_HEALTH, MultiProviderLLM, ProviderSlot
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
from .providers_catalog import CATALOG, catalog_dict, is_known_choice

cfg = load_config()


def _cfg():
    """Re-read config on demand so env-var changes mid-test (or mid-deploy)
    take effect for routes that genuinely need the latest values. We keep
    the module-level `cfg` for handlers that ran fine with cached values
    (e.g., file paths)."""
    return load_config()


app = FastAPI(
    title="factory-rag · Manufacturing Floor RAG",
    description=(
        "Floor-supervisor RAG over Safety / Maintenance / Quality docs. "
        "Strict cite-or-refuse. Multi-provider (OpenRouter + Groq) with model "
        "selection and automatic failover. Online + offline LLM-as-judge. "
        "JSONL telemetry."
    ),
    version="0.2.0",
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---- Pydantic models -------------------------------------------------------
class ModelChoiceModel(BaseModel):
    provider: Literal["openrouter", "groq"]
    model: str = Field(..., min_length=1, max_length=200)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    llm: Optional[ModelChoiceModel] = None
    judge: Optional[ModelChoiceModel] = None

    @field_validator("llm", "judge")
    @classmethod
    def _validate_choice(cls, v, info: ValidationInfo):  # type: ignore[no-redef]
        if v is None:
            return v
        role = "answer" if info.field_name == "llm" else "judge"
        if not is_known_choice(v.provider, v.model, role):  # type: ignore[arg-type]
            raise ValueError(
                f"{role} model {v.provider}/{v.model} not in catalog. "
                f"Call GET /providers/catalog for the supported list."
            )
        return v


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_heading: str
    score: float
    body_preview: str = ""


class JudgeVerdictModel(BaseModel):
    grounded: bool
    routing_ok: bool
    score: float
    reasons: list[str]
    model: str
    provider: str = "unknown"
    provider_fallback_used: bool = False
    fallback_chain: list[str] = []
    judge_errored: bool = False


class RouteCandidate(BaseModel):
    route: str
    confidence: float
    reason: str
    source: str


class LlmUsage(BaseModel):
    provider: str
    model: str
    provider_fallback_used: bool
    fallback_chain: list[str]


class Warning(BaseModel):
    type: Literal[
        "provider_fallback",
        "judge_errored",
        "provider_degraded",
        "no_route",
    ]
    severity: Literal["info", "warning", "error"]
    message: str
    detail: dict[str, Any] = {}


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    route: Literal["safety", "maintenance", "quality", "none"]
    route_confidence: float
    route_source: str
    route_candidates: list[RouteCandidate]
    route_used_fallback: bool
    collections_queried: list[str]
    retrieval_confidence: float
    citations: list[Citation]
    judge: Optional[JudgeVerdictModel] = None
    llm_used: Optional[LlmUsage] = None
    warnings: list[Warning] = []
    latency_ms: dict[str, int]


# ---- Wiring ----------------------------------------------------------------
def _build_llm(c=None) -> LLMClient:
    """Build the LLM client. Always a MultiProviderLLM, even when only one
    provider is configured — keeps the call path identical and lets the
    health monitor track per-(provider,model) usage."""
    c = c or _cfg()
    slots: list[ProviderSlot] = []
    if c.openrouter_api_key:
        slots.append(
            ProviderSlot(
                OpenRouterLLM(
                    api_key=c.openrouter_api_key,
                    base_url=c.openrouter_base_url,
                    default_model=c.llm_model,
                    http_referer=c.http_referer,
                    app_title=c.app_title,
                ),
                default_model=c.llm_model,
            )
        )
    if c.groq_api_key:
        slots.append(
            ProviderSlot(
                GroqLLM(
                    api_key=c.groq_api_key,
                    base_url=c.groq_base_url,
                    default_model=c.groq_model,
                ),
                default_model=c.groq_model,
            )
        )
    if not slots:
        raise RuntimeError(
            "No LLM provider configured. Set OPENROUTER_API_KEY and/or GROQ_API_KEY."
        )
    return MultiProviderLLM(slots, health=GLOBAL_HEALTH)


def _build_pipeline_runner():
    c = _cfg()
    llm = _build_llm(c)
    embed_backend = c.embed_backend
    embedder = get_embedder(
        embed_backend,
        api_key=c.openrouter_api_key,
        base_url=c.openrouter_base_url,
        model=c.embed_model,
    )
    store = get_vector_store(
        c.vector_backend,
        numpy_path=c.index_path,
        qdrant_url=c.qdrant_url,
        qdrant_api_key=c.qdrant_api_key,
    )
    nodes = [
        make_classify_node(llm, model=c.llm_model),
        make_retrieve_node(store, embedder),
        make_guard_node(
            route_conf_threshold=c.route_conf_threshold,
            retrieval_conf_threshold=c.retrieval_conf_threshold,
        ),
        make_generate_node(llm, model=c.llm_model),
        make_judge_node(llm, model=c.judge_model),
        make_telemetry_node(c.telemetry_path),
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
    if not creds or creds.credentials != _cfg().admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad admin token")


# ---- Routes ----------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    c = _cfg()
    store = get_store()
    return {
        "ok": True,
        "vector_backend": c.vector_backend,
        "collections": {
            "kb_safety": store.collection_size("kb_safety"),
            "kb_maintenance": store.collection_size("kb_maintenance"),
            "kb_quality": store.collection_size("kb_quality"),
        },
        "llm_default_model": c.llm_model,
        "judge_default_model": c.judge_model,
        "embed_model": c.embed_model,
        "providers_configured": [
            p for p in ["openrouter", "groq"]
            if (p == "openrouter" and c.openrouter_api_key)
            or (p == "groq" and c.groq_api_key)
        ],
    }


@app.get("/providers/catalog")
def providers_catalog() -> dict[str, Any]:
    """Validated list of (provider, model, roles) the UI can offer."""
    c = _cfg()
    out = catalog_dict()
    configured = {
        "openrouter": bool(c.openrouter_api_key),
        "groq": bool(c.groq_api_key),
    }
    out["entries"] = [
        e for e in out["entries"] if configured.get(e["provider"], False)
    ]
    out["defaults"] = {
        "answer": {"provider": "openrouter", "model": c.llm_model}
        if configured["openrouter"]
        else {"provider": "groq", "model": c.groq_model},
        "judge": {"provider": "openrouter", "model": c.judge_model}
        if configured["openrouter"]
        else {"provider": "groq", "model": c.groq_model},
    }
    return out


@app.get("/providers/health")
def providers_health() -> dict[str, Any]:
    return GLOBAL_HEALTH.snapshot()


def _build_warnings(
    *,
    llm_used: Optional[LlmUsage],
    judge: Optional[dict],
    refused: bool,
    refusal_reason: Optional[str],
    route: str,
    route_used_fallback: bool,
) -> list[Warning]:
    """Backend owns warning policy; UI just renders these. One responsibility
    per warning type — don't pile multiple concerns into one message."""
    out: list[Warning] = []

    # Primary LLM provider failed during this request — served by a fallback.
    if llm_used and llm_used.provider_fallback_used:
        primary = llm_used.fallback_chain[0] if llm_used.fallback_chain else "primary"
        out.append(Warning(
            type="provider_fallback",
            severity="warning",
            message=(
                f"Primary provider '{primary}' was unavailable. "
                f"Served by {llm_used.provider}/{llm_used.model}."
            ),
            detail={
                "primary": primary,
                "served_by": {"provider": llm_used.provider, "model": llm_used.model},
                "chain": llm_used.fallback_chain,
            },
        ))

    # Judge couldn't render a verdict.
    if judge and judge.get("judge_errored"):
        reason = (judge.get("reasons") or ["unknown"])[0]
        out.append(Warning(
            type="judge_errored",
            severity="warning",
            message=(
                f"Judge could not evaluate this answer ({reason}). "
                f"Verdict is neutral, not a downgrade."
            ),
            detail={"reasons": judge.get("reasons") or []},
        ))

    # Background context: any catalog model currently degraded.
    snap = GLOBAL_HEALTH.snapshot()
    degraded = [p for p in snap["providers"] if p["degraded"]]
    # Skip the one we just used — we already warned about it above.
    if llm_used:
        degraded = [
            p for p in degraded
            if not (p["provider"] == llm_used.provider and p["model"] == llm_used.model)
        ]
    for p in degraded:
        last_err = p.get("last_error") or "unknown error"
        out.append(Warning(
            type="provider_degraded",
            severity="info",
            message=(
                f"{p['provider']}/{p['model']} has recent failures. "
                f"Last error: {last_err[:160]}"
            ),
            detail={"provider": p["provider"], "model": p["model"],
                    "last_error_status": p.get("last_error_status")},
        ))

    # No route + refused → tell the user which categories didn't match,
    # so they understand why the answer was refused.
    if refused and refusal_reason == "out_of_scope" and route == "none":
        out.append(Warning(
            type="no_route",
            severity="info",
            message=(
                "Question did not match any known domain (safety, maintenance, "
                "or quality). Try rephrasing with domain-specific terms."
            ),
        ))

    return out


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    run = get_pipeline()
    try:
        state = run(
            req.query,
            session_id=req.session_id,
            llm_choice=req.llm.model_dump() if req.llm else None,
            judge_choice=req.judge.model_dump() if req.judge else None,
        )
    except ProviderError as e:
        # Pinned provider with no fallback, or all providers failed.
        raise HTTPException(
            status_code=503 if e.retryable else 400,
            detail={
                "error": "llm_provider_error",
                "provider": e.provider,
                "status_code": e.status_code,
                "retryable": e.retryable,
                "message": str(e)[:600],
            },
        )

    citations: list[Citation] = []
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
                body_preview=(r.get("body") or "")[:400],
            )
        )

    judge = state.get("judge")
    raw_candidates = state.get("route_candidates") or []
    candidates = [
        RouteCandidate(
            route=str(c.get("route", "")),
            confidence=float(c.get("confidence", 0.0)),
            reason=str(c.get("reason", "")),
            source=str(c.get("source", "")),
        )
        for c in raw_candidates
        if c.get("route") in {"safety", "maintenance", "quality", "none"}
    ]
    collections_queried = list((state.get("retrieved_per_collection") or {}).keys())

    gen_meta = state.get("generation_meta") or {}
    llm_used: Optional[LlmUsage] = None
    if gen_meta:
        llm_used = LlmUsage(
            provider=str(gen_meta.get("provider", "unknown")),
            model=str(gen_meta.get("model", "unknown")),
            provider_fallback_used=bool(gen_meta.get("provider_fallback_used", False)),
            fallback_chain=list(gen_meta.get("fallback_chain") or []),
        )

    warnings = _build_warnings(
        llm_used=llm_used,
        judge=judge,
        refused=bool(state.get("refused")),
        refusal_reason=state.get("refusal_reason"),
        route=state.get("route", "none"),
        route_used_fallback=bool(state.get("route_used_fallback", False)),
    )

    return ChatResponse(
        trace_id=state["trace_id"],
        answer=state.get("answer", ""),
        refused=bool(state.get("refused")),
        refusal_reason=state.get("refusal_reason"),
        route=state.get("route", "none"),
        route_confidence=float(state.get("route_confidence", 0.0)),
        route_source=str(state.get("route_source", "llm")),
        route_candidates=candidates,
        route_used_fallback=bool(state.get("route_used_fallback", False)),
        collections_queried=collections_queried,
        retrieval_confidence=float(state.get("retrieval_confidence", 0.0)),
        citations=citations,
        judge=JudgeVerdictModel(**judge) if judge else None,
        llm_used=llm_used,
        warnings=warnings,
        latency_ms=state.get("latency_ms", {}),
    )


@app.get("/telemetry")
def telemetry(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(500, limit))
    p = _cfg().telemetry_path
    if not p.exists():
        return {"count": 0, "records": []}
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
    p = _cfg().scorecard_path
    if not p.exists():
        raise HTTPException(status_code=404, detail="scorecard not generated yet")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/admin/ingest", dependencies=[Depends(require_admin)])
def admin_ingest() -> dict[str, Any]:
    out = ingest_path()
    global _pipeline_runner, _store
    _pipeline_runner = None
    _store = None
    return out


@app.post("/admin/eval", dependencies=[Depends(require_admin)])
def admin_eval() -> dict[str, Any]:
    from .eval.run_offline import run_offline_eval

    return run_offline_eval()


# ---- Web UI ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    # Catalog + defaults are injected so first paint can populate the dropdowns
    # without an extra round trip.
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "factory-rag",
            "app_version": app.version,
        },
    )
