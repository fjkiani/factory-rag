"""Env-driven config. No secrets in code; everything is overridable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv optional; in prod, env is set by the platform
    pass


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    judge_model: str
    embed_model: str
    vector_backend: str
    qdrant_url: str
    qdrant_api_key: str
    route_conf_threshold: float
    retrieval_conf_threshold: float
    admin_token: str
    data_dir: Path
    http_referer: str
    app_title: str
    embed_backend: str

    @property
    def telemetry_path(self) -> Path:
        return self.data_dir / "telemetry.jsonl"

    @property
    def scorecard_path(self) -> Path:
        return self.data_dir / "scorecard.json"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.npz"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25.pkl"


def load_config() -> Config:
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_model=os.getenv("LLM_MODEL", "openrouter/auto"),
        judge_model=os.getenv("JUDGE_MODEL", "google/gemini-flash-1.5:free"),
        embed_model=os.getenv("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free"),
        vector_backend=os.getenv("VECTOR_BACKEND", "numpy"),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        route_conf_threshold=float(os.getenv("ROUTE_CONF_THRESHOLD", "0.5")),
        retrieval_conf_threshold=float(os.getenv("RETRIEVAL_CONF_THRESHOLD", "0.35")),
        admin_token=os.getenv("ADMIN_TOKEN", "change-me"),
        data_dir=data_dir,
        http_referer=os.getenv("HTTP_REFERER", "https://github.com/fjkiani/factory-rag"),
        app_title=os.getenv("APP_TITLE", "factory-rag-mvp"),
        embed_backend=os.getenv("EMBED_BACKEND", "openrouter"),
    )
