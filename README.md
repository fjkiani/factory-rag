# Manufacturing Floor RAG MVP

Floor-supervisor RAG over Safety / Maintenance / Quality docs. Strict cite-or-refuse. Online + offline LLM-as-judge. JSONL telemetry. **Zero-dependency vector store: NumPy in-memory (persisted via pickle).** Qdrant adapter is in the tree as a placeholder behind the same `VectorStore` protocol.

## Stack

- **FastAPI** + Pydantic v2, Swagger `/docs` is the demo surface
- **OpenRouter** for LLM and embeddings (one swappable adapter; flip `LLM_MODEL` / `JUDGE_MODEL` / `EMBED_MODEL` env vars)
- **NumPy + rank_bm25**: dense (cosine) + sparse (BM25) + **RRF fusion**, all in-process. No external DB.
- **Pure-function state pipeline** (TypedDict + `(state)->state` nodes) shaped so it can be lifted into LangGraph later **without rewriting any node**.

## Architecture

```
POST /chat -> classify -> retrieve (RRF: BM25 + dense) -> guard -> generate -> judge -> telemetry
```

- **Strict collection isolation:** one collection per domain (`kb_safety`, `kb_maintenance`, `kb_quality`); classify picks exactly one.
- **Cite-or-refuse contract:** every claim must cite a retrieved chunk id. Fabricated citations or uncited claims downgrade to refusal.
- **Online judge:** every response gated by a cheap LLM-as-judge call; ungrounded answers are downgraded to refusal.
- **Offline scorecard:** `python -m app.eval.run_offline` scores routing accuracy, refusal precision/recall, citation validity, mean groundedness against a fixed seed set.
- **Telemetry:** every query appends one JSON line to `${DATA_DIR}/telemetry.jsonl` — greppable, durable, zero infra.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENROUTER_API_KEY and ADMIN_TOKEN

# 1) Build the index from the 3 synthetic JSON docs
python -m app.corpus.ingest

# 2) Serve
uvicorn app.main:app --reload --port 8000

# 3) Demo via Swagger
open http://localhost:8000/docs
```

### Offline / no-network mode

For local development without hitting OpenRouter (deterministic, free, fast):

```bash
EMBED_BACKEND=hashing python -m app.corpus.ingest
EMBED_BACKEND=hashing uvicorn app.main:app --reload
```

(LLM and judge still need an OpenRouter key; embeddings switch to the in-process `HashingEmbedder`.)

## Tests

```bash
PYTHONPATH=. pytest -q
```

No tests hit the network. `FakeLLM` + `HashingEmbedder` exercise the full pipeline.

## Endpoints

| Method | Path                  | Description                                                          |
|--------|-----------------------|----------------------------------------------------------------------|
| GET    | `/healthz`            | Service + collection sizes + active models                           |
| POST   | `/chat`               | Ask a question, get answer + citations + judge verdict + latencies   |
| GET    | `/telemetry?limit=50` | Last N query telemetry records                                       |
| GET    | `/eval/scorecard`     | Latest offline eval scorecard                                        |
| POST   | `/admin/ingest`       | Rebuild index from `seed_docs.json` (bearer-auth)                    |
| POST   | `/admin/eval`         | Re-run offline eval and write scorecard (bearer-auth)                |

## Swap to a different LLM / embedder / vector store

- **LLM model:** set `LLM_MODEL=<openrouter-model-id>`. Default `openrouter/auto`.
- **Judge model:** set `JUDGE_MODEL=<openrouter-model-id>`. Default `google/gemini-flash-1.5:free`.
- **Embed model:** set `EMBED_MODEL=<openrouter-model-id>`. Default `nvidia/llama-nemotron-embed-vl-1b-v2:free`.
- **Vector backend:** set `VECTOR_BACKEND=numpy` (default) or `qdrant` (placeholder, not implemented in MVP).

No code changes required for any of the above.

## Deploy (Render)

`render.yaml` declares one web service with a 1 GB persistent disk mounted at `/data` so `telemetry.jsonl`, `index.npz`, and `scorecard.json` survive restarts. Set `OPENROUTER_API_KEY` and `ADMIN_TOKEN` in the Render dashboard.

## What's intentionally out of scope (MVP)

- LangGraph integration (nodes are already LangGraph-shaped)
- Real PDF ingestion (corpus is structured JSON)
- Multi-turn memory
- Per-user auth on `/chat`
- Streaming responses
- Cross-encoder reranker
