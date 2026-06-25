"""End-to-end demo against real OpenRouter.

Runs 5 representative queries through the production pipeline:
    classify -> retrieve -> guard -> generate -> judge -> telemetry
and writes per-query JSON artifacts so you can inspect every stage.

Outputs:
    /mnt/results/rag-mvp/runs/<ts>/<qid>.state.json   <- full final state
    /mnt/results/rag-mvp/runs/<ts>/<qid>.summary.md   <- human-readable summary
    /mnt/results/rag-mvp/runs/<ts>/telemetry.jsonl    <- the telemetry log
    /mnt/results/rag-mvp/runs/<ts>/manifest.json      <- run config + models used
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make sure we're using the workspace copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.embeddings import get_embedder
from app.adapters.llm import OpenRouterLLM
from app.adapters.vectordb import NumpyVectorStore, Point
from app.config import load_config
from app.corpus.ingest import DEFAULT_CORPUS
from app.nodes.classify import make_classify_node
from app.nodes.generate import make_generate_node
from app.nodes.guard import make_guard_node
from app.nodes.judge import make_judge_node
from app.nodes.retrieve import make_retrieve_node
from app.nodes.telemetry import make_telemetry_node
from app.pipeline import make_pipeline


DEMO_QUERIES = [
    {
        "qid": "S1",
        "expected_route": "safety",
        "query": "What PPE is required for HP-200 lockout?",
        "purpose": "Single-domain SAFETY query. Tests cite-or-answer happy path.",
    },
    {
        "qid": "M1",
        "expected_route": "maintenance",
        "query": "How do I clear fault code E-318 on the L-450 lathe?",
        "purpose": "Single-domain MAINTENANCE query with a numeric fault code.",
    },
    {
        "qid": "Q1",
        "expected_route": "quality",
        "query": "What sample size and accept/reject limits apply for a lot of 200 parts under AQL 1.0?",
        "purpose": "Single-domain QUALITY query needing the AQL table.",
    },
    {
        "qid": "AMB1",
        "expected_route": "safety|maintenance",
        "query": "What PPE is required during the 500-hour service of the L-450?",
        "purpose": "MULTI-DOMAIN: spans safety (PPE) AND maintenance (500h service). Exercises route fanout.",
    },
    {
        "qid": "OOS1",
        "expected_route": "none",
        "query": "Who is the CEO of Acme Corporation?",
        "purpose": "OUT-OF-SCOPE. Should refuse with out_of_scope, never hallucinate.",
    },
]


def build_pipeline(cfg, run_dir: Path):
    llm = OpenRouterLLM(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        default_model=cfg.llm_model,
        http_referer=cfg.http_referer,
        app_title=cfg.app_title,
    )
    embedder = get_embedder(
        cfg.embed_backend,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        model=cfg.embed_model,
        http_referer=cfg.http_referer,
        app_title=cfg.app_title,
    )

    # Fresh in-memory store under run_dir (so each demo run is reproducible)
    store = NumpyVectorStore(run_dir / "index.pkl")
    with open(DEFAULT_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    texts, keys = [], []
    for doc in corpus["documents"]:
        for sec in doc["sections"]:
            cid = f"{doc['doc_id']}#{sec['section_id']}"
            payload = {
                "chunk_id": cid,
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "domain": doc["domain"],
                "section_id": sec["section_id"],
                "heading": sec["heading"],
                "body": sec["body"],
            }
            texts.append(f"{sec['heading']}\n\n{sec['body']}")
            keys.append((f"kb_{doc['domain']}", cid, payload))
    vecs = embedder.embed(texts)
    grouped: dict[str, list[Point]] = {}
    for (collection, cid, payload), v in zip(keys, vecs):
        grouped.setdefault(collection, []).append(
            Point(chunk_id=cid, vector=v, payload=payload)
        )
    for collection, points in grouped.items():
        store.upsert(collection, points)

    nodes = [
        make_classify_node(llm),
        make_retrieve_node(store, embedder, max_collections=2, min_secondary_conf=0.30),
        make_guard_node(
            route_conf_threshold=cfg.route_conf_threshold,
            retrieval_conf_threshold=cfg.retrieval_conf_threshold,
        ),
        make_generate_node(llm),
        make_judge_node(llm, model=cfg.judge_model),
        make_telemetry_node(run_dir / "telemetry.jsonl"),
    ]
    return make_pipeline(nodes), llm, embedder, store


def summarize_md(qid: str, q: str, expected: str, state: dict, purpose: str) -> str:
    L = []
    L.append(f"# {qid}\n")
    L.append(f"**Purpose**: {purpose}\n")
    L.append(f"**Question**: {q}\n")
    L.append(f"**Expected route**: `{expected}`\n")
    L.append("---\n")
    L.append("## Routing\n")
    L.append(f"- primary route: `{state.get('route')}`")
    L.append(f"- route confidence: `{state.get('route_confidence'):.3f}`")
    L.append(f"- route source: `{state.get('route_source')}`")
    L.append(f"- used keyword fallback: `{state.get('route_used_fallback')}`")
    if state.get("route_llm_error"):
        L.append(f"- LLM error: `{state.get('route_llm_error')}`")
    cands = state.get("route_candidates") or []
    if len(cands) > 1:
        L.append("- alternates queried:")
        for c in cands[1:]:
            L.append(f"  - `{c['route']}` (conf={c.get('confidence'):.2f}, src={c.get('source')})")
    L.append("")
    L.append("## Retrieval\n")
    L.append(f"- collections queried: `{list((state.get('retrieved_per_collection') or {}).keys())}`")
    L.append(f"- top-N fused chunks: `{len(state.get('retrieved') or [])}`")
    for r in (state.get("retrieved") or [])[:5]:
        L.append(
            f"  - `[{r['chunk_id']}]` domain=`{r['domain']}` "
            f"rrf=`{r.get('rrf_score', 0):.4f}` "
            f"dense=`{r.get('dense_score', 0):.4f}` sparse=`{r.get('sparse_score', 0):.4f}`"
        )
    L.append("")
    L.append("## Generation\n")
    if state.get("refused"):
        L.append(f"- **REFUSED** — reason=`{state.get('refusal_reason')}`")
        L.append(f"- refusal text: {state.get('answer', '')!r}")
    else:
        L.append("```")
        L.append(state.get("answer", ""))
        L.append("```")
        L.append(f"- citations: `{state.get('citations')}`")
    L.append("")
    L.append("## Judge\n")
    j = state.get("judge") or {}
    if j:
        L.append(f"- grounded: `{j.get('grounded')}`")
        L.append(f"- score: `{j.get('score')}`")
        L.append(f"- rationale: {j.get('rationale', '')}")
    else:
        L.append("- (no judge output — refused or judge skipped)")
    L.append("")
    L.append("## Timing (ms)\n")
    for stage, ms in (state.get("latency_ms") or {}).items():
        L.append(f"- `{stage}`: {ms} ms")
    return "\n".join(L)


def main():
    cfg = load_config()
    assert cfg.openrouter_api_key, "OPENROUTER_API_KEY required for live demo"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path("/mnt/results/rag-mvp/runs") / ts
    work_root = Path("/workspace/rag-mvp/runs") / ts  # avoid S3 random-access for pickle
    work_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[demo] run dir: {out_root}")
    print(f"[demo] using LLM={cfg.llm_model}, JUDGE={cfg.judge_model}, EMBED={cfg.embed_model} ({cfg.embed_backend})")

    pipe, _llm, _emb, _store = build_pipeline(cfg, work_root)

    summary_rows = []
    for q in DEMO_QUERIES:
        print(f"\n[demo] === {q['qid']}: {q['query']!r}")
        t0 = time.perf_counter()
        try:
            state = pipe(q["query"])
        except Exception as e:
            print(f"[demo]  -> EXCEPTION {type(e).__name__}: {str(e)[:300]}")
            (out_root / f"{q['qid']}.error.txt").write_text(
                f"{type(e).__name__}: {e}"
            )
            summary_rows.append({
                "qid": q["qid"],
                "expected": q["expected_route"],
                "exception": f"{type(e).__name__}: {str(e)[:300]}",
            })
            # Brief cooldown to let upstream rate limits clear
            time.sleep(3.0)
            continue
        dt_ms = int((time.perf_counter() - t0) * 1000)

        # Write per-query artifacts
        (out_root / f"{q['qid']}.state.json").write_text(
            json.dumps(state, indent=2, default=str)
        )
        (out_root / f"{q['qid']}.summary.md").write_text(
            summarize_md(q["qid"], q["query"], q["expected_route"], state, q["purpose"])
        )

        print(
            f"[demo]  -> route={state.get('route')} "
            f"refused={state.get('refused')} "
            f"citations={state.get('citations')} "
            f"judge={(state.get('judge') or {}).get('grounded')} "
            f"total_ms={dt_ms}"
        )
        summary_rows.append({
            "qid": q["qid"],
            "expected": q["expected_route"],
            "route": state.get("route"),
            "route_conf": state.get("route_confidence"),
            "used_fallback": state.get("route_used_fallback"),
            "collections_queried": list((state.get("retrieved_per_collection") or {}).keys()),
            "refused": state.get("refused"),
            "refusal_reason": state.get("refusal_reason"),
            "citations": state.get("citations"),
            "judge_grounded": (state.get("judge") or {}).get("grounded"),
            "judge_score": (state.get("judge") or {}).get("score"),
            "total_ms": dt_ms,
        })

    # Manifest
    (out_root / "manifest.json").write_text(json.dumps({
        "timestamp": ts,
        "llm_model": cfg.llm_model,
        "judge_model": cfg.judge_model,
        "embed_model": cfg.embed_model,
        "embed_backend": cfg.embed_backend,
        "vector_backend": cfg.vector_backend,
        "route_conf_threshold": cfg.route_conf_threshold,
        "retrieval_conf_threshold": cfg.retrieval_conf_threshold,
        "queries": DEMO_QUERIES,
        "summary": summary_rows,
    }, indent=2))

    # Copy telemetry log
    tel_src = work_root / "telemetry.jsonl"
    if tel_src.exists():
        shutil.copy(tel_src, out_root / "telemetry.jsonl")

    print(f"\n[demo] DONE. Artifacts in {out_root}")


if __name__ == "__main__":
    main()
