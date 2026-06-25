"""Offline eval suite. Runs the full pipeline against a fixed seed question
set, computes a scorecard, writes it to disk. Surfaced via GET /eval/scorecard.

Acceptance bar (configurable): routing accuracy >= 0.90, refusal precision
>= 0.80, citation validity = 1.00.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..adapters.embeddings import get_embedder
from ..adapters.llm import OpenRouterLLM
from ..adapters.vectordb import get_vector_store
from ..config import load_config
from ..nodes.classify import make_classify_node
from ..nodes.generate import make_generate_node
from ..nodes.guard import make_guard_node
from ..nodes.judge import make_judge_node
from ..nodes.retrieve import make_retrieve_node
from ..nodes.telemetry import make_telemetry_node
from ..pipeline import make_pipeline

SEED_PATH = Path(__file__).resolve().parent / "seed_questions.json"


def _build_runner(cfg):
    llm = OpenRouterLLM(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        default_model=cfg.llm_model,
    )
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
    # Eval telemetry goes to a side file so it doesn't pollute prod telemetry.
    eval_tel = cfg.data_dir / "telemetry.eval.jsonl"
    nodes = [
        make_classify_node(llm, model=cfg.llm_model),
        make_retrieve_node(store, embedder),
        make_guard_node(
            route_conf_threshold=cfg.route_conf_threshold,
            retrieval_conf_threshold=cfg.retrieval_conf_threshold,
        ),
        make_generate_node(llm, model=cfg.llm_model),
        make_judge_node(llm, model=cfg.judge_model),
        make_telemetry_node(eval_tel),
    ]
    return make_pipeline(nodes)


def run_offline_eval() -> dict[str, Any]:
    cfg = load_config()
    run = _build_runner(cfg)
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed = json.load(f)

    results = []
    t0 = time.perf_counter()
    for q in seed["questions"]:
        state = run(q["query"])
        result = {
            "id": q["id"],
            "query": q["query"],
            "expected_route": q["expected_route"],
            "expected_refusal": q["expected_refusal"],
            "must_cite_any_of": q.get("must_cite_any_of", []),
            "predicted_route": state.get("route"),
            "refused": bool(state.get("refused")),
            "refusal_reason": state.get("refusal_reason"),
            "citations": state.get("citations", []),
            "judge": state.get("judge"),
            "answer": state.get("answer"),
            "retrieval_confidence": state.get("retrieval_confidence"),
            "route_confidence": state.get("route_confidence"),
        }
        results.append(result)

    # Metrics
    total = len(results)
    routing_correct = sum(
        1 for r in results if r["predicted_route"] == r["expected_route"]
    )
    routing_accuracy = routing_correct / total if total else 0.0

    # Refusal precision/recall
    tp = sum(1 for r in results if r["refused"] and r["expected_refusal"])
    fp = sum(1 for r in results if r["refused"] and not r["expected_refusal"])
    fn = sum(1 for r in results if not r["refused"] and r["expected_refusal"])
    refusal_precision = tp / (tp + fp) if (tp + fp) else 1.0
    refusal_recall = tp / (tp + fn) if (tp + fn) else 1.0

    # Citation validity: every cited id must appear in must_cite_any_of when defined
    citation_hits = 0
    citation_evaluated = 0
    for r in results:
        if r["expected_refusal"] or not r["must_cite_any_of"]:
            continue
        citation_evaluated += 1
        if any(c in r["must_cite_any_of"] for c in r["citations"]):
            citation_hits += 1
    citation_accuracy = (citation_hits / citation_evaluated) if citation_evaluated else 1.0

    # Mean groundedness from judge
    scores = [r["judge"]["score"] for r in results if r.get("judge")]
    mean_groundedness = sum(scores) / len(scores) if scores else 0.0

    elapsed_s = time.perf_counter() - t0

    scorecard = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 2),
        "model": cfg.llm_model,
        "judge_model": cfg.judge_model,
        "embed_model": cfg.embed_model,
        "vector_backend": cfg.vector_backend,
        "n_questions": total,
        "metrics": {
            "routing_accuracy": round(routing_accuracy, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusal_recall": round(refusal_recall, 4),
            "citation_accuracy": round(citation_accuracy, 4),
            "mean_groundedness": round(mean_groundedness, 4),
        },
        "acceptance": {
            "routing_accuracy_min": 0.90,
            "refusal_precision_min": 0.80,
            "citation_accuracy_min": 1.00,
            "passed": (
                routing_accuracy >= 0.90
                and refusal_precision >= 0.80
                and citation_accuracy >= 1.00
            ),
        },
        "results": results,
    }
    with open(cfg.scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    return scorecard


if __name__ == "__main__":
    sc = run_offline_eval()
    print(json.dumps(sc["metrics"], indent=2))
    print("PASSED" if sc["acceptance"]["passed"] else "FAILED")
