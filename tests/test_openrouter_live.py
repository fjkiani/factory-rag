"""Live OpenRouter smoke. SKIPPED unless OPENROUTER_API_KEY is set.

Runs a small, real conversation against OpenRouter to validate:
- The chosen LLM_MODEL returns parseable JSON for classify and judge.
- The chosen EMBED_MODEL is reachable on the /embeddings endpoint
  (and if not, fails LOUDLY with the actionable error from the adapter).
- One full /chat round-trip through real classify -> retrieve -> generate -> judge.

This is the test you want to run before a demo.
"""
from __future__ import annotations

import os

import pytest

requires_live = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set; skipping live OpenRouter smoke",
)


@requires_live
def test_openrouter_llm_classify_real_call():
    from app.adapters.llm import OpenRouterLLM, parse_json_strict
    from app.config import load_config

    cfg = load_config()
    llm = OpenRouterLLM(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        default_model=cfg.llm_model,
    )
    resp = llm.complete(
        system=(
            'You classify into one of {safety, maintenance, quality, none}. '
            'Return STRICT JSON: {"primary":{"route":"...","confidence":0..1,"reason":"..."},"alternates":[]}.'
        ),
        user="What PPE is required for HP-200 lockout?",
        temperature=0.0,
        max_tokens=200,
        response_format_json=True,
    )
    obj = parse_json_strict(resp.text)
    assert "primary" in obj, obj
    assert obj["primary"]["route"] == "safety", obj


@requires_live
def test_openrouter_embeddings_real_call():
    """Confirms EMBED_MODEL is reachable. If this fails, swap EMBED_MODEL or
    set EMBED_BACKEND=hashing per README."""
    from app.adapters.embeddings import OpenRouterEmbeddings
    from app.config import load_config

    cfg = load_config()
    emb = OpenRouterEmbeddings(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        model=cfg.embed_model,
    )
    vecs = emb.embed(["What PPE is required for HP-200 lockout?"])
    assert len(vecs) == 1
    assert len(vecs[0]) > 0


@requires_live
def test_full_pipeline_real_openrouter(tmp_path, monkeypatch):
    """End-to-end: real OpenRouter for classify + generate + judge. Embeddings
    use the in-process hashing embedder so this test never depends on the
    embedding endpoint being live (the previous test covers that)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")

    from app.adapters.embeddings import HashingEmbedder
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
    import json

    cfg = load_config()
    llm = OpenRouterLLM(api_key=cfg.openrouter_api_key, base_url=cfg.openrouter_base_url, default_model=cfg.llm_model)
    embedder = HashingEmbedder(dim=256)
    store = NumpyVectorStore(tmp_path / "index.pkl")
    with open(DEFAULT_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    texts, keys = [], []
    for doc in corpus["documents"]:
        for sec in doc["sections"]:
            cid = f"{doc['doc_id']}#{sec['section_id']}"
            payload = {
                "chunk_id": cid, "doc_id": doc["doc_id"], "doc_title": doc["title"],
                "domain": doc["domain"], "section_id": sec["section_id"],
                "heading": sec["heading"], "body": sec["body"],
            }
            texts.append(f"{sec['heading']}\n\n{sec['body']}")
            keys.append((f"kb_{doc['domain']}", cid, payload))
    vecs = embedder.embed(texts)
    grouped: dict[str, list[Point]] = {}
    for (collection, cid, payload), v in zip(keys, vecs):
        grouped.setdefault(collection, []).append(Point(chunk_id=cid, vector=v, payload=payload))
    for collection, points in grouped.items():
        store.upsert(collection, points)

    nodes = [
        make_classify_node(llm),
        make_retrieve_node(store, embedder),
        # Use the configured thresholds, not hard-coded values, so the live
        # smoke matches what's deployed.
        make_guard_node(
            route_conf_threshold=cfg.route_conf_threshold,
            retrieval_conf_threshold=cfg.retrieval_conf_threshold,
        ),
        make_generate_node(llm),
        make_judge_node(llm, model=cfg.judge_model),
        make_telemetry_node(tmp_path / "telemetry.jsonl"),
    ]
    run = make_pipeline(nodes)
    state = run("What PPE is required for HP-200 lockout?")
    # The real model should route to safety
    assert state["route"] == "safety", state
    # And produce a cited answer that references the PPE chunk
    if not state["refused"]:
        assert state["citations"], state
        assert any(c.startswith("SAFETY-LOTO-001") for c in state["citations"]), state
    else:
        # If the model refused, surface why so we can diagnose
        pytest.fail(f"Live pipeline refused: reason={state['refusal_reason']} answer={state.get('answer','')!r}")
