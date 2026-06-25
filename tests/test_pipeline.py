"""End-to-end pipeline happy paths using FakeLLM + HashingEmbedder."""
from __future__ import annotations

import json
from pathlib import Path

from app.nodes.classify import make_classify_node
from app.nodes.generate import make_generate_node
from app.nodes.guard import make_guard_node
from app.nodes.judge import make_judge_node
from app.nodes.retrieve import make_retrieve_node
from app.nodes.telemetry import make_telemetry_node
from app.pipeline import make_pipeline
from tests.conftest import FakeLLM


def _llm_for(route: str, gen_text: str, judge_grounded: bool = True) -> FakeLLM:
    llm = FakeLLM()
    # classify (new shape: primary + alternates)
    llm.register(
        lambda s, u: "classify" in s.lower(),
        json.dumps({
            "primary": {"route": route, "confidence": 0.95, "reason": "ok"},
            "alternates": [],
        }),
    )
    # generate
    llm.register(lambda s, u: "manufacturing-floor assistant" in s.lower(), gen_text)
    # judge
    llm.register(
        lambda s, u: "you evaluate" in s.lower(),
        json.dumps({"grounded": judge_grounded, "routing_ok": True, "score": 0.9 if judge_grounded else 0.2, "reasons": ["ok"]}),
    )
    return llm


def test_pipeline_happy_path_safety(tmp_data_dir, ingested_store, hashing_embedder):
    llm = _llm_for(
        "safety",
        "PPE includes ANSI Z87.1 safety glasses, cut-resistant gloves, steel-toed boots, hearing protection [SAFETY-LOTO-001#2.0].",
        judge_grounded=True,
    )
    nodes = [
        make_classify_node(llm),
        make_retrieve_node(ingested_store, hashing_embedder),
        make_guard_node(route_conf_threshold=0.5, retrieval_conf_threshold=0.0),  # disable conf gate; we're using deterministic embed
        make_generate_node(llm),
        make_judge_node(llm),
        make_telemetry_node(tmp_data_dir / "telemetry.jsonl"),
    ]
    run = make_pipeline(nodes)
    state = run("What PPE is required for HP-200 lockout?")
    assert state["route"] == "safety"
    assert state["refused"] is False
    assert "SAFETY-LOTO-001#2.0" in state["citations"]
    assert state["judge"]["grounded"] is True
    assert (tmp_data_dir / "telemetry.jsonl").exists()
    # Telemetry has one JSON line
    lines = (tmp_data_dir / "telemetry.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["refused"] is False and rec["route"] == "safety"


def test_pipeline_judge_downgrades_ungrounded(tmp_data_dir, ingested_store, hashing_embedder):
    llm = _llm_for(
        "safety",
        # Cites a real chunk but the answer is off-topic; judge will say ungrounded.
        "Pineapple pizza is delicious [SAFETY-LOTO-001#1.0].",
        judge_grounded=False,
    )
    nodes = [
        make_classify_node(llm),
        make_retrieve_node(ingested_store, hashing_embedder),
        make_guard_node(route_conf_threshold=0.5, retrieval_conf_threshold=0.0),
        make_generate_node(llm),
        make_judge_node(llm),
        make_telemetry_node(tmp_data_dir / "telemetry.jsonl"),
    ]
    run = make_pipeline(nodes)
    state = run("What is required PPE?")
    assert state["refused"] is True
    assert state["refusal_reason"] == "judge_ungrounded"
    assert state["citations"] == []


def test_pipeline_out_of_scope(tmp_data_dir, ingested_store, hashing_embedder):
    llm = FakeLLM()
    llm.register(lambda s, u: "classify" in s.lower(),
                 json.dumps({"primary": {"route": "none", "confidence": 0.0, "reason": "oos"}, "alternates": []}))
    # judge still called -- registers a default verdict
    llm.register(lambda s, u: "you evaluate" in s.lower(),
                 json.dumps({"grounded": True, "routing_ok": True, "score": 0.95, "reasons": ["correctly refused"]}))
    nodes = [
        make_classify_node(llm),
        make_retrieve_node(ingested_store, hashing_embedder),
        make_guard_node(route_conf_threshold=0.5, retrieval_conf_threshold=0.35),
        make_generate_node(llm),
        make_judge_node(llm),
        make_telemetry_node(tmp_data_dir / "telemetry.jsonl"),
    ]
    run = make_pipeline(nodes)
    state = run("What is the weather tomorrow?")
    assert state["refused"] is True
    assert state["refusal_reason"] == "out_of_scope"
    assert state["citations"] == []
