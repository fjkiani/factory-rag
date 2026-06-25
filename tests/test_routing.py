"""Routing tests: classify -> correct domain; retrieval hits only that collection."""
from __future__ import annotations

import json

from app.nodes.classify import make_classify_node
from app.nodes.retrieve import make_retrieve_node
from tests.conftest import FakeLLM


def _route_response(route: str, conf: float = 0.95, reason: str = "ok") -> str:
    return json.dumps({"route": route, "confidence": conf, "reason": reason})


def test_classify_safety(monkeypatch):
    llm = FakeLLM()
    llm.register(lambda s, u: "lockout" in u.lower() or "ppe" in u.lower(),
                 _route_response("safety"))
    state = {"query": "What PPE is required for lockout of the HP-200?"}
    out = make_classify_node(llm)(state)
    assert out["route"] == "safety"
    assert out["route_confidence"] >= 0.5


def test_classify_maintenance():
    llm = FakeLLM()
    llm.register(lambda s, u: "fault" in u.lower() or "cnc" in u.lower(),
                 _route_response("maintenance"))
    out = make_classify_node(llm)({"query": "Fault code E-318 on the CNC lathe?"})
    assert out["route"] == "maintenance"


def test_classify_quality():
    llm = FakeLLM()
    llm.register(lambda s, u: "aql" in u.lower() or "inspection" in u.lower(),
                 _route_response("quality"))
    out = make_classify_node(llm)({"query": "AQL 1.0 sample size for inspection?"})
    assert out["route"] == "quality"


def test_classify_out_of_scope():
    llm = FakeLLM()
    llm.register(lambda s, u: True, _route_response("none", conf=0.0, reason="oos"))
    out = make_classify_node(llm)({"query": "What's the weather?"})
    assert out["route"] == "none"
    assert out["route_confidence"] == 0.0


def test_retrieve_isolates_to_routed_collection(ingested_store, hashing_embedder):
    """Routing to 'safety' must only return chunks from kb_safety."""
    node = make_retrieve_node(ingested_store, hashing_embedder)
    state = {"query": "lockout sequence steps", "route": "safety"}
    out = node(state)
    assert len(out["retrieved"]) > 0
    for r in out["retrieved"]:
        assert r["doc_id"].startswith("SAFETY-")


def test_retrieve_returns_empty_on_route_none(ingested_store, hashing_embedder):
    node = make_retrieve_node(ingested_store, hashing_embedder)
    out = node({"query": "whatever", "route": "none"})
    assert out["retrieved"] == []
    assert out["retrieval_confidence"] == 0.0
