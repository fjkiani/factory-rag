"""HTTP smoke test against the FastAPI app via TestClient.
Patches the OpenRouter LLM with FakeLLM, uses HashingEmbedder, NumPy store.
No network. Verifies /healthz, /chat (happy + refusal), /telemetry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Sandbox env BEFORE importing the app
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-smoke-test")
    monkeypatch.setenv("LLM_MODEL", "fake-llm")
    monkeypatch.setenv("JUDGE_MODEL", "fake-judge")
    monkeypatch.setenv("ROUTE_CONF_THRESHOLD", "0.5")
    monkeypatch.setenv("RETRIEVAL_CONF_THRESHOLD", "0.0")  # we use deterministic hashing embeddings; don't gate on confidence

    # First ingest
    from app.corpus.ingest import ingest_path

    out = ingest_path()
    assert out["counts"] == {"safety": 5, "maintenance": 4, "quality": 4}

    # Patch OpenRouterLLM.complete to a fake
    from app.adapters.llm import LLMResponse
    from app.adapters import llm as llm_mod

    def fake_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                      model=None, response_format_json=False, pinned_provider=None):
        s = system.lower()
        if "classify" in s:
            if "weather" in user.lower() or "fibonacci" in user.lower():
                obj = {"primary": {"route": "none", "confidence": 0.0, "reason": "oos"}, "alternates": []}
            elif "ppe" in user.lower() or "lockout" in user.lower():
                obj = {"primary": {"route": "safety", "confidence": 0.95, "reason": "safety"}, "alternates": []}
            elif "fault" in user.lower() or "lathe" in user.lower():
                obj = {"primary": {"route": "maintenance", "confidence": 0.95, "reason": "maintenance"}, "alternates": []}
            elif "aql" in user.lower() or "inspection" in user.lower():
                obj = {"primary": {"route": "quality", "confidence": 0.95, "reason": "quality"}, "alternates": []}
            else:
                obj = {"primary": {"route": "none", "confidence": 0.0, "reason": "default"}, "alternates": []}
            return LLMResponse(text=json.dumps(obj), model=model or "fake", prompt_tokens=10, completion_tokens=10, raw={})
        if "manufacturing-floor assistant" in s:
            # Cite the section we expect: PPE -> SAFETY-LOTO-001#2.0
            if "ppe" in user.lower():
                txt = "Required PPE: ANSI Z87.1 safety glasses, A4 cut-resistant gloves, steel-toed boots, hearing protection [SAFETY-LOTO-001#2.0]."
            elif "fault" in user.lower() or "e-318" in user.lower():
                txt = "For E-318, measure ball-screw axial play; if > 0.02 mm, schedule MAINT-L450-BS [MAINT-CNC-014#4.0]."
            elif "aql" in user.lower():
                txt = "For 200 parts under AQL 1.0: sample 32, accept 1 / reject 2 [QC-ISO-009#1.0]."
            else:
                txt = "I don't have that information in the safety documentation."
            return LLMResponse(text=txt, model=model or "fake", prompt_tokens=10, completion_tokens=30, raw={})
        if "you evaluate" in s:
            obj = {"grounded": True, "routing_ok": True, "score": 0.95, "reasons": ["ok"]}
            return LLMResponse(text=json.dumps(obj), model=model or "fake-judge", prompt_tokens=10, completion_tokens=10, raw={})
        raise AssertionError(f"unexpected fake_complete call; system={s[:60]!r}")

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", fake_complete)

    # Build the app AFTER env + patches; reset singletons.
    from app import main as app_main

    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient

    return TestClient(app_main.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["collections"]["kb_safety"] >= 1
    assert data["collections"]["kb_maintenance"] >= 1
    assert data["collections"]["kb_quality"] >= 1


def test_chat_happy_safety(client):
    r = client.post("/chat", json={"query": "What PPE is required for HP-200 lockout?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is False
    assert data["route"] == "safety"
    assert any(c["chunk_id"] == "SAFETY-LOTO-001#2.0" for c in data["citations"]), data
    assert data["judge"]["grounded"] is True


def test_chat_happy_maintenance(client):
    r = client.post("/chat", json={"query": "How do I respond to fault E-318 on the lathe?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is False
    assert data["route"] == "maintenance"
    assert any(c["chunk_id"] == "MAINT-CNC-014#4.0" for c in data["citations"])


def test_chat_happy_quality(client):
    r = client.post("/chat", json={"query": "What is the AQL 1.0 sample size for a lot of 200 parts?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is False
    assert data["route"] == "quality"
    assert any(c["chunk_id"] == "QC-ISO-009#1.0" for c in data["citations"])


def test_chat_out_of_scope_refuses(client):
    r = client.post("/chat", json={"query": "What is the weather tomorrow?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is True
    assert data["refusal_reason"] == "out_of_scope"
    assert data["citations"] == []
    assert data["route"] == "none"


def test_telemetry_endpoint(client):
    # Drive a couple of queries first
    client.post("/chat", json={"query": "What PPE is required for HP-200 lockout?"})
    client.post("/chat", json={"query": "What is the weather tomorrow?"})
    r = client.get("/telemetry?limit=10")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] >= 2
    assert all("trace_id" in rec for rec in data["records"])
    assert any(rec["refused"] for rec in data["records"])
    assert any(not rec["refused"] for rec in data["records"])
