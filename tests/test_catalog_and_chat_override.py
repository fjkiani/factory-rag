"""Tests for /providers/catalog, /providers/health, and per-request model override on /chat."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-or-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-oss-20b:free")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("ROUTE_CONF_THRESHOLD", "0.5")
    monkeypatch.setenv("RETRIEVAL_CONF_THRESHOLD", "0.0")

    # Ingest first
    from app.corpus.ingest import ingest_path
    ingest_path()

    # Patch BOTH OpenRouterLLM and GroqLLM .complete to a tracking fake so we
    # can verify the right one was called when the request pins a provider.
    from app.adapters import llm as llm_mod
    from app.adapters.llm import LLMResponse

    calls: list[dict] = []

    def make_fake(provider_name: str):
        def fake_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                          model=None, response_format_json=False, pinned_provider=None):
            calls.append({
                "provider": provider_name, "model": model, "system_prefix": system[:30],
                "user_prefix": user[:60],
            })
            s = system.lower()
            if "classify" in s:
                if "weather" in user.lower():
                    obj = {"primary": {"route": "none", "confidence": 0.0, "reason": "oos"}, "alternates": []}
                elif "ppe" in user.lower():
                    obj = {"primary": {"route": "safety", "confidence": 0.95, "reason": "x"}, "alternates": []}
                else:
                    obj = {"primary": {"route": "none", "confidence": 0.0, "reason": "?"}, "alternates": []}
                return LLMResponse(text=json.dumps(obj), model=model or provider_name,
                                   prompt_tokens=1, completion_tokens=1, raw={}, provider=provider_name)
            if "manufacturing-floor assistant" in s:
                txt = "PPE: glasses, gloves, boots, hearing [SAFETY-LOTO-001#2.0]."
                return LLMResponse(text=txt, model=model or provider_name,
                                   prompt_tokens=1, completion_tokens=10, raw={}, provider=provider_name)
            if "you evaluate" in s:
                obj = {"grounded": True, "routing_ok": True, "score": 0.9, "reasons": []}
                return LLMResponse(text=json.dumps(obj), model=model or provider_name,
                                   prompt_tokens=1, completion_tokens=1, raw={}, provider=provider_name)
            raise AssertionError(f"unexpected: {system[:60]!r}")
        return fake_complete

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", make_fake("openrouter"))
    monkeypatch.setattr(llm_mod.GroqLLM, "complete", make_fake("groq"))

    from app import main as app_main
    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient
    c = TestClient(app_main.app)
    c.calls_log = calls  # type: ignore[attr-defined]
    return c


def test_catalog_lists_both_providers_when_both_configured(client):
    r = client.get("/providers/catalog")
    assert r.status_code == 200, r.text
    data = r.json()
    providers = {e["provider"] for e in data["entries"]}
    assert providers == {"openrouter", "groq"}
    # Defaults reflect env vars.
    assert data["defaults"]["answer"]["provider"] == "openrouter"
    assert data["defaults"]["answer"]["model"] == "openai/gpt-oss-120b:free"


def test_catalog_entries_have_required_fields(client):
    r = client.get("/providers/catalog")
    for e in r.json()["entries"]:
        assert {"provider", "model", "label", "roles", "id"} <= set(e.keys())
        assert e["id"] == f"{e['provider']}/{e['model']}"


def test_chat_with_groq_override_calls_groq(client):
    r = client.post("/chat", json={
        "query": "What PPE is required for HP-200 lockout?",
        "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is False
    assert data["llm_used"]["provider"] == "groq"
    assert data["llm_used"]["model"] == "llama-3.3-70b-versatile"
    assert data["llm_used"]["provider_fallback_used"] is False
    # Verify the generate call (system contains "manufacturing-floor assistant")
    # went to groq.
    gen_calls = [c for c in client.calls_log if "manufacturing-floor" in c["system_prefix"].lower()]
    assert gen_calls, "no generate call observed"
    assert all(c["provider"] == "groq" for c in gen_calls)
    assert all(c["model"] == "llama-3.3-70b-versatile" for c in gen_calls)


def test_chat_without_override_uses_default_openrouter(client):
    r = client.post("/chat", json={"query": "What PPE is required for HP-200 lockout?"})
    assert r.status_code == 200
    data = r.json()
    assert data["llm_used"]["provider"] == "openrouter"
    assert data["llm_used"]["model"] == "openai/gpt-oss-120b:free"


def test_chat_with_judge_override(client):
    r = client.post("/chat", json={
        "query": "What PPE is required for HP-200 lockout?",
        "judge": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["judge"]["provider"] == "groq"
    # Judge model came from the override.
    judge_calls = [c for c in client.calls_log if "you evaluate" in c["system_prefix"].lower()]
    assert judge_calls and all(c["provider"] == "groq" for c in judge_calls)


def test_chat_with_unknown_model_returns_422(client):
    r = client.post("/chat", json={
        "query": "What PPE is required?",
        "llm": {"provider": "openrouter", "model": "openrouter/auto"},  # not in catalog
    })
    assert r.status_code == 422


def test_chat_with_unknown_provider_returns_422(client):
    r = client.post("/chat", json={
        "query": "What PPE is required?",
        "llm": {"provider": "anthropic", "model": "claude-3"},  # provider not allowed
    })
    assert r.status_code == 422


def test_health_endpoint_returns_records_after_chat(client):
    # Drive one chat to populate health.
    client.post("/chat", json={"query": "What PPE is required?"})
    r = client.get("/providers/health")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    # At least the default (openrouter, gpt-oss-120b:free) should be recorded.
    assert any(p["provider"] == "openrouter" for p in data["providers"])


def test_healthz_lists_configured_providers(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert set(data["providers_configured"]) == {"openrouter", "groq"}


def test_index_html_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # Placeholder page links to docs and catalog.
    body = r.text.lower()
    assert "factory-rag" in body or "manufacturing" in body
