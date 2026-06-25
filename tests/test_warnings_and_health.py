"""Warning policy + health-monitor snapshot fields used by the UI."""
from __future__ import annotations

import json
from typing import Optional

import pytest

from app.adapters.llm import LLMResponse, ProviderError
from app.adapters.providers import GLOBAL_HEALTH, HealthMonitor


# --- New health fields -----------------------------------------------------

def test_snapshot_includes_last_event_fields():
    h = HealthMonitor()
    h.record("groq", "llama-3.3-70b-versatile", ok=True, latency_ms=42)
    snap = h.snapshot()
    p = snap["providers"][0]
    # Even before MIN_EVENTS_FOR_RATE, the UI gets last_event_* signals.
    assert p["last_event_ok"] is True
    assert p["last_event_latency_ms"] == 42
    assert p["last_event_ts"] is not None
    # Still no rate (need 3 events).
    assert p["recent_success_rate"] is None


def test_snapshot_last_event_reflects_most_recent():
    h = HealthMonitor()
    h.record("openrouter", "m", ok=True, latency_ms=10)
    h.record("openrouter", "m", ok=False, latency_ms=20, status_code=429, error="r")
    snap = h.snapshot()
    p = snap["providers"][0]
    assert p["last_event_ok"] is False
    assert p["last_event_latency_ms"] == 20


# --- Warning policy via /chat ---------------------------------------------

@pytest.fixture
def client_with_failover(monkeypatch, tmp_path):
    """Smoke client where OpenRouter raises retryable 429 and Groq succeeds."""
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

    from app.corpus.ingest import ingest_path
    ingest_path()

    # Clear any leaked health state from prior tests in the same process.
    GLOBAL_HEALTH._entries.clear()  # type: ignore[attr-defined]

    from app.adapters import llm as llm_mod

    # OpenRouter ALWAYS fails with retryable 429; max_retries default is 3 so
    # we'd hammer the stub 4 times. Override retry knobs at construction.
    def or_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                    model=None, response_format_json=False, pinned_provider=None):
        raise ProviderError(
            "openrouter 429 rate limit (test stub)",
            provider="openrouter",
            status_code=429,
            retryable=True,
        )

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", or_complete)

    # Groq always succeeds, with route/answer/judge handlers.
    def groq_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                      model=None, response_format_json=False, pinned_provider=None):
        s = system.lower()
        if "classify" in s:
            obj = {"primary": {"route": "safety", "confidence": 0.95, "reason": "x"},
                   "alternates": []}
            return LLMResponse(text=json.dumps(obj), model=model or "groq-fake",
                               prompt_tokens=1, completion_tokens=1, raw={}, provider="groq")
        if "manufacturing-floor assistant" in s:
            return LLMResponse(
                text="PPE: glasses, gloves, boots, hearing protection [SAFETY-LOTO-001#2.0].",
                model=model or "groq-fake",
                prompt_tokens=1, completion_tokens=10, raw={}, provider="groq",
            )
        if "you evaluate" in s:
            obj = {"grounded": True, "routing_ok": True, "score": 0.9, "reasons": []}
            return LLMResponse(text=json.dumps(obj), model=model or "groq-fake",
                               prompt_tokens=1, completion_tokens=1, raw={}, provider="groq")
        raise AssertionError(f"unexpected groq fake call: {system[:60]!r}")

    monkeypatch.setattr(llm_mod.GroqLLM, "complete", groq_complete)

    from app import main as app_main
    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient
    return TestClient(app_main.app)


def test_chat_fallback_warning_is_emitted(client_with_failover):
    r = client_with_failover.post(
        "/chat", json={"query": "What PPE is required for HP-200 lockout?"}
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is False
    # The answer was served by groq even though openrouter was primary.
    assert data["llm_used"]["provider"] == "groq"
    assert data["llm_used"]["provider_fallback_used"] is True
    # We expect a warning of type provider_fallback.
    warn_types = [w["type"] for w in data["warnings"]]
    assert "provider_fallback" in warn_types
    fallback_warning = next(w for w in data["warnings"] if w["type"] == "provider_fallback")
    assert fallback_warning["severity"] == "warning"
    assert fallback_warning["detail"]["primary"] == "openrouter"
    assert fallback_warning["detail"]["served_by"]["provider"] == "groq"


def test_health_endpoint_after_fallback_shows_openrouter_degraded(client_with_failover):
    client_with_failover.post(
        "/chat", json={"query": "What PPE is required for HP-200 lockout?"}
    )
    r = client_with_failover.get("/providers/health")
    snap = r.json()
    or_entry = next(
        (p for p in snap["providers"]
         if p["provider"] == "openrouter" and p["model"] == "openai/gpt-oss-120b:free"),
        None,
    )
    assert or_entry is not None
    assert or_entry["degraded"] is True
    assert or_entry["last_error_status"] == 429
    groq_entry = next(
        (p for p in snap["providers"] if p["provider"] == "groq"), None
    )
    assert groq_entry is not None
    assert groq_entry["degraded"] is False
    assert groq_entry["last_event_ok"] is True


def test_no_route_warning_on_out_of_scope(monkeypatch, tmp_path):
    """OOS query refuses with route=none; emits no_route info warning."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-oss-20b:free")
    monkeypatch.setenv("ROUTE_CONF_THRESHOLD", "0.5")
    monkeypatch.setenv("RETRIEVAL_CONF_THRESHOLD", "0.0")

    from app.corpus.ingest import ingest_path
    ingest_path()
    GLOBAL_HEALTH._entries.clear()  # type: ignore[attr-defined]

    from app.adapters import llm as llm_mod

    def oos_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                     model=None, response_format_json=False, pinned_provider=None):
        s = system.lower()
        if "classify" in s:
            obj = {"primary": {"route": "none", "confidence": 0.0, "reason": "oos"},
                   "alternates": []}
            return LLMResponse(text=json.dumps(obj), model="or", prompt_tokens=1,
                               completion_tokens=1, raw={}, provider="openrouter")
        # No other LLM calls should fire (guard short-circuits).
        if "you evaluate" in s:
            obj = {"grounded": True, "routing_ok": True, "score": 0.9, "reasons": []}
            return LLMResponse(text=json.dumps(obj), model="or", prompt_tokens=1,
                               completion_tokens=1, raw={}, provider="openrouter")
        raise AssertionError(f"unexpected: {system[:60]!r}")

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", oos_complete)

    from app import main as app_main
    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient
    c = TestClient(app_main.app)
    r = c.post("/chat", json={"query": "What's the weather tomorrow?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["refused"] is True
    assert data["refusal_reason"] == "out_of_scope"
    assert data["route"] == "none"
    warn_types = [w["type"] for w in data["warnings"]]
    assert "no_route" in warn_types
    no_route_warn = next(w for w in data["warnings"] if w["type"] == "no_route")
    assert no_route_warn["severity"] == "info"


def test_degraded_provider_not_warned_when_it_serves(client_with_failover):
    """The provider that just served the request must NOT also appear in a
    provider_degraded background warning — that would be double-counting."""
    # First call: openrouter degrades, groq serves.
    r1 = client_with_failover.post(
        "/chat", json={"query": "What PPE is required for HP-200 lockout?"}
    )
    data1 = r1.json()
    assert data1["llm_used"]["provider"] == "groq"
    # Pin to groq explicitly on the second call so groq is what serves
    # AND is currently in the degraded set (it isn't, but openrouter is).
    r2 = client_with_failover.post(
        "/chat",
        json={
            "query": "What PPE is required for HP-200 lockout?",
            "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            "judge": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        },
    )
    data2 = r2.json()
    # No fallback this time (pinned to groq, groq works).
    assert data2["llm_used"]["provider_fallback_used"] is False
    # The openrouter degraded background warning should still appear...
    degraded = [w for w in data2["warnings"] if w["type"] == "provider_degraded"]
    assert any(w["detail"]["provider"] == "openrouter" for w in degraded), data2
    # ...but groq (the serving provider) must NOT be in any provider_degraded
    # warning even if it had failures, because we dedupe.
    served = (data2["llm_used"]["provider"], data2["llm_used"]["model"])
    for w in degraded:
        assert (w["detail"]["provider"], w["detail"]["model"]) != served


def test_no_warnings_when_primary_works(monkeypatch, tmp_path):
    """Happy path: primary OpenRouter works, no fallback, no warnings."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    # No groq key -> single-provider mode.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-oss-20b:free")
    monkeypatch.setenv("ROUTE_CONF_THRESHOLD", "0.5")
    monkeypatch.setenv("RETRIEVAL_CONF_THRESHOLD", "0.0")

    from app.corpus.ingest import ingest_path
    ingest_path()
    GLOBAL_HEALTH._entries.clear()  # type: ignore[attr-defined]

    from app.adapters import llm as llm_mod

    def ok_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                    model=None, response_format_json=False, pinned_provider=None):
        s = system.lower()
        if "classify" in s:
            obj = {"primary": {"route": "safety", "confidence": 0.95, "reason": "x"},
                   "alternates": []}
            return LLMResponse(text=json.dumps(obj), model=model or "or",
                               prompt_tokens=1, completion_tokens=1, raw={}, provider="openrouter")
        if "manufacturing-floor assistant" in s:
            return LLMResponse(
                text="PPE: glasses, gloves, boots, hearing protection [SAFETY-LOTO-001#2.0].",
                model=model or "or",
                prompt_tokens=1, completion_tokens=10, raw={}, provider="openrouter",
            )
        if "you evaluate" in s:
            obj = {"grounded": True, "routing_ok": True, "score": 0.9, "reasons": []}
            return LLMResponse(text=json.dumps(obj), model=model or "or",
                               prompt_tokens=1, completion_tokens=1, raw={}, provider="openrouter")
        raise AssertionError("unexpected")

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", ok_complete)

    from app import main as app_main
    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient
    c = TestClient(app_main.app)
    r = c.post("/chat", json={"query": "What PPE is required for HP-200 lockout?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["llm_used"]["provider_fallback_used"] is False
    assert data["warnings"] == []


def test_judge_errored_warning(monkeypatch, tmp_path):
    """If the judge errors (parse failure), we emit a judge_errored warning
    but do NOT downgrade the answer."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-oss-20b:free")
    monkeypatch.setenv("ROUTE_CONF_THRESHOLD", "0.5")
    monkeypatch.setenv("RETRIEVAL_CONF_THRESHOLD", "0.0")

    from app.corpus.ingest import ingest_path
    ingest_path()
    GLOBAL_HEALTH._entries.clear()  # type: ignore[attr-defined]

    from app.adapters import llm as llm_mod

    def selective_complete(self, system, user, *, temperature=0.0, max_tokens=800,
                           model=None, response_format_json=False, pinned_provider=None):
        s = system.lower()
        if "classify" in s:
            obj = {"primary": {"route": "safety", "confidence": 0.95, "reason": "x"},
                   "alternates": []}
            return LLMResponse(text=json.dumps(obj), model="or",
                               prompt_tokens=1, completion_tokens=1, raw={}, provider="openrouter")
        if "manufacturing-floor assistant" in s:
            return LLMResponse(
                text="PPE: glasses [SAFETY-LOTO-001#2.0].",
                model="or", prompt_tokens=1, completion_tokens=5,
                raw={}, provider="openrouter",
            )
        if "you evaluate" in s:
            # Garbage that parse_json_strict can't decode.
            return LLMResponse(
                text="THE JUDGE WAS DRUNK AND DID NOT RETURN JSON",
                model="or", prompt_tokens=1, completion_tokens=1,
                raw={}, provider="openrouter",
            )
        raise AssertionError("unexpected")

    monkeypatch.setattr(llm_mod.OpenRouterLLM, "complete", selective_complete)

    from app import main as app_main
    app_main._pipeline_runner = None
    app_main._store = None

    from fastapi.testclient import TestClient
    c = TestClient(app_main.app)
    r = c.post("/chat", json={"query": "What PPE is required for HP-200 lockout?"})
    assert r.status_code == 200, r.text
    data = r.json()
    # Crucially: judge errored, but the answer was NOT downgraded.
    assert data["refused"] is False
    assert data["judge"]["judge_errored"] is True
    warn_types = [w["type"] for w in data["warnings"]]
    assert "judge_errored" in warn_types
    judge_warn = next(w for w in data["warnings"] if w["type"] == "judge_errored")
    assert judge_warn["severity"] == "warning"
