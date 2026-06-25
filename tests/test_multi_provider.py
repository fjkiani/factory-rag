"""Unit tests for MultiProviderLLM (failover) and HealthMonitor."""
from __future__ import annotations

import time
from typing import Optional

import pytest

from app.adapters.llm import LLMResponse, ProviderError
from app.adapters.providers import (
    DEGRADED_LOOKBACK_S,
    HealthMonitor,
    MultiProviderLLM,
    ProviderSlot,
)


# --- Stub LLM clients -------------------------------------------------------

class StubLLM:
    """Records every call and either returns a stub response or raises."""

    def __init__(self, name: str, behavior=None, model: str = "stub-model"):
        self.name = name
        self.behavior = behavior or (lambda: LLMResponse(
            text=f"ok from {name}",
            model=model,
            prompt_tokens=1,
            completion_tokens=1,
            raw={},
            provider=name,
        ))
        self.model = model
        self.calls: list[dict] = []

    def complete(self, system, user, *, temperature=0.0, max_tokens=800,
                 model=None, response_format_json=False):
        self.calls.append({"system": system, "user": user, "model": model})
        out = self.behavior()
        if isinstance(out, Exception):
            raise out
        return out


def _raise_429(name: str):
    return lambda: ProviderError(
        f"{name} 429 rate limit",
        provider=name,
        status_code=429,
        retryable=True,
    )


def _raise_400(name: str):
    return lambda: ProviderError(
        f"{name} 400 bad request",
        provider=name,
        status_code=400,
        retryable=False,
    )


def _raise_500(name: str):
    return lambda: ProviderError(
        f"{name} 500 server boom",
        provider=name,
        status_code=500,
        retryable=True,
    )


# --- Failover semantics -----------------------------------------------------

def test_primary_succeeds_no_fallback():
    health = HealthMonitor()
    p = StubLLM("openrouter")
    s = StubLLM("groq")
    llm = MultiProviderLLM(
        [ProviderSlot(p, "model-a"), ProviderSlot(s, "model-b")],
        health=health,
    )
    out = llm.complete("sys", "user")
    assert out.text == "ok from openrouter"
    assert out.provider_fallback_used is False
    assert out.fallback_chain == ["openrouter"]
    assert len(p.calls) == 1
    assert len(s.calls) == 0


def test_primary_429_falls_through_to_secondary():
    health = HealthMonitor()
    p = StubLLM("openrouter", behavior=_raise_429("openrouter"))
    s = StubLLM("groq")
    llm = MultiProviderLLM(
        [ProviderSlot(p, "openai/gpt-oss-120b:free"),
         ProviderSlot(s, "llama-3.3-70b-versatile")],
        health=health,
    )
    out = llm.complete("sys", "user")
    assert out.text == "ok from groq"
    assert out.provider_fallback_used is True
    assert out.fallback_chain == ["openrouter", "groq"]
    # Health recorded one failure (openrouter) and one success (groq).
    snap = health.snapshot()
    by_provider = {(p["provider"], p["model"]): p for p in snap["providers"]}
    assert by_provider[("openrouter", "openai/gpt-oss-120b:free")]["degraded"] is True
    assert by_provider[("groq", "llama-3.3-70b-versatile")]["degraded"] is False


def test_primary_400_does_not_fall_through():
    """400 is a caller bug. Don't pretend the secondary will save us."""
    health = HealthMonitor()
    p = StubLLM("openrouter", behavior=_raise_400("openrouter"))
    s = StubLLM("groq")
    llm = MultiProviderLLM(
        [ProviderSlot(p, "m"), ProviderSlot(s, "n")],
        health=health,
    )
    with pytest.raises(ProviderError) as excinfo:
        llm.complete("sys", "user")
    assert excinfo.value.status_code == 400
    assert excinfo.value.provider == "openrouter"
    assert len(s.calls) == 0  # secondary was NOT tried


def test_both_429_exhausts_with_chain():
    health = HealthMonitor()
    p = StubLLM("openrouter", behavior=_raise_429("openrouter"))
    s = StubLLM("groq", behavior=_raise_500("groq"))
    llm = MultiProviderLLM(
        [ProviderSlot(p, "m"), ProviderSlot(s, "n")],
        health=health,
    )
    with pytest.raises(ProviderError) as excinfo:
        llm.complete("sys", "user")
    # Last error was the secondary's 500.
    assert excinfo.value.provider == "groq"
    assert excinfo.value.status_code == 500
    # Both providers were tried.
    assert len(p.calls) == 1
    assert len(s.calls) == 1
    snap = health.snapshot()
    assert all(p["degraded"] for p in snap["providers"])


def test_network_exception_treated_as_retryable():
    health = HealthMonitor()
    p = StubLLM("openrouter", behavior=lambda: ConnectionError("DNS down"))
    s = StubLLM("groq")
    llm = MultiProviderLLM(
        [ProviderSlot(p, "m"), ProviderSlot(s, "n")],
        health=health,
    )
    out = llm.complete("sys", "user")
    assert out.provider_fallback_used is True
    assert out.fallback_chain == ["openrouter", "groq"]


def test_pinned_provider_uses_only_that_slot():
    """The UI's model picker pins both provider AND model. No silent fallback."""
    health = HealthMonitor()
    p = StubLLM("openrouter")
    s = StubLLM("groq")
    llm = MultiProviderLLM(
        [ProviderSlot(p, "m"), ProviderSlot(s, "n")],
        health=health,
    )
    out = llm.complete("sys", "user", pinned_provider="groq", model="llama-3.3-70b-versatile")
    assert out.text == "ok from groq"
    assert out.provider_fallback_used is False
    assert out.fallback_chain == ["groq"]
    assert len(p.calls) == 0
    # The pinned model was forwarded.
    assert s.calls[0]["model"] == "llama-3.3-70b-versatile"


def test_pinned_provider_failure_does_not_fall_through():
    health = HealthMonitor()
    p = StubLLM("openrouter")
    s = StubLLM("groq", behavior=_raise_429("groq"))
    llm = MultiProviderLLM(
        [ProviderSlot(p, "m"), ProviderSlot(s, "n")],
        health=health,
    )
    with pytest.raises(ProviderError) as excinfo:
        llm.complete("sys", "user", pinned_provider="groq", model="llama-3.3-70b-versatile")
    assert excinfo.value.provider == "groq"
    # Primary was never tried even though groq failed.
    assert len(p.calls) == 0


def test_pinned_provider_unknown_raises():
    health = HealthMonitor()
    p = StubLLM("openrouter")
    llm = MultiProviderLLM([ProviderSlot(p, "m")], health=health)
    with pytest.raises(ProviderError) as excinfo:
        llm.complete("sys", "user", pinned_provider="anthropic")
    assert excinfo.value.retryable is False
    assert "not configured" in str(excinfo.value)


# --- HealthMonitor ----------------------------------------------------------

def test_health_records_and_snapshots():
    h = HealthMonitor()
    h.record("openrouter", "gpt-oss-120b", ok=True, latency_ms=120)
    h.record("openrouter", "gpt-oss-120b", ok=True, latency_ms=150)
    h.record("openrouter", "gpt-oss-120b", ok=False, latency_ms=200, status_code=429, error="rate")
    snap = h.snapshot()
    assert len(snap["providers"]) == 1
    p = snap["providers"][0]
    assert p["event_count"] == 3
    assert p["last_error"] == "rate"
    assert p["last_error_status"] == 429
    assert p["degraded"] is True
    # 2/3 success rate.
    assert p["recent_success_rate"] == pytest.approx(2 / 3, rel=1e-3)


def test_health_not_degraded_after_recovery_outside_window():
    """If the failure is older than DEGRADED_LOOKBACK_S we treat it as cleared."""
    h = HealthMonitor()
    # Manually push an old failure event.
    h.record("openrouter", "m", ok=False, latency_ms=1, status_code=429, error="old")
    # Backdate it.
    entry = h._entries[("openrouter", "m")]
    old_event = entry.events[0]
    old_event.ts = time.time() - (DEGRADED_LOOKBACK_S + 100)
    # Record a fresh success.
    h.record("openrouter", "m", ok=True, latency_ms=10)
    snap = h.snapshot()
    p = snap["providers"][0]
    assert p["degraded"] is False


def test_health_separate_keys_per_model():
    h = HealthMonitor()
    h.record("openrouter", "model-a", ok=True, latency_ms=10)
    h.record("openrouter", "model-b", ok=False, latency_ms=10, status_code=429, error="x")
    snap = h.snapshot()
    by_model = {(p["provider"], p["model"]): p for p in snap["providers"]}
    assert by_model[("openrouter", "model-a")]["degraded"] is False
    assert by_model[("openrouter", "model-b")]["degraded"] is True


def test_health_is_degraded_query():
    h = HealthMonitor()
    assert h.is_degraded("openrouter", "m") is False  # unknown key = not degraded
    h.record("openrouter", "m", ok=False, latency_ms=1, status_code=429, error="r")
    assert h.is_degraded("openrouter", "m") is True
