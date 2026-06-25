"""Provider failover + health monitoring.

`HealthMonitor` is in-process state: a rolling window of recent events per
(provider, model) key, derived into a `degraded` flag the UI can render.

`MultiProviderLLM` wraps an ordered list of LLM clients and falls through on
retryable upstream errors (429/5xx). Caller-error 4xx (400/401/402/404) is
re-raised because that's a config bug, not an outage.

Health is derived from real `/chat` traffic; we deliberately don't add a
synthetic ping so the signal reflects what the user actually experienced.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .llm import LLMClient, LLMResponse, ProviderError


# ---- HealthMonitor ---------------------------------------------------------

WINDOW_SIZE = 50           # max events retained per key
DEGRADED_LOOKBACK_S = 300  # any failure within this window flags the key as degraded
MIN_EVENTS_FOR_RATE = 3    # don't compute success rate until we have this many events


@dataclass
class HealthEvent:
    ts: float
    ok: bool
    latency_ms: int
    status_code: int          # 0 = network/exception, 200 = ok, 429/4xx/5xx otherwise
    error: Optional[str]


@dataclass
class HealthEntry:
    provider: str
    model: str
    events: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))

    def record(self, event: HealthEvent) -> None:
        self.events.append(event)

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        events_list = list(self.events)
        n = len(events_list)
        ok_count = sum(1 for e in events_list if e.ok)
        last_ok_ts = next((e.ts for e in reversed(events_list) if e.ok), None)
        last_err_event = next(
            (e for e in reversed(events_list) if not e.ok), None
        )
        # Degraded if: any failure in the last DEGRADED_LOOKBACK_S window.
        recent_failure = any(
            (not e.ok) and (now - e.ts) <= DEGRADED_LOOKBACK_S for e in events_list
        )
        success_rate: Optional[float] = None
        if n >= MIN_EVENTS_FOR_RATE:
            success_rate = ok_count / n
        last_event = events_list[-1] if events_list else None
        return {
            "provider": self.provider,
            "model": self.model,
            "event_count": n,
            "degraded": recent_failure,
            "last_ok_ts": last_ok_ts,
            "last_error": last_err_event.error if last_err_event else None,
            "last_error_ts": last_err_event.ts if last_err_event else None,
            "last_error_status": last_err_event.status_code if last_err_event else None,
            "recent_success_rate": success_rate,
            # Cheap signals the UI can render before MIN_EVENTS_FOR_RATE is hit.
            "last_event_ok": (last_event.ok if last_event else None),
            "last_event_ts": (last_event.ts if last_event else None),
            "last_event_latency_ms": (last_event.latency_ms if last_event else None),
        }


class HealthMonitor:
    """Thread-safe in-process health log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], HealthEntry] = {}

    def record(
        self,
        provider: str,
        model: str,
        *,
        ok: bool,
        latency_ms: int,
        status_code: int = 200,
        error: Optional[str] = None,
    ) -> None:
        key = (provider, model)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = HealthEntry(provider=provider, model=model)
                self._entries[key] = entry
            entry.record(
                HealthEvent(
                    ts=time.time(),
                    ok=ok,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error=error,
                )
            )

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            providers = [e.snapshot(now=now) for e in self._entries.values()]
        # Sort: degraded first, then by provider/model for stable UI rendering.
        providers.sort(key=lambda d: (not d["degraded"], d["provider"], d["model"]))
        return {"checked_at": now, "providers": providers}

    def is_degraded(self, provider: str, model: str) -> bool:
        key = (provider, model)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            return entry.snapshot()["degraded"]


# Single process-wide monitor. main.py imports this.
GLOBAL_HEALTH = HealthMonitor()


# ---- MultiProviderLLM ------------------------------------------------------

@dataclass
class ProviderSlot:
    client: LLMClient
    default_model: str


class MultiProviderLLM:
    """Tries providers in order. Falls through on retryable errors (429/5xx /
    network). Re-raises non-retryable caller errors (400/401/402/404).

    Each successful or failed attempt is logged to the HealthMonitor so the UI
    can show degraded badges and the chat response can surface a warning when
    the primary provider was unavailable.

    If `pinned_provider` is set on `complete()`, ONLY that provider is tried
    (and only with the supplied model). This is how the UI's model picker
    expresses "use exactly this; don't silently switch providers behind my
    back".
    """

    name = "multi"

    def __init__(
        self,
        slots: Iterable[ProviderSlot],
        health: Optional[HealthMonitor] = None,
    ):
        self.slots = list(slots)
        if not self.slots:
            raise ValueError("MultiProviderLLM requires at least one slot")
        self.health = health or GLOBAL_HEALTH

    def _slot_for(self, provider: str) -> Optional[ProviderSlot]:
        for s in self.slots:
            if s.client.name == provider:
                return s
        return None

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model: Optional[str] = None,
        response_format_json: bool = False,
        pinned_provider: Optional[str] = None,
    ) -> LLMResponse:
        # Pinned mode: try exactly one slot, no fallback.
        if pinned_provider is not None:
            slot = self._slot_for(pinned_provider)
            if slot is None:
                available = [s.client.name for s in self.slots]
                raise ProviderError(
                    f"Pinned provider {pinned_provider!r} not configured "
                    f"(available: {available})",
                    provider=pinned_provider,
                    status_code=0,
                    retryable=False,
                )
            return self._invoke(
                slot,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model or slot.default_model,
                response_format_json=response_format_json,
                chain=[slot.client.name],
                fallback_used=False,
            )

        # Failover mode: walk the slots in order.
        last_error: Optional[ProviderError] = None
        chain: list[str] = []
        for i, slot in enumerate(self.slots):
            chain.append(slot.client.name)
            try:
                return self._invoke(
                    slot,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=slot.default_model if i > 0 else (model or slot.default_model),
                    response_format_json=response_format_json,
                    chain=list(chain),
                    fallback_used=(i > 0),
                )
            except ProviderError as e:
                last_error = e
                if not e.retryable:
                    # Caller error: don't keep trying. The other providers
                    # would just give the same answer to the same bad input.
                    raise
                # Retryable: try the next slot.
                continue
            except Exception as e:  # network errors etc.
                # Record and try next.
                self.health.record(
                    provider=slot.client.name,
                    model=slot.default_model,
                    ok=False,
                    latency_ms=0,
                    status_code=0,
                    error=f"{type(e).__name__}: {e}"[:300],
                )
                last_error = ProviderError(
                    f"{slot.client.name} network error: {e}",
                    provider=slot.client.name,
                    status_code=0,
                    retryable=True,
                )
                continue
        # Exhausted all providers.
        assert last_error is not None
        raise ProviderError(
            f"All providers failed (chain={chain}): last={last_error}",
            provider=last_error.provider,
            status_code=last_error.status_code,
            retryable=last_error.retryable,
        )

    def _invoke(
        self,
        slot: ProviderSlot,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        model: str,
        response_format_json: bool,
        chain: list[str],
        fallback_used: bool,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = slot.client.complete(
                system,
                user,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                response_format_json=response_format_json,
            )
        except ProviderError as e:
            self.health.record(
                provider=slot.client.name,
                model=model,
                ok=False,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                status_code=e.status_code,
                error=str(e)[:300],
            )
            raise
        # Success.
        self.health.record(
            provider=slot.client.name,
            model=model,
            ok=True,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            status_code=200,
            error=None,
        )
        resp.provider_fallback_used = fallback_used
        resp.fallback_chain = list(chain)
        return resp
