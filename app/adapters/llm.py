"""LLM adapters.

Two concrete providers (OpenAI-compatible chat completions API):
- OpenRouterLLM
- GroqLLM

Both share a common HTTP path via _ChatCompletionsLLM. Each returns LLMResponse
with the provider tagged so callers can attribute usage and surface health.

`MultiProviderLLM` (in app/adapters/providers.py) wraps several of these and
falls through on retryable upstream errors. It also records per-provider
health events on a HealthMonitor instance so the frontend can warn the user.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    raw: dict
    provider: str = "unknown"
    # Set by MultiProviderLLM when the primary failed and we fell through.
    provider_fallback_used: bool = False
    # Ordered list of providers we tried, e.g. ["openrouter", "groq"].
    fallback_chain: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    name: str
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
    ) -> LLMResponse: ...


class ProviderError(RuntimeError):
    """LLM provider error. `retryable` lets the failover layer decide whether
    to try the next provider or fail fast (4xx caller errors)."""

    def __init__(self, message: str, *, provider: str, status_code: int, retryable: bool):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class _ChatCompletionsLLM:
    """Shared HTTP plumbing for any OpenAI-compatible /chat/completions endpoint.

    Subclasses set: name, base_url, default_model, and supply request headers
    via `_extra_headers()`. They can also override `_munge_messages()` for
    provider-specific quirks (Groq's JSON-mode requirement).
    """

    name: str = "generic"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s

    # --- hooks subclasses can override ---
    def _extra_headers(self) -> dict[str, str]:
        return {}

    def _munge_messages(
        self, messages: list[dict], *, response_format_json: bool
    ) -> list[dict]:
        return messages

    # --- main entry ---
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model: Optional[str] = None,
        response_format_json: bool = False,
        pinned_provider: Optional[str] = None,  # accepted for protocol parity; ignored
    ) -> LLMResponse:
        if not self.api_key:
            raise ProviderError(
                f"{self.name} API key not set",
                provider=self.name,
                status_code=0,
                retryable=False,
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        messages = self._munge_messages(messages, response_format_json=response_format_json)
        payload: dict = {
            "model": model or self.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self._extra_headers(),
        }
        last_error: Optional[str] = None
        last_status = 0
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(self.max_retries + 1):
                resp = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                last_status = resp.status_code
                if resp.status_code < 400:
                    break
                retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
                last_error = (
                    f"{self.name} {resp.status_code} for model={payload['model']!r}: "
                    f"{resp.text[:600]}"
                )
                if attempt < self.max_retries and retryable:
                    sleep_s = self.retry_base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
                    continue
                # Out of retries OR non-retryable 4xx
                raise ProviderError(
                    last_error,
                    provider=self.name,
                    status_code=resp.status_code,
                    retryable=retryable,
                )
            data = resp.json()
        # Some upstream errors come back as 200 with {"error": {...}} and no "choices"
        if "error" in data and "choices" not in data:
            err = data["error"]
            raise ProviderError(
                f"{self.name} error for model={payload['model']!r}: "
                f"code={err.get('code')} message={err.get('message')!r}",
                provider=self.name,
                status_code=200,
                retryable=False,
            )
        if "choices" not in data or not data["choices"]:
            raise ProviderError(
                f"{self.name} returned no choices for model={payload['model']!r}: "
                f"keys={list(data.keys())} body={str(data)[:400]}",
                provider=self.name,
                status_code=200,
                retryable=False,
            )
        choice = data["choices"][0]
        # gpt-oss models occasionally emit content=None when the answer is in reasoning_details
        # (because output landed in the reasoning channel only); coalesce to "".
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", payload["model"]),
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )


class OpenRouterLLM(_ChatCompletionsLLM):
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
        http_referer: str = "https://github.com/fjkiani/factory-rag",
        app_title: str = "factory-rag-mvp",
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
        )
        self.http_referer = http_referer
        self.app_title = app_title

    def _extra_headers(self) -> dict[str, str]:
        # OpenRouter requires these for free-tier rate-limit allocation.
        return {
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_title,
        }


class GroqLLM(_ChatCompletionsLLM):
    """Groq's API is OpenAI-compatible. One quirk: when response_format=json_object
    is set, the word 'json' MUST appear in at least one message. We inject a
    minimal hint into the system message if it's not already present."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
        )

    def _munge_messages(
        self, messages: list[dict], *, response_format_json: bool
    ) -> list[dict]:
        if not response_format_json:
            return messages
        joined = " ".join((m.get("content") or "") for m in messages).lower()
        if "json" in joined:
            return messages
        # Inject a minimal hint; we mutate a copy so we don't surprise the caller.
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "system":
                m["content"] = (m.get("content") or "") + "\n\nReturn JSON."
                return out
        # No system message — prepend one.
        return [{"role": "system", "content": "Return JSON."}, *out]


def parse_json_strict(text: str) -> dict:
    """Robust JSON extraction. Tolerates:
      - code fences (```json ... ```)
      - prefix/suffix prose
      - gpt-oss-style malformed prefix followed by long whitespace and the
        actual JSON later in the same content field (we scan for the LARGEST
        balanced top-level {...} block).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM output (likely model returned reasoning_only).")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Fast path: full string is JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Walk every '{' position. For each, try to consume the longest valid JSON
    # object with json.JSONDecoder.raw_decode. Keep the LARGEST successful parse
    # by consumed chars. This handles gpt-oss-120b's pattern of a malformed
    # stub first and the real JSON later, AND avoids returning a nested
    # sub-object when the outer object also parses.
    starts = [i for i, c in enumerate(text) if c == "{"]
    decoder = json.JSONDecoder()
    best: tuple[int, dict] | None = None  # (consumed_chars, obj)
    for s in starts:
        try:
            obj, end = decoder.raw_decode(text[s:])
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if best is None or end > best[0]:
            best = (end, obj)
    if best is not None:
        return best[1]
    raise ValueError(f"Could not parse JSON from LLM output: {text[:300]!r}")
