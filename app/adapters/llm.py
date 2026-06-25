"""LLM adapter over OpenRouter. One Protocol, one default impl, one registry.

Swapping models is an env var change. Swapping providers is a new class
that implements LLMClient.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    raw: dict


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model: Optional[str] = None,
        response_format_json: bool = False,
    ) -> LLMResponse: ...


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_s: float = 60.0,
        http_referer: str = "https://github.com/fjkiani/factory-rag",
        app_title: str = "factory-rag-mvp",
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.http_referer = http_referer
        self.app_title = app_title
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model: Optional[str] = None,
        response_format_json: bool = False,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        payload: dict = {
            "model": model or self.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Required by OpenRouter to allocate free-tier rate-limit budget
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_title,
        }
        # Retry transient 429/5xx with exponential backoff + jitter. Don't retry
        # 4xx that are caller errors (400/401/402/403/404).
        last_error: Optional[str] = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(self.max_retries + 1):
                resp = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                if resp.status_code < 400:
                    break
                # Retryable?
                retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
                last_error = (
                    f"OpenRouter {resp.status_code} for model={payload['model']!r}: "
                    f"{resp.text[:600]}"
                )
                if attempt < self.max_retries and retryable:
                    sleep_s = self.retry_base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
                    continue
                # Non-retryable or out of retries
                raise RuntimeError(last_error)
            data = resp.json()
        # Some upstream errors come back as 200 with {"error": {...}} and no "choices"
        if "error" in data and "choices" not in data:
            err = data["error"]
            raise RuntimeError(
                f"OpenRouter error for model={payload['model']!r}: "
                f"code={err.get('code')} message={err.get('message')!r}"
            )
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(
                f"OpenRouter returned no choices for model={payload['model']!r}: "
                f"keys={list(data.keys())} body={str(data)[:400]}"
            )
        choice = data["choices"][0]
        # gpt-oss models occasionally emit content=None when the answer is in reasoning_details
        # (because output landed in the reasoning channel only); coalesce to "".
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )


def parse_json_strict(text: str) -> dict:
    """Robust JSON extraction. Tolerates:
      - code fences (```json ... ```)
      - prefix/suffix prose
      - gpt-oss-style malformed prefix followed by long whitespace and the
        actual JSON later in the same content field (we scan for the LAST
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
    # object with json.JSONDecoder.raw_decode. Keep the LARGEST (most fields,
    # then longest by chars) successful parse. This handles gpt-oss-120b's
    # pattern of a malformed stub first and the real JSON later, AND avoids
    # returning a nested sub-object when the outer object also parses.
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
