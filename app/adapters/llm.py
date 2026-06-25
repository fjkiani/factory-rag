"""LLM adapter over OpenRouter. One Protocol, one default impl, one registry.

Swapping models is an env var change. Swapping providers is a new class
that implements LLMClient.
"""
from __future__ import annotations

import json
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
    def __init__(self, api_key: str, base_url: str, default_model: str, timeout_s: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s

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
            # Recommended by OpenRouter for analytics, harmless if absent
            "HTTP-Referer": "https://rag-mvp.local",
            "X-Title": "rag-mvp",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )


def parse_json_strict(text: str) -> dict:
    """Robust JSON extraction: tolerates code fences and prefix/suffix prose."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # Try direct first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: extract the largest {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")
