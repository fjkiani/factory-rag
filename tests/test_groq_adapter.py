"""Unit tests for the Groq LLM adapter. No network — httpx is mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.adapters.llm import GroqLLM, ProviderError


def _make_ok_response(content: str = "ok", model: str = "llama-3.3-70b-versatile"):
    """Build a stub httpx.Response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content, "role": "assistant"}}],
        "model": model,
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    resp.text = "200 ok"
    return resp


def _make_err_response(status: int, body: dict | str):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body if isinstance(body, str) else json.dumps(body)
    resp.json.return_value = body if isinstance(body, dict) else {}
    return resp


@pytest.fixture
def llm():
    return GroqLLM(
        api_key="fake-key",
        default_model="llama-3.3-70b-versatile",
        max_retries=2,
        retry_base_delay_s=0.0,  # don't sleep in tests
    )


def test_groq_sends_bearer_auth(llm):
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_ok_response("hi")
        llm.complete("sys", "user")
        call_kwargs = instance.post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer fake-key"
        assert "Content-Type" in call_kwargs["headers"]


def test_groq_json_mode_injects_word_json_into_system_when_absent(llm):
    """Groq returns 400 if messages don't contain 'json' when response_format=json_object.
    We inject it into the system message."""
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_ok_response('{"a": 1}')
        llm.complete("classify the route", "What PPE?", response_format_json=True)
        payload = instance.post.call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}
        system_msg = next(m for m in payload["messages"] if m["role"] == "system")
        assert "json" in system_msg["content"].lower()


def test_groq_json_mode_does_not_inject_when_word_already_present(llm):
    """If the system message already contains 'json' we leave it alone."""
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_ok_response('{"a": 1}')
        llm.complete("Return JSON with {route, confidence}", "Q", response_format_json=True)
        payload = instance.post.call_args.kwargs["json"]
        sys_msg = next(m for m in payload["messages"] if m["role"] == "system")
        # The original text is preserved (no duplicated 'Return JSON.' appended).
        assert sys_msg["content"] == "Return JSON with {route, confidence}"


def test_groq_no_json_injection_when_not_json_mode(llm):
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_ok_response("free text")
        llm.complete("plain prompt", "Q", response_format_json=False)
        payload = instance.post.call_args.kwargs["json"]
        assert "response_format" not in payload
        sys_msg = next(m for m in payload["messages"] if m["role"] == "system")
        assert sys_msg["content"] == "plain prompt"


def test_groq_400_is_non_retryable_provider_error(llm):
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_err_response(400, {"error": {"message": "bad json"}})
        with pytest.raises(ProviderError) as excinfo:
            llm.complete("sys", "user")
        assert excinfo.value.status_code == 400
        assert excinfo.value.retryable is False
        assert excinfo.value.provider == "groq"
        # We must NOT retry on 400.
        assert instance.post.call_count == 1


def test_groq_429_is_retryable_then_exhausts(llm):
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = _make_err_response(429, "rate limit")
        with pytest.raises(ProviderError) as excinfo:
            llm.complete("sys", "user")
        assert excinfo.value.status_code == 429
        assert excinfo.value.retryable is True
        # max_retries=2 means 3 total attempts.
        assert instance.post.call_count == 3


def test_groq_500_then_recovery(llm):
    """5xx triggers a retry; if a later attempt succeeds, we should get the response."""
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.side_effect = [
            _make_err_response(500, "server boom"),
            _make_ok_response("recovered"),
        ]
        out = llm.complete("sys", "user")
        assert out.text == "recovered"
        assert out.provider == "groq"
        assert instance.post.call_count == 2


def test_groq_200_with_error_key_no_choices(llm):
    """Upstream sometimes returns 200 with {'error': ...} and no 'choices'."""
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"error": {"code": "invalid", "message": "no model"}}
        resp.text = "stub"
        instance.post.return_value = resp
        with pytest.raises(ProviderError) as excinfo:
            llm.complete("sys", "user")
        assert excinfo.value.provider == "groq"
        assert excinfo.value.retryable is False
