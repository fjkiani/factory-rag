"""Smoke tests for the HTML chat UI mounted at GET /.

These do not exercise the JS — they only assert the server returns a well-formed
HTML page with all the wiring the frontend code depends on. If the template ever
drifts away from the API contract (catalog/health endpoints, /chat fetch shape),
these tests will tell us.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_index_returns_html_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "").lower()
    assert len(r.text) > 2000, "template appears truncated"


def test_index_has_model_picker_dropdowns(client):
    r = client.get("/")
    body = r.text
    # Two dropdowns: answer LLM + judge LLM.
    assert 'id="llmSelect"' in body, "answer-model dropdown missing"
    assert 'id="judgeSelect"' in body, "judge-model dropdown missing"


def test_index_has_chat_form(client):
    r = client.get("/")
    body = r.text
    assert 'id="chatForm"' in body
    assert 'id="q"' in body, "query input missing"
    assert 'id="sendBtn"' in body, "send button missing"


def test_index_wires_to_backend_endpoints(client):
    r = client.get("/")
    body = r.text
    # All three endpoints the UI relies on must be referenced.
    assert "fetch('/chat'" in body
    assert "fetch('/providers/catalog'" in body
    assert "fetch('/providers/health'" in body


def test_index_has_warning_banner_area(client):
    r = client.get("/")
    body = r.text
    assert 'id="bannerRow"' in body, "warning banner row missing"
    # Make sure we wire up the chat-warning render path.
    assert "renderChatWarnings" in body
    # And the persistent (health-poll) banner path.
    assert "renderHealthBanners" in body


def test_index_uses_app_title_and_version(client):
    r = client.get("/")
    body = r.text
    # The template injects title + app version into the header.
    from app.main import app
    assert app.version in body
    # And the app/title default.
    assert "factory-rag" in body or "manufacturing" in body


def test_index_renders_citation_chip_logic(client):
    """The body parser converts [CHUNK_ID] tokens into clickable pills."""
    r = client.get("/")
    body = r.text
    assert "renderBodyWithCitations" in body
    assert "cite-pill" in body, "citation pill CSS class missing"
    # The chunk-id regex must be present so we don't accidentally drop it on
    # a future refactor.
    assert "A-Z0-9" in body, "citation regex missing"


def test_index_renders_judge_chip_branch(client):
    """The judge chip has an errored-neutral branch — make sure both exist."""
    r = client.get("/")
    body = r.text
    assert "judgeChip" in body
    assert "judge: neutral (errored)" in body
    assert "grounded" in body
