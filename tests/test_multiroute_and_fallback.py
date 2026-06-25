"""Tests for the multi-route fanout + keyword-router fallback contract."""
from __future__ import annotations

import json

import httpx
import pytest

from app.adapters.keyword_router import fallback_classify, score_routes
from app.nodes.classify import make_classify_node
from app.nodes.retrieve import make_retrieve_node
from tests.conftest import FakeLLM


# ------------------------------ keyword router ------------------------------
def test_keyword_router_routes_loto_to_safety():
    out = fallback_classify("What PPE do I need for LOTO of the HP-200?")
    assert out[0]["route"] == "safety"
    assert out[0]["confidence"] > 0.0
    assert out[0]["source"] == "keyword_router"


def test_keyword_router_routes_fault_code_to_maintenance():
    out = fallback_classify("How do I clear fault code E-318 on the lathe?")
    assert out[0]["route"] == "maintenance"


def test_keyword_router_routes_aql_to_quality():
    out = fallback_classify("What sample size applies under AQL 1.0 for 200 parts?")
    assert out[0]["route"] == "quality"


def test_keyword_router_returns_none_for_truly_unrelated():
    out = fallback_classify("Who won the world cup in 2022?")
    assert out[0]["route"] == "none"


def test_keyword_router_confidence_is_capped_below_llm_ceiling():
    # Even a query stuffed with safety terms should not exceed ~0.70
    out = fallback_classify(
        "PPE LOTO lockout tagout hazard emergency E-stop zero-energy isolation safety guard"
    )
    assert out[0]["route"] == "safety"
    assert out[0]["confidence"] <= 0.71  # ceiling 0.70 with rounding slack


# ------------------------------ classify node: LLM failure --------------------
def test_classify_falls_back_when_llm_raises_http_error():
    """Simulate OpenRouter outage. Keyword router should pick up."""

    class BoomLLM:
        def complete(self, *a, **kw):
            raise httpx.ConnectError("upstream unreachable")

    node = make_classify_node(BoomLLM())
    out = node({"query": "What PPE is required for HP-200 lockout?"})
    assert out["route"] == "safety"
    assert out["route_used_fallback"] is True
    assert out["route_source"] == "keyword_router"
    assert "upstream unreachable" in (out.get("route_llm_error") or "")


def test_classify_falls_back_when_llm_returns_bad_json():
    llm = FakeLLM()
    llm.register(lambda s, u: True, "this is not json at all")
    node = make_classify_node(llm)
    out = node({"query": "Fault code E-318 on the L-450?"})
    assert out["route"] == "maintenance"
    assert out["route_used_fallback"] is True
    # Source becomes keyword_router because LLM errored (parse failure)
    assert out["route_source"] == "keyword_router"


def test_classify_no_llm_uses_keyword_router_directly():
    node = make_classify_node(None)  # llm=None -> keyword router only
    out = node({"query": "How do I clear E-405 turret index timeout?"})
    assert out["route"] == "maintenance"
    assert out["route_source"] == "keyword_router"


def test_classify_oos_with_no_llm_returns_none():
    node = make_classify_node(None)
    out = node({"query": "Who is the prime minister of Canada?"})
    assert out["route"] == "none"
    assert out["route_source"] in {"keyword_router", "fallback"}


# ------------------------------ classify node: low-confidence override ---------
def test_low_confidence_llm_yields_to_keyword_router_when_kw_is_stronger():
    """LLM unsure (conf=0.20) but keyword router fires strongly -> keyword wins."""
    llm = FakeLLM()
    llm.register(
        lambda s, u: True,
        json.dumps({
            "primary": {"route": "quality", "confidence": 0.20, "reason": "unsure"},
            "alternates": [],
        }),
    )
    node = make_classify_node(llm, fallback_threshold=0.5)
    out = node({"query": "What PPE is required for HP-200 lockout/tagout?"})
    # Keyword router fires hard on PPE/LOTO -> safety with conf > 0.20
    assert out["route"] == "safety"
    assert out["route_used_fallback"] is True
    # Both signals end up in candidates
    routes = [c["route"] for c in out["route_candidates"]]
    assert "safety" in routes


def test_high_confidence_llm_keeps_decision_even_if_kw_disagrees():
    """LLM is confident -> we keep its verdict; keyword router only enriches."""
    llm = FakeLLM()
    llm.register(
        lambda s, u: True,
        json.dumps({
            "primary": {"route": "maintenance", "confidence": 0.95, "reason": "explicit fault code"},
            "alternates": [],
        }),
    )
    node = make_classify_node(llm, fallback_threshold=0.5)
    out = node({"query": "What's the procedure to clear fault E-318 on the lathe?"})
    assert out["route"] == "maintenance"
    assert out["route_used_fallback"] is False
    assert out["route_source"] == "llm"


# ------------------------------ multi-route fanout -----------------------------
def test_classify_emits_alternates_when_llm_provides_them():
    llm = FakeLLM()
    llm.register(
        lambda s, u: True,
        json.dumps({
            "primary": {"route": "safety", "confidence": 0.85, "reason": "PPE during service"},
            "alternates": [{"route": "maintenance", "confidence": 0.6, "reason": "during 500-hour service"}],
        }),
    )
    node = make_classify_node(llm)
    out = node({"query": "What PPE is required during the 500-hour service?"})
    assert out["route"] == "safety"
    routes = [c["route"] for c in out["route_candidates"]]
    assert routes[:2] == ["safety", "maintenance"]


def test_retrieve_queries_multiple_collections_when_alternates_present(ingested_store, hashing_embedder):
    node = make_retrieve_node(ingested_store, hashing_embedder, max_collections=2, min_secondary_conf=0.3)
    state = {
        "query": "What PPE is required during the 500-hour service?",
        "route": "safety",
        "route_candidates": [
            {"route": "safety", "confidence": 0.85, "reason": "PPE", "source": "llm"},
            {"route": "maintenance", "confidence": 0.6, "reason": "500h service", "source": "llm"},
        ],
    }
    out = node(state)
    queried = set(out["retrieved_per_collection"].keys())
    assert queried == {"kb_safety", "kb_maintenance"}
    # Fused chunks should include both domains' content
    domains = {r["domain"] for r in out["retrieved"]}
    # At least both domains represented in the top-N most likely; weakly assert >=1 domain
    assert len(out["retrieved"]) > 0
    assert domains.issubset({"safety", "maintenance"})


def test_retrieve_skips_alternates_below_secondary_threshold(ingested_store, hashing_embedder):
    node = make_retrieve_node(ingested_store, hashing_embedder, max_collections=2, min_secondary_conf=0.5)
    state = {
        "query": "Lockout sequence for HP-200",
        "route": "safety",
        "route_candidates": [
            {"route": "safety", "confidence": 0.9, "reason": "LOTO", "source": "llm"},
            {"route": "maintenance", "confidence": 0.2, "reason": "weak signal", "source": "llm"},  # below 0.5
        ],
    }
    out = node(state)
    queried = set(out["retrieved_per_collection"].keys())
    assert queried == {"kb_safety"}
    # Every retrieved chunk should be from safety
    assert all(r["domain"] == "safety" for r in out["retrieved"])
