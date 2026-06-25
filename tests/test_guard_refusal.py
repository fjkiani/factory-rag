"""Guard + post-generation refusal contracts."""
from __future__ import annotations

import json

from app.nodes.guard import make_guard_node
from app.nodes.generate import make_generate_node
from tests.conftest import FakeLLM


def test_guard_refuses_out_of_scope():
    guard = make_guard_node(route_conf_threshold=0.5, retrieval_conf_threshold=0.35)
    state = {"route": "none", "route_confidence": 0.0, "retrieval_confidence": 0.0, "retrieved": []}
    out = guard(state)
    assert out["refused"] is True
    assert out["refusal_reason"] == "out_of_scope"
    assert out["citations"] == []


def test_guard_refuses_low_retrieval_confidence():
    guard = make_guard_node(route_conf_threshold=0.5, retrieval_conf_threshold=0.35)
    state = {
        "route": "safety",
        "route_confidence": 0.9,
        "retrieval_confidence": 0.10,  # well below threshold
        "retrieved": [
            {"chunk_id": "SAFETY-LOTO-001#1.0", "rrf_score": 0.001, "doc_id": "x", "doc_title": "t",
             "domain": "safety", "section_id": "1.0", "heading": "h", "body": "b",
             "dense_score": 0.0, "sparse_score": 0.0}
        ],
    }
    out = guard(state)
    assert out["refused"] is True
    assert out["refusal_reason"] == "low_confidence"


def test_generate_refuses_fabricated_citation():
    """LLM cites a chunk_id that wasn't retrieved -> refuse."""
    llm = FakeLLM()
    llm.register(lambda s, u: True,
                 lambda s, u: "Step 1: do thing [GHOST-DOC#9.9].")
    gen = make_generate_node(llm)
    state = {
        "query": "anything",
        "route": "safety",
        "refused": False,
        "retrieved": [
            {"chunk_id": "SAFETY-LOTO-001#3.0", "doc_id": "SAFETY-LOTO-001", "doc_title": "t",
             "domain": "safety", "section_id": "3.0", "heading": "Six-Step", "body": "body",
             "dense_score": 1.0, "sparse_score": 1.0, "rrf_score": 0.03}
        ],
    }
    out = gen(state)
    assert out["refused"] is True
    assert out["refusal_reason"] == "fabricated_citation"
    assert out["citations"] == []


def test_generate_refuses_uncited_answer():
    llm = FakeLLM()
    llm.register(lambda s, u: True, lambda s, u: "Just do it; trust me.")
    gen = make_generate_node(llm)
    state = {
        "query": "anything",
        "route": "safety",
        "refused": False,
        "retrieved": [
            {"chunk_id": "SAFETY-LOTO-001#3.0", "doc_id": "SAFETY-LOTO-001", "doc_title": "t",
             "domain": "safety", "section_id": "3.0", "heading": "Six-Step", "body": "body",
             "dense_score": 1.0, "sparse_score": 1.0, "rrf_score": 0.03}
        ],
    }
    out = gen(state)
    assert out["refused"] is True
    assert out["refusal_reason"] == "uncited_answer"


def test_generate_accepts_valid_citation():
    llm = FakeLLM()
    llm.register(lambda s, u: True,
                 lambda s, u: "(1) Notify employees [SAFETY-LOTO-001#3.0]. (2) Shut down [SAFETY-LOTO-001#3.0].")
    gen = make_generate_node(llm)
    state = {
        "query": "lockout steps",
        "route": "safety",
        "refused": False,
        "retrieved": [
            {"chunk_id": "SAFETY-LOTO-001#3.0", "doc_id": "SAFETY-LOTO-001", "doc_title": "t",
             "domain": "safety", "section_id": "3.0", "heading": "Six-Step", "body": "body",
             "dense_score": 1.0, "sparse_score": 1.0, "rrf_score": 0.03}
        ],
    }
    out = gen(state)
    assert out["refused"] is False
    assert "SAFETY-LOTO-001#3.0" in out["citations"]
