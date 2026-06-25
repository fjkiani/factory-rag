"""Guard node: strict cite-or-refuse gate BEFORE generation.

Refusal reasons handled here:
- out_of_scope: route='none' OR route_confidence < ROUTE_CONF_THRESHOLD
- low_confidence: retrieval_confidence < RETRIEVAL_CONF_THRESHOLD OR < min_chunks above floor
"""
from __future__ import annotations

import time
from typing import Callable

from ..state import AgentState

REFUSAL_TEMPLATES = {
    "out_of_scope": "I don't have that in the manufacturing documentation (Safety, Maintenance, or Quality).",
    "low_confidence": "I don't have a confident answer in the {domain} documentation for that question.",
}


def make_guard_node(
    *,
    route_conf_threshold: float,
    retrieval_conf_threshold: float,
    min_chunks: int = 1,
) -> Callable[[AgentState], AgentState]:
    def guard(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        route = state.get("route", "none")
        route_conf = state.get("route_confidence", 0.0)
        ret_conf = state.get("retrieval_confidence", 0.0)
        retrieved = state.get("retrieved", []) or []

        refused = False
        reason: str | None = None
        answer = ""

        if route == "none" or route_conf < route_conf_threshold:
            refused = True
            reason = "out_of_scope"
            answer = REFUSAL_TEMPLATES["out_of_scope"]
        elif len(retrieved) < min_chunks or ret_conf < retrieval_conf_threshold:
            refused = True
            reason = "low_confidence"
            answer = REFUSAL_TEMPLATES["low_confidence"].format(domain=route)

        state["refused"] = refused
        state["refusal_reason"] = reason
        if refused:
            state["answer"] = answer
            state["citations"] = []
        state.setdefault("latency_ms", {})["guard"] = int((time.perf_counter() - t0) * 1000)
        return state

    return guard
