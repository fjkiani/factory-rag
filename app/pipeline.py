"""Pure-function state pipeline. Composed linearly; each node is reusable
verbatim inside a LangGraph StateGraph later (no business-logic changes).
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from .state import AgentState


def make_pipeline(nodes: list[Callable[[AgentState], AgentState]]) -> Callable[[str, str | None], AgentState]:
    def run(query: str, session_id: str | None = None) -> AgentState:
        state: AgentState = {
            "trace_id": uuid.uuid4().hex,
            "query": query,
            "session_id": session_id,
            "latency_ms": {},
        }
        t0 = time.perf_counter()
        for node in nodes:
            state = node(state)
        state.setdefault("latency_ms", {})["total"] = int((time.perf_counter() - t0) * 1000)
        return state

    return run
