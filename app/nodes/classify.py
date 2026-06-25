"""Classify node: route a query to exactly one of {safety, maintenance, quality}
or 'none' if it doesn't fit. Deterministic JSON output.
"""
from __future__ import annotations

import time
from typing import Callable

from ..adapters.llm import LLMClient, parse_json_strict
from ..state import AgentState, Domain

SYSTEM = """You classify a manufacturing-floor question into exactly one of these domains:
- safety: lockout/tagout, PPE, hazards, emergency procedures
- maintenance: machine upkeep, preventive maintenance, fault codes, lubrication, repairs
- quality: inspection, sampling plans, tolerances, non-conformance, ISO procedures

Return STRICT JSON with this shape and nothing else:
{"route": "safety|maintenance|quality|none", "confidence": 0.0-1.0, "reason": "one short sentence"}

If the question clearly does not fit any of the three domains, return route="none" and confidence=0.0.
"""


def make_classify_node(llm: LLMClient, *, model: str | None = None) -> Callable[[AgentState], AgentState]:
    def classify(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        resp = llm.complete(
            system=SYSTEM,
            user=state["query"],
            temperature=0.0,
            max_tokens=120,
            model=model,
            response_format_json=True,
        )
        try:
            obj = parse_json_strict(resp.text)
            route = str(obj.get("route", "none")).lower()
            if route not in {"safety", "maintenance", "quality", "none"}:
                route = "none"
            confidence = float(obj.get("confidence", 0.0))
            reason = str(obj.get("reason", ""))[:300]
        except Exception as e:
            route, confidence, reason = "none", 0.0, f"parse_error: {e}"
        latency = int((time.perf_counter() - t0) * 1000)
        state["route"] = route  # type: ignore[typeddict-item]
        state["route_confidence"] = max(0.0, min(1.0, confidence))
        state["route_reason"] = reason
        state.setdefault("latency_ms", {})["classify"] = latency
        return state

    return classify
