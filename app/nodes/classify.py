"""Classify node: intent routing for the floor-supervisor RAG.

Behavior:
1. Ask the LLM for a RANKED list of plausible routes
   (primary + optional secondary), with per-route confidence.
2. If the LLM call fails (network, 5xx, parse error) OR the primary route's
   confidence is below `fallback_threshold`, invoke the deterministic
   keyword router as a backup. Both sources contribute to telemetry.
3. Output:
   - state["route"]: the primary route name
   - state["route_confidence"]: confidence of the primary route
   - state["route_candidates"]: ranked list of {route, confidence, reason, source}
                                used downstream for multi-route fanout

Tradeoffs:
- The keyword router is intentionally calibrated lower than LLM confidence
  (max ~0.70) so a working LLM signal wins. When the LLM is down or unsure,
  we still get a deterministic, observable decision instead of a hard failure.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from ..adapters.keyword_router import fallback_classify
from ..adapters.llm import LLMClient, parse_json_strict
from ..state import AgentState

SYSTEM = """You classify a manufacturing-floor question into one or more of these domains:
- safety: lockout/tagout (LOTO), PPE, hazards, emergency stop, energy isolation, guarding
- maintenance: machine upkeep, preventive maintenance, fault codes, lubrication, calibration, repairs
- quality: inspection, sampling plans (AQL), GD&T, tolerances, surface finish, non-conformance, ISO 9001

Return STRICT JSON with this shape and nothing else:
{
  "primary": {"route": "safety|maintenance|quality|none", "confidence": 0.0-1.0, "reason": "one short sentence"},
  "alternates": [
    {"route": "safety|maintenance|quality", "confidence": 0.0-1.0, "reason": "..."}
  ]
}

Rules:
- "alternates" lists OTHER domains that meaningfully apply (max 2). Use it ONLY when the question genuinely spans domains.
  Example: "What PPE is required during the 500-hour service?" => primary=safety, alternates=[maintenance]
- If the question clearly fits exactly one domain, "alternates" must be [].
- If the question clearly does not fit any domain, primary.route="none", primary.confidence=0.0, alternates=[].
- Never invent a domain not in the list.
"""


def _merge_candidates(llm_ranked: list[dict], kw_ranked: list[dict], max_keep: int = 3) -> list[dict]:
    """Merge LLM and keyword candidates by domain. Preserve LLM ordering and
    confidence where present; append keyword routes not already covered.
    """
    by_route: dict[str, dict] = {}
    order: list[str] = []
    for c in llm_ranked:
        r = c.get("route")
        if r and r != "none" and r not in by_route:
            by_route[r] = {**c, "source": "llm"}
            order.append(r)
    for c in kw_ranked:
        r = c.get("route")
        if r and r != "none":
            if r in by_route:
                by_route[r] = {
                    **by_route[r],
                    "keyword_confidence": c.get("confidence"),
                    "keyword_matched": c.get("matched", []),
                    "source": "llm+keyword_router",
                }
            else:
                by_route[r] = {**c, "source": "keyword_router"}
                order.append(r)
    return [by_route[r] for r in order[:max_keep]]


def make_classify_node(
    llm: Optional[LLMClient],
    *,
    model: str | None = None,
    fallback_threshold: float = 0.5,
    enable_keyword_fallback: bool = True,
) -> Callable[[AgentState], AgentState]:
    """Build the classify node.

    Args:
        llm: LLMClient or None. If None, we go straight to the keyword router.
        model: optional override model id passed to the LLM adapter.
        fallback_threshold: if the LLM primary route's confidence is below
            this AND the keyword router agrees on a different route with at
            least equal confidence, we trust the keyword router for the primary.
            Otherwise we keep the LLM's verdict but still append the keyword
            router's top candidate to `route_candidates`.
        enable_keyword_fallback: turn off only for tests that assert pure LLM behavior.
    """

    def classify(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        query = state["query"]

        llm_primary: dict | None = None
        llm_alternates: list[dict] = []
        llm_error: str | None = None

        if llm is not None:
            choice = state.get("llm_choice") or {}
            try:
                resp = llm.complete(
                    system=SYSTEM,
                    user=query,
                    temperature=0.0,
                    max_tokens=220,
                    model=choice.get("model") or model,
                    response_format_json=True,
                    pinned_provider=choice.get("provider"),
                )
                obj = parse_json_strict(resp.text)
                p = obj.get("primary") or {}
                route = str(p.get("route", "none")).lower()
                if route not in {"safety", "maintenance", "quality", "none"}:
                    route = "none"
                llm_primary = {
                    "route": route,
                    "confidence": max(0.0, min(1.0, float(p.get("confidence", 0.0)))),
                    "reason": str(p.get("reason", ""))[:300],
                    "source": "llm",
                }
                for a in obj.get("alternates") or []:
                    ar = str(a.get("route", "")).lower()
                    if ar in {"safety", "maintenance", "quality"} and ar != route:
                        llm_alternates.append(
                            {
                                "route": ar,
                                "confidence": max(0.0, min(1.0, float(a.get("confidence", 0.0)))),
                                "reason": str(a.get("reason", ""))[:300],
                                "source": "llm",
                            }
                        )
                # Cap alternates at 2 to bound downstream fanout
                llm_alternates = llm_alternates[:2]
            except httpx.HTTPError as e:
                llm_error = f"http_error: {type(e).__name__}: {str(e)[:200]}"
            except Exception as e:
                llm_error = f"{type(e).__name__}: {str(e)[:200]}"

        # Decide whether to invoke the keyword fallback.
        kw_ranked: list[dict] = []
        used_fallback = False
        llm_low_conf = (
            llm_primary is None
            or llm_primary["route"] == "none"
            or llm_primary["confidence"] < fallback_threshold
        )
        if enable_keyword_fallback and (llm_error is not None or llm_low_conf):
            kw_ranked = fallback_classify(query)
            used_fallback = True

        # Combine results.
        if llm_primary is not None and not (used_fallback and llm_primary["confidence"] == 0.0):
            # LLM verdict stands as primary; keyword router enriches candidates.
            llm_ranked = [llm_primary] + llm_alternates
            candidates = _merge_candidates(llm_ranked, kw_ranked) if kw_ranked else llm_ranked
            primary = candidates[0] if candidates else {"route": "none", "confidence": 0.0, "reason": "no_candidates", "source": "fallback"}
            # If LLM was low-conf AND keyword router has a route with strictly higher confidence,
            # swap primary to the keyword pick. This is the only case where keyword overrides LLM.
            if llm_low_conf and kw_ranked:
                kw_top = next((c for c in kw_ranked if c.get("route") != "none"), None)
                if kw_top and kw_top["confidence"] > primary["confidence"]:
                    # Reorder candidates to put kw_top first
                    candidates = [kw_top] + [c for c in candidates if c["route"] != kw_top["route"]]
                    primary = candidates[0]
        else:
            # LLM unavailable / errored — use keyword router as sole source.
            candidates = [c for c in kw_ranked if c.get("route") != "none"]
            if not candidates:
                primary = {
                    "route": "none",
                    "confidence": 0.0,
                    "reason": (llm_error or "no_route_identified"),
                    "source": "fallback",
                }
                candidates = [primary]
            else:
                primary = candidates[0]

        state["route"] = primary["route"]  # type: ignore[typeddict-item]
        state["route_confidence"] = float(primary["confidence"])
        state["route_reason"] = primary.get("reason", "")
        state["route_source"] = primary.get("source", "llm")
        state["route_candidates"] = candidates  # type: ignore[typeddict-item]
        state["route_llm_error"] = llm_error
        state["route_used_fallback"] = used_fallback

        state.setdefault("latency_ms", {})["classify"] = int((time.perf_counter() - t0) * 1000)
        return state

    return classify
