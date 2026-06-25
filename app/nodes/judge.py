"""LLM-as-judge (online). Single cheap call. Verdict gates the final response:
if grounded=false, downgrade to refusal `judge_ungrounded`.
"""
from __future__ import annotations

import time
from typing import Callable

from ..adapters.llm import LLMClient, parse_json_strict
from ..state import AgentState, JudgeVerdict

SYSTEM = """You evaluate a manufacturing-assistant response. Be terse.

Inputs you will receive:
- The user question
- The chosen route (safety|maintenance|quality|none)
- The retrieved context chunks (with chunk ids)
- The candidate answer

Score the answer with STRICT JSON of this shape:
{
  "grounded": true|false,    // every non-trivial claim is supported by at least one provided chunk
  "routing_ok": true|false,  // the route is appropriate for the question (true if route is 'none' AND the question is genuinely out of scope)
  "score": 0.0-1.0,
  "reasons": ["short reasons"]
}
If the answer is the canonical refusal because the question is out of scope or low confidence, grounded=true and score>=0.8 if appropriate.
Output ONLY the JSON object.
"""


def _format_chunks(retrieved: list[dict]) -> str:
    lines = []
    for r in retrieved:
        lines.append(f"[{r['chunk_id']}] ({r.get('heading','')})\n{r.get('body','')}\n")
    return "\n".join(lines) if lines else "(no chunks)"


def make_judge_node(llm: LLMClient, *, model: str | None = None) -> Callable[[AgentState], AgentState]:
    def judge(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        retrieved = state.get("retrieved", []) or []
        user = (
            f"Question: {state.get('query','')}\n"
            f"Route: {state.get('route','none')}\n"
            f"Refused: {bool(state.get('refused'))}\n"
            f"Refusal reason: {state.get('refusal_reason')}\n"
            f"Chunks:\n{_format_chunks(retrieved)}\n"
            f"Answer:\n{state.get('answer','')}\n"
        )
        try:
            resp = llm.complete(
                system=SYSTEM,
                user=user,
                temperature=0.0,
                max_tokens=200,
                model=model,
                response_format_json=True,
            )
            obj = parse_json_strict(resp.text)
            verdict: JudgeVerdict = {
                "grounded": bool(obj.get("grounded", False)),
                "routing_ok": bool(obj.get("routing_ok", False)),
                "score": float(obj.get("score", 0.0)),
                "reasons": [str(x)[:200] for x in (obj.get("reasons") or [])][:5],
                "model": resp.model,
            }
        except Exception as e:
            verdict = {
                "grounded": False,
                "routing_ok": False,
                "score": 0.0,
                "reasons": [f"judge_parse_error: {e}"],
                "model": model or "",
            }

        # Online gate: if a non-refused answer is flagged ungrounded, downgrade.
        if not state.get("refused") and not verdict["grounded"]:
            state["refused"] = True
            state["refusal_reason"] = "judge_ungrounded"
            domain = state.get("route", "")
            state["answer"] = (
                f"I can't confidently ground that answer in the {domain} documentation."
            )
            state["citations"] = []

        state["judge"] = verdict
        state.setdefault("latency_ms", {})["judge"] = int((time.perf_counter() - t0) * 1000)
        return state

    return judge
