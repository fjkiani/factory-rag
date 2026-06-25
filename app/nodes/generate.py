"""Generate node: produce a cited answer or refuse.

Post-generation checks (strict):
- Every cited chunk_id must appear in the retrieved set; otherwise -> refuse
  with refusal_reason='fabricated_citation'.
- Non-refusal answers must contain >=1 citation; otherwise -> refuse with
  refusal_reason='uncited_answer'.
"""
from __future__ import annotations

import re
import time
from typing import Callable

from ..adapters.llm import LLMClient
from ..state import AgentState

# Accept ASCII [CHUNK] and CJK lenticular 【CHUNK】 (some free models emit these).
# We canonicalize to ASCII brackets internally — the model still SHOULD use [ ]
# per the system prompt, but accepting both prevents a wrong-bracket refusal.
_ASCII_CITE = r"\[([A-Z0-9][A-Z0-9_\-]+#\d+(?:\.\d+)*)\]"
_CJK_CITE = r"\u3010([A-Z0-9][A-Z0-9_\-]+#\d+(?:\.\d+)*)\u3011"
CITATION_RE = re.compile(f"(?:{_ASCII_CITE})|(?:{_CJK_CITE})")


def _extract_citations(text: str) -> tuple[list[str], str]:
    """Return (unique_chunk_ids_in_order, text_with_canonical_brackets)."""
    seen: list[str] = []
    def _sub(m: "re.Match[str]") -> str:
        cid = m.group(1) or m.group(2)
        if cid and cid not in seen:
            seen.append(cid)
        return f"[{cid}]"
    canon = CITATION_RE.sub(_sub, text)
    return seen, canon

SYSTEM = """You are a manufacturing-floor assistant.

You answer ONLY using the provided context chunks. Each chunk is labeled with its CHUNK_ID like [DOC-ID#section] and a domain tag (safety, maintenance, or quality).

RULES (non-negotiable):
1. Every factual claim must be followed by a citation in ASCII square brackets, exactly as the CHUNK_ID is shown. Use plain '[' and ']' (U+005B, U+005D) — NEVER 【...】 or other Unicode bracket variants.
2. If the context does not contain the answer, respond exactly: "I don't have that information in the {domain} documentation."
3. Do not invent procedures, part numbers, torque specs, sampling rates, fault codes, or section numbers.
4. Do not answer from general knowledge.
5. Prefer short, procedural answers. Use numbered steps when the source uses numbered steps.
6. If the context spans multiple domains (e.g., a question about PPE during maintenance), structure the answer so each part is cited to the correct domain chunk. Do not blur claims across domains without a citation.
"""

USER_TEMPLATE = """Question:
{query}

Context chunks (use only these):
{chunks}

Write the answer now. Cite every claim with [CHUNK_ID]."""


def _format_chunks(retrieved: list[dict]) -> str:
    """Render chunks with their source domain so the LLM (and the eventual
    answer) can disambiguate when context spans multiple collections."""
    lines = []
    for r in retrieved:
        domain = r.get("domain", "")
        heading = r.get("heading", "")
        prefix = f"[{r['chunk_id']}]"
        if domain:
            prefix = f"[{r['chunk_id']}] (domain={domain}; section={heading})"
        else:
            prefix = f"[{r['chunk_id']}] ({heading})"
        lines.append(f"{prefix}\n{r['body']}\n")
    return "\n".join(lines)


def make_generate_node(llm: LLMClient, *, model: str | None = None) -> Callable[[AgentState], AgentState]:
    def generate(state: AgentState) -> AgentState:
        # If guard already refused, no generation call.
        if state.get("refused"):
            state.setdefault("latency_ms", {})["generate"] = 0
            state.setdefault("generation_meta", {})
            return state

        t0 = time.perf_counter()
        domain = state.get("route", "")
        retrieved = state.get("retrieved", []) or []
        retrieved_ids = {r["chunk_id"] for r in retrieved}

        user = USER_TEMPLATE.format(query=state["query"], chunks=_format_chunks(retrieved))
        sys_prompt = SYSTEM.replace("{domain}", domain)
        resp = llm.complete(
            system=sys_prompt,
            user=user,
            temperature=0.0,
            max_tokens=600,
            model=model,
        )
        text = (resp.text or "").strip()
        cited, text = _extract_citations(text)

        # Validate citations
        canonical_refusal_prefix = "I don't have that information in the"
        if cited:
            bad = [c for c in cited if c not in retrieved_ids]
            if bad:
                state["refused"] = True
                state["refusal_reason"] = "fabricated_citation"
                state["answer"] = (
                    f"I can't verify that against the {domain} documentation."
                )
                state["citations"] = []
                state["generation_meta"] = {
                    "model": resp.model,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "fabricated": bad,
                    "raw_text": text,
                }
                state.setdefault("latency_ms", {})["generate"] = int((time.perf_counter() - t0) * 1000)
                return state
            state["answer"] = text
            state["citations"] = cited
            state["refused"] = False
            state["refusal_reason"] = None
        else:
            # No citations at all
            if text.startswith(canonical_refusal_prefix):
                # Model elected to refuse explicitly; honor as a refusal
                state["refused"] = True
                state["refusal_reason"] = "model_refusal"
                state["answer"] = text
                state["citations"] = []
            else:
                state["refused"] = True
                state["refusal_reason"] = "uncited_answer"
                state["answer"] = (
                    f"I can't answer that without citing the {domain} documentation."
                )
                state["citations"] = []

        state["generation_meta"] = {
            "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "raw_text": text,
        }
        state.setdefault("latency_ms", {})["generate"] = int((time.perf_counter() - t0) * 1000)
        return state

    return generate
