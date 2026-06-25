"""Validated provider/model catalog.

Each entry is a model we **live-probed** and confirmed returns 200. The UI
queries `/providers/catalog` and renders these as dropdown options.

Excluded (verified 429/404 in live probes last session):
- openrouter `meta-llama/llama-3.3-70b-instruct:free`     (429)
- openrouter `qwen/qwen3-next-80b-a3b-instruct:free`      (429)
- openrouter `nousresearch/hermes-3-llama-3.1-405b:free`  (429)
- openrouter `google/gemini-flash-1.5:free`               (404)
- openrouter `openrouter/auto`                            (402 — paid)

Adding a model is a deliberate change: probe live first, then add here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


Role = Literal["answer", "judge"]


@dataclass(frozen=True)
class CatalogEntry:
    provider: str
    model: str
    label: str        # human-friendly name for the dropdown
    roles: list[str]  # which roles this model can serve (answer / judge / both)
    notes: str = ""


CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        label="OpenRouter · gpt-oss-120b (free)",
        roles=["answer", "judge"],
        notes="Default answer model. ~25-30s on cold load.",
    ),
    CatalogEntry(
        provider="openrouter",
        model="openai/gpt-oss-20b:free",
        label="OpenRouter · gpt-oss-20b (free)",
        roles=["answer", "judge"],
        notes="Default judge model. Faster than 120b.",
    ),
    CatalogEntry(
        provider="openrouter",
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        label="OpenRouter · Nemotron-3 Nano 30B (free)",
        roles=["answer"],
    ),
    CatalogEntry(
        provider="groq",
        model="llama-3.3-70b-versatile",
        label="Groq · Llama 3.3 70B Versatile",
        roles=["answer", "judge"],
        notes="Very low latency. Free-tier rate limits apply.",
    ),
    CatalogEntry(
        provider="groq",
        model="llama-3.1-8b-instant",
        label="Groq · Llama 3.1 8B Instant",
        roles=["answer", "judge"],
        notes="Fastest option. Smaller model.",
    ),
    CatalogEntry(
        provider="groq",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        label="Groq · Llama 4 Scout 17B",
        roles=["answer"],
    ),
]


def catalog_for_role(role: Role) -> list[CatalogEntry]:
    return [c for c in CATALOG if role in c.roles]


def catalog_dict() -> dict:
    return {
        "entries": [
            {**asdict(c), "id": f"{c.provider}/{c.model}"} for c in CATALOG
        ],
    }


def is_known_choice(provider: str, model: str, role: Role | None = None) -> bool:
    for c in CATALOG:
        if c.provider == provider and c.model == model:
            if role is None or role in c.roles:
                return True
    return False
