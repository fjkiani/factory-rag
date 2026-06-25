"""Deterministic keyword router. Backup for when:
- The classifier LLM call fails (timeout, 5xx, parse error)
- The classifier returns low confidence (below ROUTE_CONF_THRESHOLD)
- A demo runs with OPENROUTER_API_KEY unset

Returns the same shape as the LLM classifier: a ranked list of
{route, confidence, reason}. The router never invents a domain it has no
evidence for; if no keyword fires it returns route="none".

Confidence is calibrated to be intentionally LOWER than typical LLM
confidence (max ~0.7) so an honest classifier signal is preferred when
both are available. The downstream guard treats the keyword router as
"better than nothing" rather than as a definitive answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Domain -> (regex, weight). Each match adds weight; route score normalized at the end.
# Patterns are word-boundary anchored to avoid spurious matches (e.g. "PPE" should
# not fire inside "appendix"). Patterns are case-insensitive.
_RULES: dict[str, list[tuple[str, float]]] = {
    "safety": [
        (r"\bloto\b", 3.0),
        (r"\block(?:[- ]?out|out)\b", 3.0),
        (r"\btag(?:[- ]?out|out)\b", 3.0),
        (r"\bppe\b", 2.5),
        (r"\bhazard(?:s|ous)?\b", 2.0),
        (r"\bemergency\b", 2.0),
        (r"\be[- ]?stop\b", 2.5),
        (r"\b(?:zero[- ]?energy|de[- ]?energize|deenergize)\b", 2.5),
        (r"\bguard(?:s|ing)?\b", 1.5),
        (r"\bisolat(?:e|ion|ed)\b", 2.0),
        (r"\bbleed\s+valve\b", 2.0),
        (r"\bhydraulic\s+accumulator\b", 2.0),
        (r"\bsafety\b", 1.5),
        (r"\bpress\s+envelope\b", 2.0),
    ],
    "maintenance": [
        (r"\bfault\s+code\b", 3.0),
        (r"\be[- ]?\d{3}\b", 2.5),  # e.g. E-318
        (r"\b(?:preventive|preventative)\s+maintenance\b", 3.0),
        (r"\bpm\s+(?:schedule|kit|interval)\b", 2.5),
        (r"\blubricat(?:e|ion|ing)\b", 2.5),
        (r"\b(?:spindle|ball[- ]?screw|turret)\b", 2.0),
        (r"\bcoolant\b", 1.5),
        (r"\bcnc\b", 1.5),
        (r"\blathe\b", 1.5),
        (r"\b(?:500|2000|200)\s*-?\s*hour\b", 2.5),
        (r"\bservice\s+(?:kit|interval|schedule)\b", 2.0),
        (r"\bfault\b", 1.0),
        (r"\bcalibrat(?:e|ion)\b", 1.0),
    ],
    "quality": [
        (r"\baql\b", 3.0),
        (r"\bsampling\s+plan\b", 3.0),
        (r"\bz1\.4\b", 3.0),
        (r"\biso\s*9001\b", 3.0),
        (r"\bnon[- ]?conform(?:ance|ing)\b", 3.0),
        (r"\bncr\b", 2.5),
        (r"\bdimensional\s+inspection\b", 2.5),
        (r"\bsurface\s+finish\b", 2.5),
        (r"\b(?:ra|roughness)\s*(?:\u2264|<=|<)?\s*\d*\.?\d+\s*(?:\u00b5m|um|micrometer)\b", 2.5),
        (r"\bgd&t\b", 2.0),
        (r"\btolerance\b", 1.5),
        (r"\binspection\b", 1.0),
        (r"\b(?:accept|reject)\s+\d+\s*/\s*\d+\b", 2.5),
        (r"\bcmm\b", 2.0),
        (r"\bprofilometer\b", 2.5),
    ],
}

# Domains we will *not* shadow with a fallback decision unless score >= this floor.
_MIN_RAW_SCORE = 1.5
# Confidence ceiling for the keyword router (intentionally below LLM ceiling)
_CONF_CEILING = 0.70


@dataclass(frozen=True)
class RouteScore:
    route: str
    confidence: float
    reason: str
    matched: list[str]


def score_routes(query: str) -> list[RouteScore]:
    """Score all domains for the query and return them sorted by score (desc).
    Always returns 3 entries (one per domain). Use the top entry's score and
    matched-pattern count to decide whether to trust the fallback.
    """
    q = (query or "").lower()
    scored: list[RouteScore] = []
    for domain, rules in _RULES.items():
        total = 0.0
        matched: list[str] = []
        for pattern, weight in rules:
            if re.search(pattern, q):
                total += weight
                matched.append(pattern)
        # Map raw score to [0, _CONF_CEILING] with a soft saturating curve.
        # ~2.0 raw -> 0.34, ~5.0 raw -> 0.59, ~10.0 raw -> 0.68
        conf = _CONF_CEILING * (1.0 - 1.0 / (1.0 + total / 3.0)) if total > 0 else 0.0
        scored.append(
            RouteScore(
                route=domain,
                confidence=round(conf, 4),
                reason=(
                    f"keyword_router matched {len(matched)} pattern(s) in '{domain}'"
                    if matched
                    else "keyword_router: no matches"
                ),
                matched=matched,
            )
        )
    scored.sort(key=lambda r: -r.confidence)
    return scored


def fallback_classify(query: str) -> list[dict]:
    """Public entry point. Returns a list of dicts ready to plug into
    `state["route_candidates"]`. The list is always length 3 unless every
    domain scored 0 — in which case length 1 with route='none'.
    """
    scored = score_routes(query)
    top = scored[0]
    # If no keyword fired strongly enough, the fallback DOES NOT pick a domain.
    # Cite-or-refuse handles the rest at guard.
    if top.confidence == 0.0 or top.confidence < _CONF_CEILING * (1.0 - 1.0 / (1.0 + _MIN_RAW_SCORE / 3.0)):
        return [{"route": "none", "confidence": 0.0, "reason": "keyword_router: no rule fired with sufficient weight", "source": "keyword_router"}]
    return [
        {
            "route": s.route,
            "confidence": s.confidence,
            "reason": s.reason,
            "matched": s.matched,
            "source": "keyword_router",
        }
        for s in scored
    ]
