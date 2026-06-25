"""Telemetry node: append one JSONL record per query. Greppable, durable,
zero external services. Survives across restarts on the mounted disk.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..state import AgentState


def make_telemetry_node(jsonl_path: Path) -> Callable[[AgentState], AgentState]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def telemetry(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        record = {
            "trace_id": state.get("trace_id"),
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": state.get("session_id"),
            "query": state.get("query"),
            "route": state.get("route"),
            "route_confidence": state.get("route_confidence"),
            "route_reason": state.get("route_reason"),
            "retrieval": {
                "collection": f"kb_{state.get('route')}" if state.get("route") not in (None, "none") else None,
                "dense_top": state.get("dense_top", []),
                "sparse_top": state.get("sparse_top", []),
                "fused_top": [
                    {"chunk_id": r["chunk_id"], "rrf": r["rrf_score"]}
                    for r in (state.get("retrieved") or [])
                ],
                "confidence": state.get("retrieval_confidence", 0.0),
            },
            "generation": state.get("generation_meta", {}),
            "judge": state.get("judge"),
            "answer": state.get("answer"),
            "refused": bool(state.get("refused")),
            "refusal_reason": state.get("refusal_reason"),
            "citations": state.get("citations", []),
            "latency_ms": state.get("latency_ms", {}),
        }
        # Atomic line append; flush + fsync to survive crashes.
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        state.setdefault("latency_ms", {})["telemetry"] = int((time.perf_counter() - t0) * 1000)
        return state

    return telemetry
