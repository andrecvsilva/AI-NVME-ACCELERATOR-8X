"""Memory-aware prefetch policy for mapped weights."""

from __future__ import annotations


def prefetch_decision(*, schedule: dict, requested_bytes: int, max_bytes: int) -> dict:
    if requested_bytes < 0 or max_bytes < 0:
        raise ValueError("prefetch byte counts cannot be negative")
    eligible = bool(schedule.get("allow_prefetch")) and requested_bytes <= max_bytes
    return {
        "eligible_by_memory_policy": eligible,
        "executed": False,
        "requested_bytes": requested_bytes,
        "max_bytes": max_bytes,
        "execution": "not_executed",
        "reason": ("Memory headroom allows a future request, but the active Python binding does not expose the mapped tensor addresses needed to prefetch model-weight pages." if eligible else
                   "Prefetch disabled by policy, memory state, or size bound."),
        "custom_weight_copy": False,
        "note": "No GGUF data is copied or read speculatively by this module. llama.cpp and Windows retain responsibility for their internal mmap/read-ahead behavior.",
    }
