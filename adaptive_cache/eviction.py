"""Removal ordering shared by RAM and persistent cache policies."""

from __future__ import annotations


def validate_policy(policy: str) -> str:
    normalized = policy.upper()
    if normalized not in {"LRU", "LFU", "HYBRID"}:
        raise ValueError("Eviction policy must be LRU, LFU, or HYBRID")
    return normalized


def rank_entries(entries: list[dict], policy: str) -> list[dict]:
    """Return lowest-value entries first; inputs stay bounded by cache index size."""
    policy = validate_policy(policy)
    if policy == "LRU":
        return sorted(entries, key=lambda item: (item.get("accessed_at", 0), item.get("frequency", 0)))
    if policy == "LFU":
        return sorted(entries, key=lambda item: (item.get("frequency", 0), item.get("accessed_at", 0)))
    now = max((item.get("accessed_at", 0) for item in entries), default=0)
    return sorted(entries, key=lambda item: (
        item.get("frequency", 0) / (1 + max(0, now - item.get("accessed_at", 0))),
        item.get("accessed_at", 0),
    ))
