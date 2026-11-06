"""State classification and inference admission policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .monitor import GIB, MemorySnapshot, read_memory_snapshot
from .thresholds import DEFAULT_GUARD_THRESHOLDS, GuardThresholds


@dataclass(frozen=True)
class GuardDecision:
    state: str
    allowed: bool
    reason: str
    snapshot: MemorySnapshot


def classify_memory(snapshot: MemorySnapshot, thresholds: GuardThresholds) -> str:
    return thresholds.classify(
        available_bytes=snapshot.available_bytes,
        physical_memory_load_percent=snapshot.load_percent,
        commit_used_percent=snapshot.commit_used_percent,
    ).upper()


class MemoryGuard:
    def __init__(
        self,
        thresholds: GuardThresholds | None = None,
        snapshot_provider: Callable[[], MemorySnapshot] = read_memory_snapshot,
    ):
        self.thresholds = thresholds or DEFAULT_GUARD_THRESHOLDS
        self.snapshot_provider = snapshot_provider

    def inspect(self) -> GuardDecision:
        snapshot = self.snapshot_provider()
        state = classify_memory(snapshot, self.thresholds)
        if state == "CRITICAL":
            return GuardDecision(state, False, "Critical memory pressure blocks new inference requests.", snapshot)
        if state == "WARNING":
            return GuardDecision(state, False, "Warning memory pressure blocks new inference requests by default.", snapshot)
        return GuardDecision(state, True, "Memory is within the configured NORMAL range.", snapshot)

    def require_admission(self) -> GuardDecision:
        decision = self.inspect()
        if not decision.allowed:
            raise MemoryError(f"{decision.state}: {decision.reason}")
        return decision
