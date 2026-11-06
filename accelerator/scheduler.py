"""Admission and bounded runtime choices driven by Memory Guard snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import AcceleratorConfig

GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class ScheduleDecision:
    allowed: bool
    guard_state: str
    reason: str
    context: int
    batch: int
    threads: int
    mmap: bool
    mlock: bool
    allow_prefetch: bool
    available_bytes: int | None
    reserve_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveRamNvmeScheduler:
    """Chooses inference settings; Windows remains the actual page manager."""

    def __init__(self, config: AcceleratorConfig):
        config.validate()
        self.config = config

    def decide(self, snapshot: dict) -> ScheduleDecision:
        guard = snapshot.get("guard") or {}
        state = str(guard.get("pressure", "unknown")).lower()
        physical = snapshot.get("physical_memory") or {}
        available = physical.get("available_bytes")
        reserve = self.config.reserve_mib * MIB
        if not self.config.enabled:
            return ScheduleDecision(False, state, "Accelerator is disabled.", self.config.min_context,
                                    self.config.min_batch, 1, True, False, False, available, reserve)
        if state != "normal" or available is None:
            return ScheduleDecision(False, state, "Memory Guard must be NORMAL with a valid RAM snapshot.",
                                    self.config.min_context, self.config.min_batch, 1, True, False,
                                    False, available, reserve)
        minimum_available = int(self.config.minimum_available_gib * GIB)
        if int(available) < minimum_available:
            return ScheduleDecision(False, state,
                                    "Available RAM is below the configured model-start threshold.",
                                    self.config.min_context, self.config.min_batch, 1, True, False,
                                    False, int(available), reserve)
        usable = max(0, int(available) - reserve)
        if usable < 512 * MIB:
            return ScheduleDecision(False, state, "Insufficient RAM headroom after the configured reserve.",
                                    self.config.min_context, self.config.min_batch, 1, True, False,
                                    False, int(available), reserve)
        if int(available) < 4 * GIB:
            # Low-RAM mmap profile: let the OS page file-backed GGUF weights,
            # while minimizing KV/batch/thread overhead. No speculative reads.
            context, batch, threads = self.config.min_context, self.config.min_batch, 1
        elif usable < 2 * GIB:
            context = min(self.config.max_context, max(self.config.min_context, 256))
            batch = min(self.config.max_batch, max(self.config.min_batch, 4))
            threads = min(self.config.max_threads, 2)
        elif usable < int(3.5 * GIB):
            context = min(self.config.max_context, max(self.config.min_context, 384))
            batch = min(self.config.max_batch, max(self.config.min_batch, 8))
            threads = min(self.config.max_threads, 3)
        else:
            context, batch = self.config.max_context, self.config.max_batch
            threads = self.config.max_threads
        prefetch = bool(self.config.prefetch_enabled and usable >= self.config.max_prefetch_mib * MIB)
        return ScheduleDecision(True, state, "NORMAL pressure and reserve satisfied; settings capped by config.",
                                context, batch, threads, True, False, prefetch,
                                int(available), reserve)
