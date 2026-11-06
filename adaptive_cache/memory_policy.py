"""Memory budget policy that delegates all state detection to Memory Guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import AdaptiveCacheConfig
from memory_guard import MemoryGuard
from memory_guard.process_monitor import collect_ollama_processes

MIB = 1024**2


@dataclass(frozen=True)
class CacheMemoryDecision:
    available: bool
    guard_state: str
    capacity_bytes: int
    available_bytes: int
    reserve_bytes: int
    relevant_process_private_working_set_bytes: int
    allow_ram_insert: bool
    allow_nvme_read: bool
    allow_nvme_write: bool
    allow_prefetch: bool
    reason: str


class AdaptiveMemoryPolicy:
    def __init__(
        self,
        config: AdaptiveCacheConfig,
        *,
        memory_guard: MemoryGuard | None = None,
        process_provider: Callable[[], dict] = collect_ollama_processes,
    ):
        self.config = config
        self.memory_guard = memory_guard or MemoryGuard()
        self.process_provider = process_provider

    def evaluate(self) -> CacheMemoryDecision:
        if not self.config.enabled:
            return CacheMemoryDecision(False, "DISABLED", 0, 0, 0, 0, False, False, False, False, "Adaptive cache is disabled.")
        try:
            decision = self.memory_guard.inspect()
            snapshot = decision.snapshot
            process_report = self.process_provider()
            if process_report.get("status") != "active":
                raise RuntimeError(process_report.get("error") or "Ollama process metrics unavailable")
            process_private_ws = sum(
                int(proc["private_working_set_bytes"])
                for proc in process_report.get("processes", [])
                if proc.get("status") == "ok" and proc.get("private_working_set_bytes") is not None
            )
        except Exception as exc:
            return CacheMemoryDecision(False, "UNAVAILABLE", 0, 0, self.config.ram_reserve_mib * MIB,
                                       0, False, False, False, False, f"Memory/process state unavailable: {exc}")

        state = decision.state
        reserve = self.config.ram_reserve_mib * MIB
        remaining_after_reserve = max(0, snapshot.available_bytes - reserve)
        process_reserve = int(process_private_ws * self.config.process_reserve_fraction)
        physical_budget = int(max(0, snapshot.total_bytes - reserve - process_reserve) * self.config.ram_max_fraction)
        configured_budget = self.config.ram_budget_mib * MIB
        process_ratio = min(1.0, process_private_ws / snapshot.total_bytes) if snapshot.total_bytes else 1.0
        process_adjusted_cache_cap = int(self.config.ram_cache_max_mib * MIB * max(
            0.0, 1.0 - process_ratio * self.config.process_cache_penalty_fraction
        ))
        max_capacity = min(process_adjusted_cache_cap, configured_budget, physical_budget,
                           max(0, remaining_after_reserve - process_reserve))

        if state == "CRITICAL":
            return CacheMemoryDecision(True, state, 0, snapshot.available_bytes, reserve, process_private_ws, False, False, False, False,
                                       "CRITICAL blocks new cache allocations, NVMe reads/writes, and prefetch.")
        if state == "WARNING":
            warning_limit = min(max_capacity, self.config.warning_cache_max_mib * MIB)
            return CacheMemoryDecision(True, state, warning_limit, snapshot.available_bytes, reserve, process_private_ws,
                                       warning_limit > 0, False, False, False,
                                       "WARNING permits only a small bounded RAM cache; NVMe reads/writes and prefetch are paused.")
        if state != "NORMAL":
            return CacheMemoryDecision(False, "UNAVAILABLE", 0, snapshot.available_bytes, reserve, process_private_ws,
                                       False, False, False, False, f"Unknown Memory Guard state: {state}")
        capacity = max_capacity if decision.allowed else 0
        return CacheMemoryDecision(True, state, capacity, snapshot.available_bytes, reserve, process_private_ws,
                                   capacity > 0, True, True,
                                   self.config.prefetch_enabled and capacity >= self.config.prefetch_block_kib * 1024,
                                   "NORMAL memory policy within configured RAM, process, and reserve limits.")
