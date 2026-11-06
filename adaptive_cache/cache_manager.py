"""Memory Guard-aware coordinator for bounded RAM/NVMe caches and prefetch."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Callable

from engine.cache_manager import AdaptiveBlockCache
from engine.prefetch_engine import PredictivePrefetchEngine
from .config import AdaptiveCacheConfig
from .memory_policy import AdaptiveMemoryPolicy
from .metrics import AdaptiveCacheMetrics, CacheEventLogger
from .nvme_cache import NvmeCache


class AdaptiveCacheEngine:
    def __init__(
        self,
        config: AdaptiveCacheConfig | None = None,
        *,
        project_root: str | Path = ".",
        policy: AdaptiveMemoryPolicy | None = None,
        logs_dir: str | Path | None = None,
    ):
        self.config = config or AdaptiveCacheConfig()
        self.project_root = Path(project_root).resolve()
        nvme_root = Path(self.config.nvme_directory)
        if not nvme_root.is_absolute():
            nvme_root = self.project_root / nvme_root
        self.nvme = NvmeCache(self.config, root=nvme_root)
        self.policy = policy or AdaptiveMemoryPolicy(self.config)
        self.ram = AdaptiveBlockCache(
            max_capacity_mib=self.config.ram_cache_max_mib,
            max_entries=self.config.ram_cache_entries,
            eviction_policy=self.config.eviction_policy,
            allow_warning_cache=True,
            warning_capacity_fraction=self.config.warning_cache_fraction,
        )
        self.metrics = AdaptiveCacheMetrics()
        self.logger = CacheEventLogger(logs_dir or self.project_root / "logs")
        self.prefetcher = None
        if self.config.prefetch_enabled:
            self.prefetcher = PredictivePrefetchEngine(max_block_bytes=self.config.prefetch_block_kib * 1024)
        self._lock = RLock()
        self._seen_ram_evictions = 0
        self._seen_nvme_removals = 0
        self._last_policy_signature = None
        self.last_decision = None

    def _sync_removed_metrics(self) -> None:
        current_ram = self.ram.evictions
        current_nvme = self.nvme.items_removed
        removed = max(0, current_ram - self._seen_ram_evictions) + max(0, current_nvme - self._seen_nvme_removals)
        if removed:
            self.metrics.increment("items_removed", removed)
        self._seen_ram_evictions = current_ram
        self._seen_nvme_removals = current_nvme

    def _log(self, operation: str, result: str, *, key=None, duration=0.0, error=None, bytes_read=0,
             bytes_written=0, items_removed=0) -> None:
        decision = self.last_decision
        try:
            self.logger.append({
                "operation": operation,
                "key_hash": hashlib.sha256((key if isinstance(key, bytes) else str(key)).encode("utf-8")
                                            if not isinstance(key, bytes) else key).hexdigest() if key is not None else "",
                "result": result,
                "bytes_read": bytes_read,
                "bytes_written": bytes_written,
                "items_removed": items_removed,
                "duration_seconds": round(duration, 6),
                "memory_state": decision.guard_state if decision else "UNKNOWN",
                "ram_used_bytes": self.ram.used_bytes,
                "nvme_used_bytes": self.nvme.stats().get("used_bytes", 0),
                "error": error,
            })
        except (OSError, RuntimeError, sqlite3.Error):
            self.metrics.increment("errors")

    def refresh_capacity(self):
        with self._lock:
            try:
                decision = self.policy.evaluate()
            except Exception as exc:
                decision = None
                reason = str(exc)
            else:
                reason = ""
            self.last_decision = decision
            signature = None if decision is None else (
                decision.available, decision.guard_state, decision.capacity_bytes,
                decision.allow_ram_insert, decision.allow_nvme_read,
                decision.allow_nvme_write, decision.allow_prefetch,
            )
            policy_changed = signature != self._last_policy_signature
            self._last_policy_signature = signature
            if decision is None or not decision.available:
                self.ram.adjust_capacity(available_bytes=0, configured_budget_bytes=0,
                                         reserve_bytes=0, pressure="critical")
                if decision is None:
                    self.metrics.increment("errors")
                if policy_changed:
                    self._log("policy", "blocked", error=reason or (decision.reason if decision else "Policy unavailable"))
                return decision
            pressure = decision.guard_state.casefold()
            capacity = decision.capacity_bytes
            self.ram.adjust_capacity(
                available_bytes=capacity + decision.reserve_bytes,
                configured_budget_bytes=capacity + decision.reserve_bytes,
                reserve_bytes=decision.reserve_bytes,
                pressure=pressure if pressure in {"normal", "warning", "critical"} else "critical",
            )
            self._sync_removed_metrics()
            if policy_changed:
                self._log("policy", "evaluated")
            return decision

    def get(self, key):
        started = time.perf_counter()
        with self._lock:
            decision = self.refresh_capacity()
            if not self.config.enabled:
                self.metrics.increment("misses")
                self._log("get", "cache_disabled", key=key, duration=time.perf_counter() - started)
                return None
            value = self.ram.get(key)
            if value is not None:
                self.metrics.increment("hits")
                self._sync_removed_metrics()
                self._log("get", "ram_hit", key=key, duration=time.perf_counter() - started)
                return value
            if decision is None or not decision.available or not decision.allow_nvme_read:
                self.metrics.increment("misses")
                self._sync_removed_metrics()
                self._log("get", "blocked_or_miss", key=key, duration=time.perf_counter() - started)
                return None
            value = self.nvme.get(key) if self.config.nvme_enabled else None
            if value is None:
                self.metrics.increment("misses")
                self._sync_removed_metrics()
                self._log("get", "miss", key=key, duration=time.perf_counter() - started)
                return None
            self.metrics.increment("hits")
            self.metrics.increment("bytes_read", len(value))
            self.metrics.increment("read_seconds", time.perf_counter() - started)
            if decision.allow_ram_insert:
                self.ram.put(key, value)
            self._sync_removed_metrics()
            self._log("get", "nvme_hit", key=key, duration=time.perf_counter() - started, bytes_read=len(value))
            return value

    def get_or_load(self, key, loader: Callable[[], bytes]) -> bytes:
        value = self.get(key)
        if value is not None:
            return value
        decision = self.last_decision
        if decision is None or decision.guard_state != "NORMAL" or not decision.allow_ram_insert:
            raise MemoryError("Adaptive Cache blocks source reads unless Memory Guard is NORMAL.")
        value = loader()
        if not isinstance(value, bytes):
            raise TypeError("loader must return immutable bytes")
        self.put(key, value)
        return value

    def put(self, key, value: bytes) -> bool:
        if not isinstance(value, bytes):
            raise TypeError("Adaptive Cache accepts immutable bytes only")
        started = time.perf_counter()
        with self._lock:
            decision = self.refresh_capacity()
            if not self.config.enabled:
                self._log("put", "cache_disabled", key=key, duration=time.perf_counter() - started)
                return False
            if decision is None or not decision.available or decision.guard_state == "CRITICAL":
                self._log("put", "blocked", key=key, duration=time.perf_counter() - started,
                          error=decision.reason if decision else "Memory policy unavailable")
                return False
            ram_stored = decision.allow_ram_insert and self.ram.put(key, value)
            nvme_stored = False
            error = None
            if decision.allow_nvme_write and self.config.nvme_enabled:
                try:
                    nvme_stored = self.nvme.put(key, value)
                    if nvme_stored:
                        self.metrics.increment("bytes_written", len(value))
                        self.metrics.increment("write_seconds", time.perf_counter() - started)
                except (OSError, sqlite3.Error) as exc:
                    error = str(exc)
                    self.metrics.increment("errors")
            self._sync_removed_metrics()
            result = "stored" if ram_stored or nvme_stored else "not_cached"
            self._log("put", result, key=key, duration=time.perf_counter() - started,
                      bytes_written=len(value) if nvme_stored else 0, error=error)
            return ram_stored or nvme_stored

    def observe_prefetch(self, file_path: str | Path, block_index: int) -> int | None:
        started = time.perf_counter()
        with self._lock:
            decision = self.refresh_capacity()
            if not self.config.enabled or self.prefetcher is None or decision is None or not decision.allow_prefetch:
                self.metrics.increment("prefetch_misses")
                self._log("prefetch", "disabled_by_policy", duration=time.perf_counter() - started)
                return None
            try:
                path = Path(file_path).resolve(strict=True)
            except OSError as exc:
                self._log("prefetch", "rejected_missing_source", duration=time.perf_counter() - started, error=str(exc))
                return None
            try:
                path.relative_to(self.nvme.root)
            except ValueError:
                self._log("prefetch", "rejected_outside_cache", duration=time.perf_counter() - started)
                return None
            if path.suffix.casefold() != ".blob" or path.stat().st_size > self.config.nvme_item_max_mib * 1024**2:
                self._log("prefetch", "rejected_source", duration=time.perf_counter() - started)
                return None
            block_bytes = self.config.prefetch_block_kib * 1024
            before_scheduled = self.prefetcher.stats()["scheduled"]
            predicted = self.prefetcher.observe_and_prefetch(
                path, block_index, block_bytes,
                pressure=decision.guard_state.casefold(),
                available_bytes=decision.available_bytes,
                reserve_bytes=decision.reserve_bytes,
            )
            scheduled = self.prefetcher.stats()["scheduled"] > before_scheduled
            self._log("prefetch", "scheduled" if scheduled else "observed", key=path.stem,
                      duration=time.perf_counter() - started)
            return predicted

    def take_prefetched_block(self, file_path: str | Path, block_index: int) -> bytes | None:
        if self.prefetcher is None:
            return None
        block_bytes = self.config.prefetch_block_kib * 1024
        result = self.prefetcher.take(file_path, block_index, block_bytes, wait_timeout_s=0)
        self.metrics.increment("prefetch_hits" if result is not None else "prefetch_misses")
        return result

    def clear_ram(self) -> None:
        with self._lock:
            self.ram.clear()
            self._log("clear_ram", "released")

    def stats(self) -> dict:
        decision = self.refresh_capacity()
        self._sync_removed_metrics()
        ram = self.ram.stats()
        nvme = self.nvme.stats()
        metrics = self.metrics.snapshot()
        prefetch = self.prefetcher.stats() if self.prefetcher is not None else {"enabled": False, "scheduled": 0}
        return {
            "enabled": self.config.enabled,
            "memory_state": decision.guard_state if decision else "UNAVAILABLE",
            "cache_active": bool(self.config.enabled and decision and decision.guard_state in {"NORMAL", "WARNING"}
                                and (ram["capacity_bytes"] > 0 or decision.allow_nvme_read)),
            "ram": ram,
            "ram_limit_bytes": ram["capacity_bytes"],
            "nvme": nvme,
            "metrics": metrics,
            "prefetch": prefetch,
            "prefetch_enabled": bool(self.prefetcher and decision and decision.allow_prefetch),
            "policy_reason": decision.reason if decision else "Memory Guard unavailable",
        }

    def close(self) -> None:
        if self.prefetcher is not None:
            self.prefetcher.close()
