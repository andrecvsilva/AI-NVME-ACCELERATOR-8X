"""Pressure-aware, bounded byte-block cache for controlled experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable

MIB = 1024 * 1024
ENTRY_OVERHEAD_BYTES = 512  # Conservative accounting for key/container metadata.


@dataclass
class CacheEntry:
    value: bytes
    frequency: int
    last_access: int

    @property
    def accounted_bytes(self) -> int:
        return len(self.value) + ENTRY_OVERHEAD_BYTES


class AdaptiveBlockCache:
    """Small application-level cache with an explicit RAM-pressure gate.

    This cache copies bytes and therefore can compete with the Windows page cache.
    It stays disabled for `warning` or `critical` memory pressure. llama.cpp's mmap
    and Windows' own cache should be benchmarked separately before enabling it.
    """

    def __init__(self, *, max_capacity_mib: int = 64, max_entries: int = 2048, eviction_policy: str = "HYBRID", allow_warning_cache: bool = False, warning_capacity_fraction: float = 0.25):
        if max_capacity_mib < 0 or max_entries < 1:
            raise ValueError("max_capacity_mib must be non-negative and max_entries positive")
        eviction_policy = eviction_policy.upper()
        if eviction_policy not in {"LRU", "LFU", "HYBRID"}:
            raise ValueError("eviction_policy must be LRU, LFU, or HYBRID")
        if not 0 <= warning_capacity_fraction <= 1:
            raise ValueError("warning_capacity_fraction must be between 0 and 1")
        self.max_capacity_bytes = max_capacity_mib * MIB
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self.allow_warning_cache = allow_warning_cache
        self.warning_capacity_fraction = warning_capacity_fraction
        self.capacity_bytes = 0
        self.used_bytes = 0
        self._clock = 0
        self._entries: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.disabled_bypasses = 0

    def adjust_capacity(
        self,
        *,
        available_bytes: int,
        configured_budget_bytes: int,
        reserve_bytes: int,
        pressure: str,
    ) -> int:
        """Recompute capacity from a memory sample; does not reserve OS memory."""
        if min(available_bytes, configured_budget_bytes, reserve_bytes) < 0:
            raise ValueError("Memory budget values cannot be negative")
        if pressure not in {"normal", "warning", "critical"}:
            raise ValueError("pressure must be normal, warning, or critical")
        if pressure == "critical" or (pressure == "warning" and not self.allow_warning_cache):
            target = 0
        else:
            target = min(
                self.max_capacity_bytes,
                max(0, available_bytes - reserve_bytes),
                max(0, configured_budget_bytes - reserve_bytes),
            )
            if pressure == "warning":
                target = int(target * self.warning_capacity_fraction)
        self.capacity_bytes = target
        while self.used_bytes > self.capacity_bytes and self._entries:
            self._evict_one()
        return self.capacity_bytes

    def _touch(self, key: Hashable, entry: CacheEntry) -> None:
        self._clock += 1
        entry.frequency += 1
        entry.last_access = self._clock
        self._entries.move_to_end(key)

    def _evict_one(self) -> None:
        if not self._entries:
            return
        if self.eviction_policy == "LRU":
            key, entry = min(self._entries.items(), key=lambda item: item[1].last_access)
        elif self.eviction_policy == "LFU":
            key, entry = min(self._entries.items(), key=lambda item: (item[1].frequency, item[1].last_access))
        else:
            key, entry = min(
                self._entries.items(),
                key=lambda item: (item[1].frequency / (1 + self._clock - item[1].last_access), item[1].last_access),
            )
        del self._entries[key]
        self.used_bytes -= entry.accounted_bytes
        self.evictions += 1

    def get(self, key: Hashable) -> bytes | None:
        if self.capacity_bytes <= 0:
            self.disabled_bypasses += 1
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._touch(key, entry)
        return entry.value

    def put(self, key: Hashable, value: bytes) -> bool:
        if not isinstance(value, bytes):
            raise TypeError("cache accepts immutable bytes only")
        accounted = len(value) + ENTRY_OVERHEAD_BYTES
        if self.capacity_bytes <= 0:
            self.disabled_bypasses += 1
            return False
        if accounted > self.capacity_bytes:
            return False

        previous = self._entries.pop(key, None)
        if previous is not None:
            self.used_bytes -= previous.accounted_bytes
        while self._entries and (self.used_bytes + accounted > self.capacity_bytes or len(self._entries) >= self.max_entries):
            self._evict_one()
        self._clock += 1
        self._entries[key] = CacheEntry(value=value, frequency=(previous.frequency + 1 if previous else 1), last_access=self._clock)
        self.used_bytes += accounted
        return True

    def get_or_load(self, key: Hashable, loader: Callable[[], bytes]) -> bytes:
        value = self.get(key)
        if value is not None:
            return value
        value = loader()
        if not isinstance(value, bytes):
            raise TypeError("loader must return immutable bytes")
        self.put(key, value)
        return value

    def clear(self) -> None:
        self._entries.clear()
        self.used_bytes = 0

    def stats(self) -> dict:
        total_lookups = self.hits + self.misses
        return {
            "enabled": self.capacity_bytes > 0,
            "capacity_bytes": self.capacity_bytes,
            "used_accounted_bytes": self.used_bytes,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total_lookups, 4) if total_lookups else None,
            "evictions": self.evictions,
            "disabled_bypasses": self.disabled_bypasses,
            "eviction_policy": self.eviction_policy,
        }
