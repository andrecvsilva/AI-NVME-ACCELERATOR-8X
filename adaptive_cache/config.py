"""Centralized, validated Adaptive Cache configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdaptiveCacheConfig:
    enabled: bool = True
    nvme_enabled: bool = True
    ram_cache_max_mib: int = 64
    ram_cache_entries: int = 2048
    ram_budget_mib: int = 256
    ram_reserve_mib: int = 1024
    ram_max_fraction: float = 0.05
    process_reserve_fraction: float = 0.10
    process_cache_penalty_fraction: float = 0.50
    warning_cache_fraction: float = 0.25
    warning_cache_max_mib: int = 4
    nvme_directory: str = "cache/nvme"
    nvme_max_mib: int = 2048
    nvme_min_free_mib: int = 10240
    nvme_item_max_mib: int = 8
    entry_ttl_seconds: int = 604800
    eviction_policy: str = "HYBRID"
    prefetch_enabled: bool = False
    prefetch_block_kib: int = 64
    prefetch_max_pending: int = 1

    def __post_init__(self) -> None:
        if min(self.ram_cache_max_mib, self.ram_cache_entries, self.ram_budget_mib, self.ram_reserve_mib,
               self.warning_cache_max_mib, self.nvme_max_mib, self.nvme_min_free_mib,
               self.nvme_item_max_mib, self.entry_ttl_seconds, self.prefetch_max_pending) < 0:
            raise ValueError("Cache size, count, and TTL settings cannot be negative")
        if self.ram_cache_entries < 1 or self.prefetch_max_pending != 1:
            raise ValueError("At least one RAM entry is required; this version supports one prefetch request")
        for name, value in (("ram_max_fraction", self.ram_max_fraction),
                            ("process_reserve_fraction", self.process_reserve_fraction),
                            ("process_cache_penalty_fraction", self.process_cache_penalty_fraction),
                            ("warning_cache_fraction", self.warning_cache_fraction)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.eviction_policy.upper() not in {"LRU", "LFU", "HYBRID"}:
            raise ValueError("eviction_policy must be LRU, LFU, or HYBRID")
        if not 4 <= self.prefetch_block_kib <= 64 or self.prefetch_block_kib % 4:
            raise ValueError("prefetch block must be a 4 KiB multiple between 4 and 64 KiB")

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveCacheConfig":
        config_path = Path(path)
        values = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**values)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
