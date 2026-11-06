"""Central configuration for the integrated accelerator prototype."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcceleratorConfig:
    enabled: bool = True
    model_path: str = str(
        Path.home() / "Downloads" / "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    )
    memory_budget_gib: float = 3.0
    # Allow file-backed mmap to be tested with less than the GGUF file size in
    # available RAM. Memory Guard's NORMAL/CRITICAL limits and reserve still apply.
    minimum_available_gib: float = 2.0
    reserve_mib: int = 1024
    max_context: int = 512
    min_context: int = 128
    max_batch: int = 16
    min_batch: int = 4
    max_threads: int = 4
    observation_interval_seconds: float = 1.0
    use_mmap: bool = True
    use_mlock: bool = False
    # llama.cpp's CPU extra buffer types may create a large runtime-repacked
    # copy of quantized weights. Disable them for memory-constrained testing.
    use_extra_bufts: bool = False
    # PDH instance name on this host: disk 0 contains volume C:.
    physical_disk_instance: str = "0 C:"
    prefetch_enabled: bool = False
    max_prefetch_mib: int = 1
    max_generation_tokens: int = 32
    benchmark_repetitions: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "AcceleratorConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "model_path" in data:
            data["model_path"] = os.path.expandvars(data["model_path"])
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.25 <= self.memory_budget_gib <= 64:
            raise ValueError("memory_budget_gib must be between 0.25 and 64")
        if not 2 <= self.minimum_available_gib <= 64:
            raise ValueError("minimum_available_gib must be between 2 and 64")
        if not 256 <= self.reserve_mib <= 8192:
            raise ValueError("reserve_mib must be between 256 and 8192")
        if not 128 <= self.min_context <= self.max_context <= 4096:
            raise ValueError("context limits must satisfy 128 <= min <= max <= 4096")
        if not 1 <= self.min_batch <= self.max_batch <= 128:
            raise ValueError("batch limits must satisfy 1 <= min <= max <= 128")
        if not 1 <= self.max_threads <= 8:
            raise ValueError("max_threads must be between 1 and 8")
        if not 0.25 <= self.observation_interval_seconds <= 10:
            raise ValueError("observation interval must be between 0.25 and 10 seconds")
        if self.use_mlock:
            raise ValueError("use_mlock is unsupported by the safe profile; it can pin model pages in RAM")
        if (not isinstance(self.use_extra_bufts, bool)
                or not isinstance(self.physical_disk_instance, str)
                or not self.physical_disk_instance.strip()):
            raise ValueError("use_extra_bufts must be boolean and physical_disk_instance cannot be empty")
        if not self.use_mmap:
            raise ValueError("use_mmap must remain enabled for file-backed model loading")
        if self.max_generation_tokens < 1 or self.benchmark_repetitions != 1:
            raise ValueError("safe comparison is limited to one repetition; generation tokens must be positive")
        if not 0 <= self.max_prefetch_mib <= 4:
            raise ValueError("max_prefetch_mib must be between 0 and 4")
