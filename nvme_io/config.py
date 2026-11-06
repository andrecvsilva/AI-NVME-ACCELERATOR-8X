"""Validated limits for explicitly requested storage measurements."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NvmeIoConfig:
    drive: str = "C:"
    test_sizes_mib: tuple[int, ...] = (16, 64, 256)
    sequential_block_kib: int = 1024
    random_operations: int = 64
    random_block_kib: tuple[int, ...] = (4, 64)
    warning_max_file_mib: int = 16
    warning_random_operations: int = 16
    maximum_total_temp_mib: int = 336
    minimum_free_mib: int = 1024
    memory_poll_seconds: float = 1.0
    temp_directory: str = "benchmark/nvme_io_runs"

    def __post_init__(self) -> None:
        normalized_drive = self.drive.strip().upper().rstrip("\\/")
        if len(normalized_drive) != 2 or normalized_drive[1] != ":" or not normalized_drive[0].isalpha():
            raise ValueError("drive must be a Windows drive letter such as C:")
        object.__setattr__(self, "drive", normalized_drive)
        if not self.test_sizes_mib or any(size < 1 for size in self.test_sizes_mib):
            raise ValueError("test_sizes_mib must contain positive sizes")
        if sum(self.test_sizes_mib) > self.maximum_total_temp_mib:
            raise ValueError("sum(test_sizes_mib) exceeds maximum_total_temp_mib")
        if max(self.test_sizes_mib) > 256 or self.maximum_total_temp_mib > 336:
            raise ValueError("file size is capped at 256 MiB and total temporary writes at 336 MiB")
        if not 4 <= self.sequential_block_kib <= 4096:
            raise ValueError("sequential_block_kib must be 4..4096")
        if not 1 <= self.random_operations <= 4096 or not 1 <= self.warning_random_operations <= self.random_operations:
            raise ValueError("random operation limits must be between 1 and 4096")
        if not self.random_block_kib or any(size not in {4, 64} for size in self.random_block_kib):
            raise ValueError("random_block_kib may contain 4 and/or 64")
        if self.warning_max_file_mib < 1 or self.minimum_free_mib < 0:
            raise ValueError("warning_max_file_mib must be positive and minimum_free_mib nonnegative")
        if not 0.1 <= self.memory_poll_seconds <= 60:
            raise ValueError("memory_poll_seconds must be between 0.1 and 60")

    @classmethod
    def load(cls, path: str | Path) -> "NvmeIoConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for field in ("test_sizes_mib", "random_block_kib"):
            if field in values:
                values[field] = tuple(values[field])
        return cls(**values)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
