"""Seeded random read measurements with per-operation latency samples."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable

from .metrics import summarize


def random_read(path: str | Path, *, block_bytes: int, operations: int, seed: int = 20261106,
                on_poll: Callable[[], None] | None = None, poll_interval_s: float = 1.0) -> dict:
    target = Path(path)
    file_size = target.stat().st_size
    if block_bytes not in {4096, 65536}:
        raise ValueError("block_bytes must be 4 KiB or 64 KiB")
    if operations < 1 or file_size < block_bytes:
        raise ValueError("file must fit one block and operations must be positive")
    rng = random.Random(seed)
    max_offset = file_size - block_bytes
    offsets = [rng.randrange(max_offset + 1) for _ in range(operations)]
    latencies = []
    total_bytes = 0
    last_poll = time.perf_counter()
    started = time.perf_counter()
    with target.open("rb", buffering=0) as handle:
        for offset in offsets:
            operation_started = time.perf_counter()
            handle.seek(offset, os.SEEK_SET)
            data = handle.read(block_bytes)
            latencies.append((time.perf_counter() - operation_started) * 1000.0)
            if len(data) != block_bytes:
                break
            total_bytes += len(data)
            now = time.perf_counter()
            if on_poll and now - last_poll >= poll_interval_s:
                on_poll()
                last_poll = now
    elapsed = time.perf_counter() - started
    return {"file_size_bytes": file_size, "block_bytes": block_bytes,
            **summarize(bytes_read=total_bytes, elapsed_seconds=elapsed, operations=total_bytes // block_bytes,
                        latencies_ms=latencies),
            "complete": total_bytes == operations * block_bytes}
