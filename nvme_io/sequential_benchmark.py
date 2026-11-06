"""Bounded chunked sequential reads; never loads a test file all at once."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .metrics import summarize


def sequential_read(path: str | Path, *, block_bytes: int, on_poll: Callable[[], None] | None = None,
                    poll_interval_s: float = 1.0) -> dict:
    target = Path(path)
    if block_bytes < 4096:
        raise ValueError("block_bytes must be at least 4096")
    size = target.stat().st_size
    bytes_read = 0
    last_poll = time.perf_counter()
    started = time.perf_counter()
    with target.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(block_bytes)
            if not chunk:
                break
            bytes_read += len(chunk)
            now = time.perf_counter()
            if on_poll and now - last_poll >= poll_interval_s:
                on_poll()
                last_poll = now
    elapsed = time.perf_counter() - started
    return {"file_size_bytes": size, **summarize(bytes_read=bytes_read, elapsed_seconds=elapsed),
            "block_bytes": block_bytes, "complete": bytes_read == size}
