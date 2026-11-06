"""Single-block asynchronous sequential prefetch with memory-pressure gating."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
import threading

MIN_BLOCK_BYTES = 4096
MAX_BLOCK_BYTES = 4 * 1024 * 1024


class SequentialAccessPredictor:
    """Predict the next block only after observing a stable positive stride twice."""

    def __init__(self, *, max_stride: int = 32):
        self.max_stride = max_stride
        self.previous: int | None = None
        self.stride: int | None = None
        self.stable_steps = 0

    def observe(self, block_index: int) -> int | None:
        if block_index < 0:
            raise ValueError("block_index cannot be negative")
        predicted = None
        if self.previous is not None:
            new_stride = block_index - self.previous
            if new_stride == self.stride:
                self.stable_steps += 1
            else:
                self.stride = new_stride
                self.stable_steps = 1
            if 0 < new_stride <= self.max_stride and self.stable_steps >= 2:
                predicted = block_index + new_stride
        self.previous = block_index
        return predicted

    def reset(self) -> None:
        self.previous = None
        self.stride = None
        self.stable_steps = 0


@dataclass
class PendingPrefetch:
    path: Path
    block_index: int
    block_size: int
    future: Future[bytes]


class PredictivePrefetchEngine:
    """One asynchronous read at a time; it never changes system cache settings."""

    def __init__(self, *, max_block_bytes: int = 4 * 1024 * 1024):
        if not MIN_BLOCK_BYTES <= max_block_bytes <= MAX_BLOCK_BYTES or max_block_bytes % MIN_BLOCK_BYTES:
            raise ValueError("max_block_bytes must be a 4 KiB multiple between 4 KiB and 4 MiB")
        self.max_block_bytes = max_block_bytes
        self.predictor = SequentialAccessPredictor()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvme-prefetch")
        self._lock = threading.Lock()
        self._pending: PendingPrefetch | None = None
        self.scheduled = 0
        self.useful_bytes = 0
        self.wasted_bytes = 0
        self.late = 0
        self.disabled = 0
        self.errors = 0
        self._closed = False

    @staticmethod
    def _read_at(path: Path, offset: int, size: int) -> bytes:
        with path.open("rb", buffering=0) as source:
            source.seek(offset)
            return source.read(size)

    def prefetch(
        self,
        file_path: str | Path,
        block_index: int,
        block_size: int,
        *,
        pressure: str,
        available_bytes: int,
        reserve_bytes: int,
    ) -> bool:
        """Schedule one file block only with normal pressure and enough headroom."""
        if self._closed:
            raise RuntimeError("prefetch engine is closed")
        if block_index < 0 or block_size < MIN_BLOCK_BYTES or block_size > self.max_block_bytes or block_size % MIN_BLOCK_BYTES:
            raise ValueError("block index/size is outside the configured aligned bounds")
        if available_bytes < 0 or reserve_bytes < 0:
            raise ValueError("memory values cannot be negative")
        if pressure != "normal" or available_bytes - reserve_bytes < block_size:
            self.disabled += 1
            return False

        path = Path(file_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Prefetch source must be a regular file: {path}")
        offset = block_index * block_size
        size = min(block_size, max(0, path.stat().st_size - offset))
        if size <= 0:
            return False

        with self._lock:
            if self._pending is not None:
                pending = self._pending
                if pending.path == path and pending.block_index == block_index and pending.block_size == block_size:
                    return False
                if pending.future.done():
                    try:
                        self.wasted_bytes += len(pending.future.result())
                    except Exception:
                        self.errors += 1
                    self._pending = None
                elif pending.future.cancel():
                    self._pending = None
                else:
                    self.disabled += 1
                    return False
            future = self._executor.submit(self._read_at, path, offset, size)
            self._pending = PendingPrefetch(path, block_index, block_size, future)
            self.scheduled += 1
            return True

    def observe_and_prefetch(
        self,
        file_path: str | Path,
        consumed_block_index: int,
        block_size: int,
        *,
        pressure: str,
        available_bytes: int,
        reserve_bytes: int,
    ) -> int | None:
        predicted = self.predictor.observe(consumed_block_index)
        if predicted is not None:
            self.prefetch(
                file_path,
                predicted,
                block_size,
                pressure=pressure,
                available_bytes=available_bytes,
                reserve_bytes=reserve_bytes,
            )
        return predicted

    def take(self, file_path: str | Path, block_index: int, block_size: int, *, wait_timeout_s: float = 0.0) -> bytes | None:
        """Return ready data; by default never waits and lets caller do normal I/O."""
        if not 0 <= wait_timeout_s <= 10:
            raise ValueError("wait_timeout_s must be between 0 and 10")
        path = Path(file_path).expanduser().resolve()
        with self._lock:
            pending = self._pending
            if pending is None:
                return None
            if pending.path != path or pending.block_index != block_index or pending.block_size != block_size:
                if pending.future.done():
                    try:
                        self.wasted_bytes += len(pending.future.result())
                    except Exception:
                        self.errors += 1
                    self._pending = None
                elif pending.future.cancel():
                    self._pending = None
                return None
            try:
                data = pending.future.result(timeout=wait_timeout_s)
            except TimeoutError:
                self.late += 1
                return None
            except Exception:
                self.errors += 1
                self._pending = None
                return None
            self._pending = None
            self.useful_bytes += len(data)
            return data

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": self._pending is not None,
                "scheduled": self.scheduled,
                "useful_bytes": self.useful_bytes,
                "wasted_bytes": self.wasted_bytes,
                "late": self.late,
                "disabled": self.disabled,
                "errors": self.errors,
                "max_block_bytes": self.max_block_bytes,
            }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
