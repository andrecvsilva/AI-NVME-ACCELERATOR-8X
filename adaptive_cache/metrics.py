"""Bounded counters and append-only operation logs for Adaptive Cache."""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_FIELDS = ["timestamp_utc", "operation", "key_hash", "result", "bytes_read", "bytes_written",
           "items_removed", "duration_seconds", "memory_state", "ram_used_bytes", "nvme_used_bytes", "error"]


class AdaptiveCacheMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.items_removed = 0
        self.read_seconds = 0.0
        self.write_seconds = 0.0
        self.corruptions = 0
        self.errors = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def increment(self, metric: str, amount: int | float = 1) -> None:
        if metric not in {"hits", "misses", "bytes_read", "bytes_written", "items_removed", "read_seconds",
                          "write_seconds", "corruptions", "errors", "prefetch_hits", "prefetch_misses"}:
            raise ValueError(f"Unknown cache metric: {metric}")
        with self._lock:
            setattr(self, metric, getattr(self, metric) + amount)

    def snapshot(self) -> dict:
        with self._lock:
            result = {key: getattr(self, key) for key in (
                "hits", "misses", "bytes_read", "bytes_written", "items_removed", "read_seconds",
                "write_seconds", "corruptions", "errors", "prefetch_hits", "prefetch_misses")}
            result["hit_rate"] = round(self.hits / (self.hits + self.misses), 4) if self.hits + self.misses else None
            return result


class CacheEventLogger:
    def __init__(self, logs_dir: str | Path):
        self.directory = Path(logs_dir)
        self.json_path = self.directory / "adaptive_cache.json"
        self.csv_path = self.directory / "adaptive_cache.csv"
        self._lock = threading.Lock()

    def append(self, event: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock:
            if not self.json_path.exists():
                self.json_path.write_text("[\n" + json.dumps(record, ensure_ascii=False) + "\n]\n", encoding="utf-8")
            else:
                with self.json_path.open("r+b") as handle:
                    handle.seek(0, 2)
                    end = handle.tell()
                    cursor = end - 1
                    while cursor >= 0:
                        handle.seek(cursor)
                        char = handle.read(1)
                        if char not in b" \t\r\n":
                            break
                        cursor -= 1
                    if cursor < 0 or char != b"]":
                        raise RuntimeError(f"Refusing to append to malformed cache log: {self.json_path}")
                    handle.seek(0)
                    if handle.read(1) != b"[":
                        raise RuntimeError(f"Refusing to append to non-array cache log: {self.json_path}")
                    previous = cursor - 1
                    while previous >= 0:
                        handle.seek(previous)
                        previous_char = handle.read(1)
                        if previous_char not in b" \t\r\n":
                            break
                        previous -= 1
                    handle.seek(cursor)
                    handle.truncate()
                    separator = b"\n" if previous_char == b"[" else b",\n"
                    handle.write(separator + json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n]\n")

            row = {key: record.get(key, "") for key in _FIELDS}
            new_csv = not self.csv_path.exists()
            with self.csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_FIELDS, extrasaction="ignore")
                if new_csv:
                    writer.writeheader()
                writer.writerow(row)
