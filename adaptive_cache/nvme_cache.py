"""Content-addressed persistent cache restricted to its configured directory."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from threading import RLock

from .cache_index import CacheIndex
from .config import AdaptiveCacheConfig

_HASH = re.compile(r"^[0-9a-f]{64}$")


class NvmeCache:
    def __init__(self, config: AdaptiveCacheConfig, *, root: str | Path | None = None, disk_usage=shutil.disk_usage):
        self.config = config
        self.root = Path(root if root is not None else config.nvme_directory).expanduser().resolve()
        self._disk_usage = disk_usage
        self._lock = RLock()
        self.bytes_read = 0
        self.bytes_written = 0
        self.items_removed = 0
        self.corruptions = 0
        self.errors = 0
        self.last_read_seconds = 0.0
        self.last_write_seconds = 0.0
        if not config.enabled or not config.nvme_enabled:
            self.index = None
            self.recovered_index = False
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = CacheIndex(self.root / "index.sqlite3")
        self.recovered_index = self.index.recovered_from is not None
        if self.recovered_index:
            self._remove_owned_orphan_blobs()
        self._remove_stale_temp_files()

    def _key_hash(self, key) -> str:
        payload = key if isinstance(key, bytes) else str(key).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _blob_path(self, content_hash: str) -> Path:
        if not _HASH.fullmatch(content_hash):
            raise ValueError("Invalid content hash")
        path = self.root / f"{content_hash}.blob"
        if path.parent.resolve() != self.root:
            raise RuntimeError("Cache blob path escaped configured directory")
        return path

    def _is_owned_blob(self, path: Path) -> bool:
        return path.suffix == ".blob" and _HASH.fullmatch(path.stem) is not None and path.parent.resolve() == self.root

    def _remove_owned_orphan_blobs(self) -> None:
        # Index recovery may orphan only content-hash blobs owned by this cache.
        for path in self.root.glob("*.blob"):
            if self._is_owned_blob(path) and path.is_file() and not self.index.content_referenced(path.stem):
                path.unlink(missing_ok=True)

    def _remove_stale_temp_files(self) -> None:
        for path in self.root.glob(".adaptive-cache-tmp-*"):
            if path.is_file() and not path.is_symlink() and path.parent.resolve() == self.root:
                path.unlink(missing_ok=True)

    def _delete_blob_if_unreferenced(self, content_hash: str) -> None:
        if not _HASH.fullmatch(content_hash):
            return
        if self.index.content_referenced(content_hash):
            return
        path = self._blob_path(content_hash)
        if path.exists() and self._is_owned_blob(path) and path.is_file() and not path.is_symlink():
            path.unlink(missing_ok=True)

    def _expire(self, now: float) -> int:
        expired = self.index.remove_expired(now)
        for content_hash, _size in expired:
            self._delete_blob_if_unreferenced(content_hash)
        self.items_removed += len(expired)
        return len(expired)

    def _evict_until(self, required_bytes: int) -> bool:
        index_bytes = (self.root / "index.sqlite3").stat().st_size if (self.root / "index.sqlite3").exists() else 0
        maximum = max(0, self.config.nvme_max_mib * 1024**2 - index_bytes)
        minimum_free = self.config.nvme_min_free_mib * 1024**2
        for key_hash, content_hash, _size in self.index.eviction_candidates(self.config.eviction_policy):
            disk_free = self._disk_usage(self.root).free
            if self.index.total_size() + required_bytes <= maximum and disk_free - required_bytes >= minimum_free:
                return True
            self.index.remove(key_hash)
            self._delete_blob_if_unreferenced(content_hash)
            self.items_removed += 1
        disk_free = self._disk_usage(self.root).free
        return self.index.total_size() + required_bytes <= maximum and disk_free - required_bytes >= minimum_free

    def get(self, key) -> bytes | None:
        if self.index is None:
            return None
        started = time.perf_counter()
        with self._lock:
            key_hash = self._key_hash(key)
            record = self.index.get(key_hash, time.time())
            if record is None:
                return None
            content_hash = record["content_hash"]
            if record.get("expired"):
                self._delete_blob_if_unreferenced(content_hash)
                self.items_removed += 1
                return None
            try:
                path = self._blob_path(content_hash)
            except ValueError:
                self.index.remove(key_hash)
                self.corruptions += 1
                return None
            if record["size_bytes"] < 0 or record["size_bytes"] > self.config.nvme_item_max_mib * 1024**2:
                self.index.remove(key_hash)
                self._delete_blob_if_unreferenced(content_hash)
                self.corruptions += 1
                return None
            try:
                if not path.is_file() or path.is_symlink():
                    raise OSError("Cache blob missing")
                if path.stat().st_size != record["size_bytes"]:
                    self.corruptions += 1
                    raise OSError("Cache blob size mismatch")
                data = path.read_bytes()
            except OSError:
                self.index.remove(key_hash)
                self._delete_blob_if_unreferenced(content_hash)
                self.errors += 1
                return None
            if hashlib.sha256(data).hexdigest() != content_hash:
                self.index.remove(key_hash)
                self._delete_blob_if_unreferenced(content_hash)
                self.corruptions += 1
                return None
            self.bytes_read += len(data)
            self.last_read_seconds = time.perf_counter() - started
            return data

    def put(self, key, data: bytes) -> bool:
        if self.index is None:
            return False
        if not isinstance(data, bytes):
            raise TypeError("NVMe cache accepts immutable bytes only")
        if len(data) > self.config.nvme_item_max_mib * 1024**2:
            return False
        started = time.perf_counter()
        now = time.time()
        with self._lock:
            self._expire(now)
            key_hash = self._key_hash(key)
            content_hash = hashlib.sha256(data).hexdigest()
            blob_path = self._blob_path(content_hash)
            existing = self.index.get(key_hash, now)
            old_content = existing.get("content_hash") if existing and not existing.get("expired") else None

            if blob_path.exists():
                try:
                    valid_existing = blob_path.is_file() and not blob_path.is_symlink() and hashlib.sha256(blob_path.read_bytes()).hexdigest() == content_hash
                    if valid_existing:
                        self.bytes_read += blob_path.stat().st_size
                except OSError:
                    valid_existing = False
                if not valid_existing:
                    blob_path.unlink(missing_ok=True)
                    self.corruptions += 1

            writing_blob = not blob_path.exists()
            if writing_blob and not self._evict_until(len(data)):
                return False
            if writing_blob:
                temp_path = self.root / f".adaptive-cache-tmp-{uuid.uuid4().hex}"
                try:
                    with temp_path.open("xb") as target:
                        target.write(data)
                        target.flush()
                    os.replace(temp_path, blob_path)
                    self.bytes_written += len(data)
                except OSError:
                    self.errors += 1
                    temp_path.unlink(missing_ok=True)
                    raise
                finally:
                    temp_path.unlink(missing_ok=True)

            self.index.put(key_hash, content_hash, len(data), now, now + self.config.entry_ttl_seconds)
            if old_content and old_content != content_hash:
                self._delete_blob_if_unreferenced(old_content)
            self.last_write_seconds = time.perf_counter() - started
            return True

    def remove_expired(self) -> int:
        if self.index is None:
            return 0
        with self._lock:
            return self._expire(time.time())

    def stats(self) -> dict:
        if self.index is None:
            return {"enabled": False, "entries": 0, "used_bytes": 0, "blob_bytes": 0,
                    "limit_bytes": 0, "free_bytes": 0, "minimum_free_bytes": 0,
                    "bytes_read": 0, "bytes_written": 0, "items_removed": 0,
                    "corruptions": 0, "index_recovered": False}
        with self._lock:
            blob_bytes = self.index.total_size()
            index_bytes = (self.root / "index.sqlite3").stat().st_size if (self.root / "index.sqlite3").exists() else 0
            usage = self._disk_usage(self.root)
            return {
                "enabled": bool(self.config.enabled and self.config.nvme_enabled),
                "entries": self.index.count(),
                "used_bytes": blob_bytes + index_bytes,
                "blob_bytes": blob_bytes,
                "limit_bytes": self.config.nvme_max_mib * 1024**2,
                "free_bytes": usage.free,
                "minimum_free_bytes": self.config.nvme_min_free_mib * 1024**2,
                "bytes_read": self.bytes_read,
                "bytes_written": self.bytes_written,
                "items_removed": self.items_removed,
                "corruptions": self.corruptions,
                "errors": self.errors,
                "index_recovered": self.recovered_index,
                "read_seconds": self.last_read_seconds,
                "write_seconds": self.last_write_seconds,
            }
