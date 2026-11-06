"""Opt-in, bounded NVMe reads guarded by the existing memory policy."""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from adaptive_cache import AdaptiveCacheConfig
from adaptive_cache.memory_policy import AdaptiveMemoryPolicy
from memory_guard import MemoryGuard

from .config import NvmeIoConfig
from .device_info import get_device_info
from .metrics import summarize
from .random_benchmark import random_read
from .report import NvmeIoReporter
from .sequential_benchmark import sequential_read

MIB = 1024 * 1024


class BenchmarkBusyError(RuntimeError):
    """Another process currently holds the project's NVMe benchmark lock."""


class NvmeBenchmarkManager:
    def __init__(self, config: NvmeIoConfig | None = None, *, project_root: str | Path = ".",
                 memory_guard: MemoryGuard | None = None, adaptive_policy: AdaptiveMemoryPolicy | None = None,
                 reporter: NvmeIoReporter | None = None, device_provider=get_device_info,
                 disk_usage=shutil.disk_usage, file_writer=None, sequential_runner=sequential_read,
                 random_runner=random_read):
        self.config = config or NvmeIoConfig()
        self.project_root = Path(project_root).resolve()
        self.memory_guard = memory_guard or MemoryGuard()
        if adaptive_policy is None:
            adaptive_config_path = self.project_root / "config" / "adaptive_cache.json"
            adaptive_config = (AdaptiveCacheConfig.load(adaptive_config_path) if adaptive_config_path.is_file()
                               else AdaptiveCacheConfig())
            adaptive_policy = AdaptiveMemoryPolicy(adaptive_config, memory_guard=self.memory_guard)
        self.adaptive_policy = adaptive_policy
        self.reporter = reporter or NvmeIoReporter(self.project_root / "logs", self.project_root / "reports")
        self.device_provider = device_provider
        self.disk_usage = disk_usage
        self._default_file_writer = file_writer is None
        self.file_writer = file_writer or self._write_test_file
        self.sequential_runner = sequential_runner
        self.random_runner = random_runner
        self._run_lock = threading.Lock()

    def device_info(self) -> dict:
        return self.device_provider(self.config.drive)

    def _memory_snapshot(self) -> dict:
        return self.memory_guard.inspect().snapshot.to_dict()

    @staticmethod
    def _write_test_file(path: Path, size_bytes: int, *, block_bytes: int = MIB,
                         on_poll=None, poll_interval_s: float = 1.0) -> int:
        pattern = bytes(range(256)) * 4096
        written = 0
        with path.open("xb", buffering=0) as handle:
            while written < size_bytes:
                chunk_size = min(block_bytes, size_bytes - written)
                handle.write(pattern[:chunk_size])
                written += chunk_size
                if on_poll:
                    on_poll()
            handle.flush()
            os.fsync(handle.fileno())
        return written

    def _safe_temp_root(self) -> Path:
        candidate = Path(self.config.temp_directory)
        base = candidate if candidate.is_absolute() else self.project_root / candidate
        raw_base = base
        try:
            relative = raw_base.absolute().relative_to(self.project_root)
        except ValueError:
            relative = None
        if relative is not None:
            cursor = self.project_root
            for part in relative.parts:
                cursor = cursor / part
                if cursor.exists() and cursor.is_symlink():
                    raise ValueError("Temporary benchmark path cannot contain symlinks.")
        base = base.resolve()
        try:
            base.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("Temporary benchmark directory must stay inside the project.") from exc
        if base == self.project_root or base.is_symlink():
            raise ValueError("Unsafe temporary benchmark directory.")
        if self.project_root.drive.upper() != self.config.drive.upper():
            raise ValueError("Project directory is not on the configured benchmark drive; refusing to test another volume.")
        return base

    @staticmethod
    def _lock_file(handle, acquire: bool) -> None:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if acquire:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise BenchmarkBusyError("Another NVMe I/O benchmark is already active for this project.") from exc
            else:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl
            if acquire:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise BenchmarkBusyError("Another NVMe I/O benchmark is already active for this project.") from exc
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _cleanup_owned_run(run_dir: Path | None, run_id: str, owned_files: list[Path], marker: Path | None) -> None:
        if run_dir is None or marker is None:
            return
        try:
            if run_dir.is_symlink() or marker.is_symlink():
                return
            resolved = run_dir.resolve(strict=True)
            parent = resolved.parent.resolve(strict=True)
            if marker.resolve(strict=True).parent != resolved or run_dir.parent.resolve(strict=True) != parent:
                return
            if marker.read_text(encoding="utf-8") != run_id:
                return
            for path in owned_files:
                if path.parent.resolve(strict=True) != resolved or not path.name.startswith(f"nvmeio-{run_id}-"):
                    continue
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            if set(resolved.iterdir()) != {marker}:
                return
            marker.unlink(missing_ok=True)
            resolved.rmdir()  # Fails safely if any unrecognized file remains.
            parent  # Keep verified parent alive in the audit path; no recursive deletion is used.
        except OSError:
            # Preserve unexpected contents for manual inspection; never recurse into the directory.
            return

    @classmethod
    def _cleanup_stale_runs(cls, base: Path) -> None:
        """Remove only abandoned, marker-validated files created by this module."""
        resolved_base = base.resolve(strict=True)
        for candidate in resolved_base.glob("run-*"):
            if candidate.is_symlink() or not candidate.is_dir() or candidate.parent.resolve() != resolved_base:
                continue
            marker = candidate / ".nvme-io-owned"
            try:
                run_id = marker.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if len(run_id) != 32 or not candidate.name.startswith(f"run-{run_id}-"):
                continue
            owned = list(candidate.glob(f"nvmeio-{run_id}-*mib.bin"))
            cls._cleanup_owned_run(candidate, run_id, owned, marker)

    def _check_during(self, initial_state: str, samples: list[dict]) -> None:
        try:
            decision = self.memory_guard.inspect()
        except Exception as exc:
            raise MemoryError(f"Memory Guard became unavailable during I/O; stopping: {exc}") from exc
        snapshot = decision.snapshot.to_dict()
        samples.append({"guard_state": decision.state, "memory": snapshot})
        if decision.state == "CRITICAL" or decision.state not in {"NORMAL", "WARNING"}:
            raise MemoryError(f"Memory Guard changed to {decision.state}; stopping additional I/O.")
        if initial_state == "NORMAL" and decision.state == "WARNING":
            raise MemoryError("Memory Guard changed to WARNING; stopping additional I/O.")

    def run_benchmark(self) -> dict:
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        event = {"timestamp_utc": started_at, "run_id": run_id, "status": "failed", "operation": "nvme_io_run",
                 "memory_guard_state": "UNKNOWN", "device": {}, "files": [], "memory": {"before": None, "during": [], "after": None},
                 "windows_cache_state": "Windows file cache remains enabled; a cold device read is not guaranteed.",
                 "error": None, "inference_started": False}
        run_dir = None
        marker = None
        lock_handle = None
        owned_files: list[Path] = []
        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            event["status"] = "blocked"
            event["error"] = "Another NVMe I/O benchmark is already active."
        try:
            if not acquired:
                return event
            policy = self.adaptive_policy.evaluate()
            event["memory_guard_state"] = policy.guard_state
            if not policy.available or policy.guard_state not in {"NORMAL", "WARNING"}:
                event["status"] = "blocked"
                event["error"] = policy.reason or f"Memory Guard {policy.guard_state} blocks the benchmark."
                return event
            event["memory"]["before"] = self._memory_snapshot()
            event["device"] = self.device_info()
            base = self._safe_temp_root()
            base.mkdir(parents=True, exist_ok=True)
            lock_path = base / ".nvme-io.lock"
            lock_handle = lock_path.open("a+b")
            if lock_path.stat().st_size == 0:
                lock_handle.write(b"0")
                lock_handle.flush()
            self._lock_file(lock_handle, acquire=True)
            self._cleanup_stale_runs(base)
            max_file_mib = max(self.config.test_sizes_mib)
            if policy.guard_state == "WARNING":
                max_file_mib = min(max_file_mib, self.config.warning_max_file_mib)
                sizes = sorted({min(value, max_file_mib) for value in self.config.test_sizes_mib})
                random_operations = self.config.warning_random_operations
            else:
                sizes = list(self.config.test_sizes_mib)
                random_operations = self.config.random_operations
            if sum(sizes) > self.config.maximum_total_temp_mib:
                raise ValueError("Selected benchmark files exceed the configured total temporary-write cap.")
            free_bytes = self.disk_usage(base).free
            required_free = max(sizes) * MIB + self.config.minimum_free_mib * MIB
            event["preflight"] = {"free_bytes": free_bytes, "required_free_bytes": required_free,
                                  "temporary_write_cap_bytes": self.config.maximum_total_temp_mib * MIB,
                                  "selected_file_sizes_mib": sizes}
            if free_bytes < required_free:
                raise OSError("Insufficient free disk space for the bounded benchmark file and configured reserve.")

            run_dir = Path(__import__("tempfile").mkdtemp(prefix=f"run-{run_id}-", dir=base))
            marker = run_dir / ".nvme-io-owned"
            marker.write_text(run_id, encoding="utf-8")
            initial_state = policy.guard_state
            memory_samples = event["memory"]["during"]
            last_poll = time.perf_counter()

            def poll_memory(force: bool = False) -> None:
                nonlocal last_poll
                now = time.perf_counter()
                if force or now - last_poll >= self.config.memory_poll_seconds:
                    self._check_during(initial_state, memory_samples)
                    last_poll = now

            for size_mib in sizes:
                poll_memory(force=True)
                file_path = run_dir / f"nvmeio-{run_id}-{size_mib}mib.bin"
                owned_files.append(file_path)
                expected_bytes = size_mib * MIB
                file_result = {"expected_file_size_bytes": expected_bytes, "file_size_bytes": expected_bytes,
                               "temporary_bytes_written": 0, "status": "running", "random": {}}
                event["files"].append(file_result)
                if self._default_file_writer:
                    written = self.file_writer(file_path, expected_bytes, on_poll=poll_memory,
                                               poll_interval_s=self.config.memory_poll_seconds)
                else:
                    written = self.file_writer(file_path, expected_bytes)
                file_size = file_path.stat().st_size
                file_result.update({"file_size_bytes": file_size, "temporary_bytes_written": written})
                if written != expected_bytes or file_size != expected_bytes:
                    file_result.update({"status": "incomplete", "error": "Generated test file size did not match the requested size."})
                    raise OSError(file_result["error"])
                poll_memory(force=True)  # Recheck after writing and before the first read phase.
                first = self.sequential_runner(file_path, block_bytes=self.config.sequential_block_kib * 1024,
                                               on_poll=poll_memory, poll_interval_s=self.config.memory_poll_seconds)
                file_result["sequential_first"] = {**first, "windows_cache_state": "first pass; cache state unknown"}
                if not first.get("complete") or first.get("bytes_read") != file_size:
                    file_result.update({"status": "incomplete", "error": "First sequential pass returned a short read."})
                    raise OSError(file_result["error"])
                poll_memory(force=True)
                repeat = self.sequential_runner(file_path, block_bytes=self.config.sequential_block_kib * 1024,
                                                on_poll=poll_memory, poll_interval_s=self.config.memory_poll_seconds)
                file_result["sequential_repeat"] = {**repeat, "windows_cache_state": "repeat pass; Windows cache may serve reads"}
                if not repeat.get("complete") or repeat.get("bytes_read") != file_size:
                    file_result.update({"status": "incomplete", "error": "Repeated sequential pass returned a short read."})
                    raise OSError(file_result["error"])
                for block_kib in self.config.random_block_kib:
                    poll_memory(force=True)
                    actual_ops = min(random_operations, file_size // (block_kib * 1024))
                    result = self.random_runner(file_path, block_bytes=block_kib * 1024, operations=actual_ops,
                                                seed=20261106 + block_kib + size_mib,
                                                on_poll=poll_memory, poll_interval_s=self.config.memory_poll_seconds)
                    result["windows_cache_state"] = "Windows file cache left enabled; source may already be cached"
                    file_result["random"][f"{block_kib}k"] = result
                    if not result.get("complete") or result.get("operations") != actual_ops:
                        file_result.update({"status": "incomplete", "error": f"Random {block_kib} KiB phase returned incomplete reads."})
                        raise OSError(file_result["error"])
                file_result["status"] = "complete"
                poll_memory(force=True)
            event["memory"]["after"] = self._memory_snapshot()
            event["status"] = "completed"
            return event
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            event["status"] = "blocked" if isinstance(exc, (MemoryError, BenchmarkBusyError)) else "failed"
            event["error"] = str(exc)
            for file_result in event["files"]:
                if file_result.get("status") == "running":
                    file_result.update({"status": "incomplete", "error": str(exc)})
            try:
                event["memory"]["after"] = self._memory_snapshot()
            except Exception:
                pass
            return event
        except KeyboardInterrupt:
            event["status"] = "interrupted"
            event["error"] = "Benchmark interrupted by user; owned temporary files will be cleaned and the report preserved."
            for file_result in event["files"]:
                if file_result.get("status") == "running":
                    file_result.update({"status": "incomplete", "error": event["error"]})
            try:
                event["memory"]["after"] = self._memory_snapshot()
            except Exception:
                pass
            return event
        finally:
            event["finished_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                self.reporter.write_report(event)
                self.reporter.append(event)
                for file_result in event.get("files", []):
                    for operation in ("sequential_first", "sequential_repeat"):
                        measure = file_result.get(operation) or {}
                        if not measure:
                            continue
                        self.reporter.append({"run_id": run_id, "status": event["status"], "drive": self.config.drive,
                            "device_model": event.get("device", {}).get("device_model"),
                            "bus_type_reported": event.get("device", {}).get("bus_type_reported"),
                            "memory_guard_state": event.get("memory_guard_state"), "operation": operation,
                            "file_size_bytes": file_result.get("file_size_bytes"), "block_bytes": measure.get("block_bytes"),
                            "operations": measure.get("operations", 0), "bytes_read": measure.get("bytes_read"),
                            "elapsed_seconds": measure.get("elapsed_seconds"), "throughput_mb_s": measure.get("throughput_mb_s"),
                            "throughput_mib_s": measure.get("throughput_mib_s"),
                            "iops": measure.get("iops"), "windows_cache_state": measure.get("windows_cache_state"),
                            "result": measure, "error": event.get("error")})
                    for label, measure in (file_result.get("random") or {}).items():
                        if not measure:
                            continue
                        self.reporter.append({"run_id": run_id, "status": event["status"], "drive": self.config.drive,
                            "device_model": event.get("device", {}).get("device_model"),
                            "bus_type_reported": event.get("device", {}).get("bus_type_reported"),
                            "memory_guard_state": event.get("memory_guard_state"), "operation": f"random_{label}",
                            "file_size_bytes": file_result.get("file_size_bytes"), "block_bytes": measure.get("block_bytes"),
                            "operations": measure.get("operations"), "bytes_read": measure.get("bytes_read"),
                            "elapsed_seconds": measure.get("elapsed_seconds"), "throughput_mb_s": measure.get("throughput_mb_s"),
                            "throughput_mib_s": measure.get("throughput_mib_s"),
                            "iops": measure.get("iops"), "latency_mean_ms": measure.get("latency_mean_ms"),
                            "latency_p50_ms": measure.get("latency_p50_ms"), "latency_p95_ms": measure.get("latency_p95_ms"),
                            "latency_p99_ms": measure.get("latency_p99_ms"), "windows_cache_state": measure.get("windows_cache_state"),
                            "result": measure, "error": event.get("error")})
            except (OSError, RuntimeError, ValueError):
                pass
            self._cleanup_owned_run(run_dir, run_id, owned_files, marker)
            if lock_handle is not None:
                self._lock_file(lock_handle, acquire=False)
                lock_handle.close()
            if acquired:
                self._run_lock.release()
