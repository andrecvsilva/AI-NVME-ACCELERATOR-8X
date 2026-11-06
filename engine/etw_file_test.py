"""Bounded ETW attribution probe using a small, owned, unbuffered-read file."""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Callable

from engine.etw_io_trace import WprEtwCapture
from memory_guard import MemoryGuard

MIB = 1024 * 1024
_BLOCK_BYTES = MIB
_MAX_SIZE_MIB = 16


def _write_probe_file(path: Path, size_mib: int) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        remaining = size_mib * MIB
        while remaining:
            block = os.urandom(min(MIB, remaining))
            stream.write(block)
            digest.update(block)
            remaining -= len(block)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def _read_unbuffered_windows(path: Path, expected_bytes: int) -> tuple[int, str]:
    """Read aligned blocks with FILE_FLAG_NO_BUFFERING to request physical I/O."""
    if os.name != "nt":
        raise OSError("Unbuffered ETW file test is supported only on Windows.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle_t = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, handle_t]
    kernel32.CreateFileW.restype = handle_t
    kernel32.ReadFile.argtypes = [handle_t, ctypes.c_void_p, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    kernel32.ReadFile.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [handle_t]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel32.VirtualFree.restype = ctypes.c_int

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x1
    FILE_SHARE_WRITE = 0x2
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    MEM_COMMIT_RESERVE = 0x3000
    PAGE_READWRITE = 0x04
    MEM_RELEASE = 0x8000
    invalid = ctypes.c_void_p(-1).value

    handle = kernel32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  None, OPEN_EXISTING,
                                  FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = kernel32.VirtualAlloc(None, _BLOCK_BYTES, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    if not buffer:
        kernel32.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        while read_bytes < expected_bytes:
            request = min(_BLOCK_BYTES, expected_bytes - read_bytes)
            count = ctypes.c_uint32()
            if not kernel32.ReadFile(handle, buffer, request, ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if count.value != request:
                raise OSError(f"Short unbuffered read: expected {request}, got {count.value}.")
            digest.update(ctypes.string_at(buffer, count.value))
            read_bytes += count.value
    finally:
        kernel32.VirtualFree(buffer, 0, MEM_RELEASE)
        kernel32.CloseHandle(handle)
    return read_bytes, digest.hexdigest()


def run_etw_file_test(logs_dir: str | Path, *, size_mib: int = 4,
                      capture_factory: Callable = WprEtwCapture,
                      unbuffered_reader: Callable = _read_unbuffered_windows,
                      memory_guard=None,
                      windows_probe: Callable[[], bool] | None = None) -> dict:
    """Write/read a 1-16 MiB temporary file, capture WPR, and safely remove own data."""
    if not 1 <= size_mib <= _MAX_SIZE_MIB:
        raise ValueError(f"size_mib must be between 1 and {_MAX_SIZE_MIB}.")
    is_windows = windows_probe() if windows_probe else os.name == "nt"
    if not is_windows:
        raise OSError("ETW file test is supported only on Windows.")
    guard = memory_guard or MemoryGuard()
    admission = guard.require_admission()

    logs_root = Path(logs_dir).resolve()
    run_dir = (logs_root / f"etw_probe_{uuid.uuid4().hex[:10]}").resolve()
    probe_path = (run_dir / "io_probe.bin").resolve()
    if logs_root not in run_dir.parents or run_dir in logs_root.parents:
        raise ValueError("Unsafe ETW probe path; refusing to create files outside logs directory.")
    capture_kwargs = {"target_file_path": probe_path}
    if capture_factory is WprEtwCapture:
        # A whole-system CSV export can be hundreds of MiB and is not needed
        # for the bounded file-attribution probe.
        capture_kwargs["export_csv"] = False
    capture = capture_factory(logs_root, **capture_kwargs)
    preflight = capture.preflight()
    if not preflight.get("available"):
        return {"status": "blocked", "phase": "preflight", "reason": preflight.get("reason"),
                "test_file_created": False}

    run_dir.mkdir(parents=True, exist_ok=False)
    trace_started = False
    try:
        started = capture.start()
        if not started.get("started"):
            return {"status": "blocked", "phase": "capture_start",
                    "reason": started.get("reason"), "test_file_created": False}
        trace_started = True
        expected_bytes = size_mib * MIB
        expected_hash = _write_probe_file(probe_path, size_mib)
        read_started = time.perf_counter()
        read_bytes, actual_hash = unbuffered_reader(probe_path, expected_bytes)
        read_seconds = time.perf_counter() - read_started
        capture_result = capture.stop()
        trace_started = False
        analysis = capture_result.get("target_file_analysis")
        return {
            "status": "passed" if read_bytes == expected_bytes and actual_hash == expected_hash
                      and bool(analysis and analysis.get("available")
                               and analysis.get("physical_file_read_bytes", 0) > 0) else "incomplete",
            "memory_state_before": admission.state,
            "test_file_bytes": expected_bytes,
            "unbuffered_read_bytes": read_bytes,
            "read_seconds": round(read_seconds, 6),
            "integrity_sha256_matches": actual_hash == expected_hash,
            "target_file_etw_analysis": analysis,
            "etw_capture": capture_result,
            "capture_start": started,
            "test_file_path": str(probe_path),
            "test_file_cleanup": "attempted after successful capture/export; ETL/XML/CSV artifacts are retained",
        }
    except Exception as exc:
        result = {"status": "failed", "phase": "write_or_read_or_capture",
                  "error": f"{type(exc).__name__}: {exc}", "test_file_path": str(probe_path)}
        if trace_started:
            try:
                result["etw_capture"] = capture.stop()
            except Exception as stop_exc:
                result["capture_stop_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        return result
    finally:
        # Only this unique, verified run directory can be cleaned; trace artifacts are elsewhere.
        if logs_root in run_dir.parents and run_dir.is_dir():
            try:
                probe_path.unlink(missing_ok=True)
                run_dir.rmdir()
            except OSError:
                pass
