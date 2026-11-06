"""Bounded read-only file I/O measurements; no test-file creation or writes."""

from __future__ import annotations

import math
import ctypes
import os
import random
import statistics
import time
from pathlib import Path

MAX_READ_MIB = 1024
RANDOM_BLOCK_BYTES = 4096
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
MEM_RESERVE = 0x2000
MEM_COMMIT = 0x1000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _unbuffered_sequential_read(path: Path, max_bytes: int, block_bytes: int) -> tuple[int, float, int]:
    """Read with Windows FILE_FLAG_NO_BUFFERING using aligned VirtualAlloc memory."""
    if os.name != "nt":
        raise RuntimeError("Unbuffered reads require Windows FILE_FLAG_NO_BUFFERING.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle_t = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, handle_t]
    kernel32.CreateFileW.restype = handle_t
    kernel32.GetDiskFreeSpaceW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetDiskFreeSpaceW.restype = ctypes.c_int
    kernel32.SetFilePointerEx.argtypes = [handle_t, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), ctypes.c_uint32]
    kernel32.SetFilePointerEx.restype = ctypes.c_int
    kernel32.ReadFile.argtypes = [handle_t, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    kernel32.ReadFile.restype = ctypes.c_int
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel32.VirtualFree.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [handle_t]
    kernel32.CloseHandle.restype = ctypes.c_int

    sectors_per_cluster = ctypes.c_uint32()
    bytes_per_sector = ctypes.c_uint32()
    free_clusters = ctypes.c_uint32()
    total_clusters = ctypes.c_uint32()
    root = path.anchor or str(path.parent)
    if not kernel32.GetDiskFreeSpaceW(root, ctypes.byref(sectors_per_cluster), ctypes.byref(bytes_per_sector), ctypes.byref(free_clusters), ctypes.byref(total_clusters)):
        raise OSError(ctypes.get_last_error(), "GetDiskFreeSpaceW failed")
    alignment = max(4096, int(bytes_per_sector.value))
    if block_bytes % alignment:
        raise ValueError(f"block size must be a multiple of the volume alignment ({alignment} bytes)")
    read_limit = min(max_bytes, path.stat().st_size)
    read_limit -= read_limit % alignment
    if read_limit <= 0:
        raise ValueError("File is too small for an aligned unbuffered read.")

    handle = kernel32.CreateFileW(
        str(path), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise OSError(ctypes.get_last_error(), f"Could not open file for unbuffered read: {path}")
    buffer = kernel32.VirtualAlloc(None, block_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
    if not buffer:
        kernel32.CloseHandle(handle)
        raise OSError(ctypes.get_last_error(), "VirtualAlloc failed for aligned read buffer")

    bytes_read_total = 0
    started = time.perf_counter()
    try:
        remaining = read_limit
        while remaining:
            request_bytes = min(block_bytes, remaining)
            read_now = ctypes.c_uint32()
            if not kernel32.ReadFile(handle, buffer, request_bytes, ctypes.byref(read_now), None):
                raise OSError(ctypes.get_last_error(), "Unbuffered ReadFile failed")
            if read_now.value != request_bytes:
                raise OSError(f"Short unbuffered read: requested {request_bytes}, received {read_now.value}")
            bytes_read_total += read_now.value
            remaining -= read_now.value
        elapsed = time.perf_counter() - started
    finally:
        kernel32.VirtualFree(buffer, 0, MEM_RELEASE)
        kernel32.CloseHandle(handle)
    return bytes_read_total, elapsed, alignment


def benchmark_file(
    file_path: str | Path,
    *,
    max_mib: int = 256,
    block_kib: int = 1024,
    random_ops: int = 128,
    seed: int = 20261105,
) -> dict:
    """Measure sequential and 4 KiB random reads from an existing file.

    The benchmark reads no more than 1 GiB sequentially and 4096 random blocks.
    Windows' file cache is left enabled, so results can include cached reads.
    """
    path = Path(file_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    if not 1 <= max_mib <= MAX_READ_MIB:
        raise ValueError(f"max_mib must be between 1 and {MAX_READ_MIB}")
    if not 4 <= block_kib <= 4096:
        raise ValueError("block_kib must be between 4 and 4096")
    if not 0 <= random_ops <= 4096:
        raise ValueError("random_ops must be between 0 and 4096")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("Cannot benchmark an empty file.")
    max_bytes = max_mib * 1024 * 1024
    read_limit = min(file_size, max_bytes)
    block_bytes = block_kib * 1024
    sequential_bytes = 0
    sequential_start = time.perf_counter()
    with path.open("rb", buffering=0) as source:
        while sequential_bytes < read_limit:
            chunk = source.read(min(block_bytes, read_limit - sequential_bytes))
            if not chunk:
                break
            sequential_bytes += len(chunk)
    sequential_seconds = time.perf_counter() - sequential_start

    read_count = min(random_ops, file_size // RANDOM_BLOCK_BYTES)
    latencies_ms: list[float] = []
    if read_count:
        upper = file_size - RANDOM_BLOCK_BYTES
        rng = random.Random(seed)
        offsets = [rng.randrange(0, upper // RANDOM_BLOCK_BYTES + 1) * RANDOM_BLOCK_BYTES for _ in range(read_count)]
        with path.open("rb", buffering=0) as source:
            for offset in offsets:
                started = time.perf_counter()
                source.seek(offset, os.SEEK_SET)
                chunk = source.read(RANDOM_BLOCK_BYTES)
                latencies_ms.append((time.perf_counter() - started) * 1000)
                if len(chunk) != RANDOM_BLOCK_BYTES:
                    break

    random_seconds = sum(latencies_ms) / 1000
    return {
        "file": str(path),
        "file_size_bytes": file_size,
        "sequential": {
            "bytes_read": sequential_bytes,
            "block_bytes": block_bytes,
            "elapsed_ms": round(sequential_seconds * 1000, 3),
            "throughput_mib_s": round(sequential_bytes / (1024 * 1024) / sequential_seconds, 3) if sequential_seconds else None,
        },
        "random_4k": {
            "operations": len(latencies_ms),
            "bytes_read": len(latencies_ms) * RANDOM_BLOCK_BYTES,
            "elapsed_ms_sum": round(random_seconds * 1000, 3),
            "throughput_mib_s": round((len(latencies_ms) * RANDOM_BLOCK_BYTES) / (1024 * 1024) / random_seconds, 3) if random_seconds else None,
            "latency_ms_mean": round(statistics.mean(latencies_ms), 3) if latencies_ms else None,
            "latency_ms_p50": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
            "latency_ms_p95": round(_percentile(latencies_ms, 0.95), 3) if latencies_ms else None,
        },
        "limitations": [
            "Read-only: no file or benchmark data is written.",
            "The Windows file cache remains enabled; cold-NVMe performance is not isolated.",
            "Use the GGUF model file on the NVMe for the most relevant workload measurement.",
            "This measures storage reads only, not LLM token generation speed.",
        ],
    }


def benchmark_block_sweep(
    file_path: str | Path,
    *,
    mib_per_block: int = 32,
    block_sizes_kib: tuple[int, ...] = (4, 64, 256, 1024, 4096),
    bypass_cache: bool = False,
) -> dict:
    """Compare bounded sequential reads using several block sizes.

    Maximum total bytes read is 320 MiB (64 MiB per size, five sizes). All passes
    begin at offset zero and Windows caching remains enabled, so later passes may
    be served from cache. Results are exploratory, not a cold-device benchmark.
    """
    if not 1 <= mib_per_block <= 64:
        raise ValueError("mib_per_block must be between 1 and 64")
    if not block_sizes_kib or len(block_sizes_kib) > 5:
        raise ValueError("block_sizes_kib must contain 1..5 block sizes")
    if any(size < 4 or size > 4096 for size in block_sizes_kib):
        raise ValueError("each block size must be between 4 and 4096 KiB")
    path = Path(file_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Benchmark source must be a non-empty existing file: {path}")

    passes = []
    for block_kib in block_sizes_kib:
        if bypass_cache:
            bytes_read, elapsed_seconds, alignment = _unbuffered_sequential_read(
                path, mib_per_block * 1024 * 1024, block_kib * 1024
            )
            sequential = {
                "bytes_read": bytes_read,
                "elapsed_ms": round(elapsed_seconds * 1000, 3),
                "throughput_mib_s": round(bytes_read / (1024 * 1024) / elapsed_seconds, 3) if elapsed_seconds else None,
            }
        else:
            result = benchmark_file(path, max_mib=mib_per_block, block_kib=block_kib, random_ops=0)
            sequential = result["sequential"]
            alignment = None
        passes.append({
            "block_kib": block_kib,
            "bytes_read": sequential["bytes_read"],
            "elapsed_ms": sequential["elapsed_ms"],
            "throughput_mib_s": sequential["throughput_mib_s"],
            "volume_alignment_bytes": alignment,
        })
    return {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "mib_per_block_size": mib_per_block,
        "maximum_total_bytes_read": mib_per_block * len(block_sizes_kib) * 1024 * 1024,
        "sequential_block_sweep": passes,
        "cache_state": "Windows FILE_FLAG_NO_BUFFERING enabled." if bypass_cache else "Windows file cache left enabled; passes read the same starting range in order.",
        "limitations": [
            "Read-only; no test data is written.",
            "Sequential sweep uses aligned Windows unbuffered I/O." if bypass_cache else "Repeated passes may be served partly or entirely from Windows cache.",
            "Do not infer inference tokens/s from these storage reads.",
        ],
    }
