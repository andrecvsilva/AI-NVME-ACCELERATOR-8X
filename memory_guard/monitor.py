"""Low overhead Windows RAM and commit monitor using GlobalMemoryStatusEx."""

from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

GIB = 1024**3


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp_utc: str
    total_bytes: int
    available_bytes: int
    used_bytes: int
    load_percent: int
    commit_limit_bytes: int
    commit_available_bytes: int
    committed_bytes: int
    commit_used_percent: float

    def to_dict(self) -> dict:
        return asdict(self)


def read_memory_snapshot() -> MemorySnapshot:
    if os.name != "nt":
        raise RuntimeError("Memory Guard currently supports Windows only.")
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    total_commit = int(status.ullTotalPageFile)
    available_commit = int(status.ullAvailPageFile)
    committed = max(0, total_commit - available_commit)
    return MemorySnapshot(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        total_bytes=int(status.ullTotalPhys),
        available_bytes=int(status.ullAvailPhys),
        used_bytes=max(0, int(status.ullTotalPhys) - int(status.ullAvailPhys)),
        load_percent=int(status.dwMemoryLoad),
        commit_limit_bytes=total_commit,
        commit_available_bytes=available_commit,
        committed_bytes=committed,
        commit_used_percent=round(committed / total_commit * 100, 2) if total_commit else 0.0,
    )
