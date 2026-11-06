"""Read-only Windows memory, commit, pagefile, and volume monitor."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from memory_guard.thresholds import DEFAULT_GUARD_THRESHOLDS, GuardThresholds

GIB = 1024**3
MIB = 1024**2


class MEMORYSTATUSEX(ctypes.Structure):
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


PAGEFILE_PROBE = r"""
$ErrorActionPreference = 'Stop'
@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage) | ConvertTo-Json -Depth 3 -Compress
"""


def evaluate_pressure(
    *,
    available_bytes: int,
    total_bytes: int,
    physical_memory_load_percent: int,
    commit_used_percent: float,
    budget_bytes: int,
    reserve_bytes: int,
    thresholds: GuardThresholds = DEFAULT_GUARD_THRESHOLDS,
) -> dict:
    """Return a recommendation only; this function does not allocate or release memory."""
    if min(available_bytes, total_bytes, budget_bytes, reserve_bytes) < 0 or total_bytes == 0:
        raise ValueError("Memory values must be non-negative and total_bytes must be positive.")
    if not 0 <= commit_used_percent <= 100 or not 0 <= physical_memory_load_percent <= 100:
        raise ValueError("Memory usage percentages must be between 0 and 100.")
    level = thresholds.classify(
        available_bytes=available_bytes,
        physical_memory_load_percent=physical_memory_load_percent,
        commit_used_percent=commit_used_percent,
    )
    safe_headroom = max(0, min(available_bytes, budget_bytes) - reserve_bytes)
    return {
        "pressure": level,
        "recommended_cache_headroom_bytes": safe_headroom,
        "configured_budget_bytes": budget_bytes,
        "reserve_bytes": reserve_bytes,
        "physical_memory_exceeds_configured_budget": total_bytes > budget_bytes * 1.25,
        "recommendation_only": True,
        "thresholds": {
            "warning_available_gib": thresholds.warning_available_gib,
            "critical_available_gib": thresholds.critical_available_gib,
            "warning_commit_percent": thresholds.warning_commit_percent,
            "critical_commit_percent": thresholds.critical_commit_percent,
            "warning_load_percent": thresholds.warning_load_percent,
            "critical_load_percent": thresholds.critical_load_percent,
        },
    }


def _read_pagefile_usage() -> list[dict] | None:
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", PAGEFILE_PROBE],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if parsed is None:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _windows_memory_status() -> MEMORYSTATUSEX:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return status


def collect_memory_snapshot(*, budget_gib: float = 3.0, reserve_mib: int = 512,
                           include_pagefile_query: bool = True) -> dict:
    """Collect one sample without changing process, pagefile, or system settings."""
    if os.name != "nt":
        raise RuntimeError("The memory guard currently supports Windows only.")
    if budget_gib <= 0 or reserve_mib < 0:
        raise ValueError("budget_gib must be positive and reserve_mib cannot be negative.")

    status = _windows_memory_status()
    total_commit = int(status.ullTotalPageFile)
    available_commit = int(status.ullAvailPageFile)
    commit_used = max(0, total_commit - available_commit)
    commit_percent = (commit_used / total_commit * 100) if total_commit else 0.0
    drive = os.environ.get("SystemDrive", "C:") + "\\"
    usage = shutil.disk_usage(drive)
    budget_bytes = int(budget_gib * GIB)
    reserve_bytes = reserve_mib * MIB
    # Periodic inference sampling must stay lightweight: querying CIM through a
    # child PowerShell process can be much slower than reading RAM counters.
    pagefiles = _read_pagefile_usage() if include_pagefile_query else None

    pressure = evaluate_pressure(
        available_bytes=int(status.ullAvailPhys),
        total_bytes=int(status.ullTotalPhys),
        physical_memory_load_percent=int(status.dwMemoryLoad),
        commit_used_percent=commit_percent,
        budget_bytes=budget_bytes,
        reserve_bytes=reserve_bytes,
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "physical_memory": {
            "total_bytes": int(status.ullTotalPhys),
            "available_bytes": int(status.ullAvailPhys),
            "load_percent": int(status.dwMemoryLoad),
        },
        "virtual_commit": {
            "limit_bytes": total_commit,
            "available_bytes": available_commit,
            "committed_bytes": commit_used,
            "used_percent": round(commit_percent, 2),
            "meaning": "System commit usage/limit; this is not equivalent to pagefile bytes currently written.",
        },
        "pagefiles": pagefiles,
        "pagefile_usage_query_available": pagefiles is not None,
        "system_volume": {
            "path": drive,
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
        "guard": pressure,
        "system_settings_changed": False,
    }
