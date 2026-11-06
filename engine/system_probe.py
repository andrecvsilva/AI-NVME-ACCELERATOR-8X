"""Windows hardware discovery using read-only CIM queries."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone


POWERSHELL_PROBE = r"""$ErrorActionPreference = 'Stop'
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory
$disks = @(Get-CimInstance Win32_DiskDrive | Select-Object Model,InterfaceType,Size,Status)
$volumes = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace)
$pagefiles = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
$gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion)
[pscustomobject]@{ CPU=$cpu; OS=$os; Disks=$disks; Volumes=$volumes; PageFiles=$pagefiles; GPUs=$gpus } | ConvertTo-Json -Depth 5 -Compress
"""


def _to_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def probe_windows() -> dict:
    """Collect current system details. This does not write to disks or change settings."""
    if os.name != "nt":
        raise RuntimeError("Stage 1 hardware probe currently supports Windows only.")
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_PROBE],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"PowerShell CIM probe failed: {completed.stderr.strip()}")
    try:
        raw = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("PowerShell returned invalid JSON during hardware probe.") from exc

    os_info = raw.get("OS") or {}
    total_kib = int(os_info.get("TotalVisibleMemorySize") or 0)
    free_kib = int(os_info.get("FreePhysicalMemory") or 0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "cpu": raw.get("CPU"),
        "os": {
            "caption": os_info.get("Caption"),
            "version": os_info.get("Version"),
            "architecture": os_info.get("OSArchitecture"),
        },
        "memory": {
            "visible_total_bytes": total_kib * 1024,
            "visible_available_bytes": free_kib * 1024,
            "visible_total_gib": round(total_kib / 1024 / 1024, 3),
            "visible_available_gib": round(free_kib / 1024 / 1024, 3),
            "note": "Windows-visible memory; compare with the user-configured model memory budget.",
        },
        "disks": _to_list(raw.get("Disks")),
        "volumes": _to_list(raw.get("Volumes")),
        "pagefiles": _to_list(raw.get("PageFiles")),
        "gpus": _to_list(raw.get("GPUs")),
        "temperature": {"available": False, "reason": "No portable Windows CIM temperature source was queried in Stage 1."},
    }
