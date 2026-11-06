"""Read-only Windows storage and volume identification."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable


def get_device_info(drive: str = "C:", *, runner: Callable = subprocess.run) -> dict:
    letter = drive.strip().upper().rstrip("\\/")
    if len(letter) != 2 or letter[1] != ":" or not letter[0].isalpha():
        raise ValueError("drive must be a Windows drive letter such as C:")
    if os.name != "nt":
        raise RuntimeError("Device identification is available through Windows CIM/Storage cmdlets only.")
    script = rf"""$ErrorActionPreference='Stop'
$letter='{letter[0]}'
$volume=Get-Volume -DriveLetter $letter | Select-Object DriveLetter,FileSystem,FileSystemLabel,Size,SizeRemaining,Path
$partition=Get-Partition -DriveLetter $letter | Select-Object -First 1
$disk=$null
if ($partition) {{ $disk=Get-Disk -Number $partition.DiskNumber | Select-Object Number,FriendlyName,BusType,Size,PartitionStyle,OperationalStatus }}
$physical=@(Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType,PNPDeviceID)
[pscustomobject]@{{Volume=$volume; Disk=$disk; PhysicalDisks=$physical}} | ConvertTo-Json -Depth 5 -Compress
"""
    completed = runner(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                       check=False, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"Windows storage query failed: {completed.stderr.strip()}")
    try:
        raw = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Windows storage query returned invalid JSON.") from exc
    volume = raw.get("Volume") or {}
    disk = raw.get("Disk") or {}
    physical_disks = raw.get("PhysicalDisks") or []
    if isinstance(physical_disks, dict):
        physical_disks = [physical_disks]
    bus = str(disk.get("BusType") or "Unknown")
    return {
        "drive": f"{letter}\\", "volume_label": volume.get("FileSystemLabel"),
        "filesystem": volume.get("FileSystem"), "capacity_bytes": volume.get("Size"),
        "free_bytes": volume.get("SizeRemaining"), "disk_number": disk.get("Number"),
        "device_model": disk.get("FriendlyName"), "bus_type_reported": bus,
        "nvme_confirmed_by_windows": bus.casefold() == "nvme",
        "nvme_status_note": ("Windows reports NVMe bus type." if bus.casefold() == "nvme" else
                             "Windows reports RAID; the underlying NVMe device cannot be inferred from this bus label." if bus.casefold() == "raid" else
                             "Bus type is reported as-is; this does not establish NVMe unless Windows reports NVMe."),
        "disk_capacity_bytes": disk.get("Size"), "partition_style": disk.get("PartitionStyle"),
        "operational_status": disk.get("OperationalStatus"), "physical_disks": physical_disks,
        "source": "Windows Get-Volume/Get-Partition/Get-Disk and Win32_DiskDrive; read-only query.",
    }
