"""Read-only attribution of this process's resident pages to one mapped GGUF."""

from __future__ import annotations

import bisect
import ctypes
import os
import string
import time
from ctypes import wintypes
from pathlib import Path

MEM_COMMIT = 0x1000
MEM_MAPPED = 0x40000


class _MemoryBasicInformation(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wintypes.DWORD),
        ("PartitionId", wintypes.WORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


class _SystemInfo(ctypes.Structure):
    _fields_ = [
        ("wProcessorArchitecture", wintypes.WORD),
        ("wReserved", wintypes.WORD),
        ("dwPageSize", wintypes.DWORD),
        ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p),
        ("dwActiveProcessorMask", ctypes.c_size_t),
        ("dwNumberOfProcessors", wintypes.DWORD),
        ("dwProcessorType", wintypes.DWORD),
        ("dwAllocationGranularity", wintypes.DWORD),
        ("wProcessorLevel", wintypes.WORD),
        ("wProcessorRevision", wintypes.WORD),
    ]


def _device_path_to_drive(path: str, kernel32) -> str | None:
    for letter in string.ascii_uppercase:
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel32.QueryDosDeviceW(f"{letter}:", buffer, len(buffer)):
            continue
        device = buffer.value.rstrip("\\")
        if path.casefold().startswith((device + "\\").casefold()):
            return f"{letter}:" + path[len(device):]
    return None


def _query_working_set_addresses(psapi, process, page_size: int) -> list[int]:
    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    # Leave room for WS growth during the query; grow and retry if necessary.
    size = max(65536, int(counters.WorkingSetSize / page_size * ctypes.sizeof(ctypes.c_size_t) * 1.5))
    for _ in range(8):
        buffer = ctypes.create_string_buffer(size)
        if psapi.QueryWorkingSet(process, buffer, size):
            count = ctypes.c_size_t.from_buffer(buffer).value
            entry_size = ctypes.sizeof(ctypes.c_size_t)
            addresses = []
            for index in range(count):
                raw = ctypes.c_size_t.from_buffer(buffer, entry_size * (index + 1)).value
                address = raw & ~(page_size - 1)
                if address:
                    addresses.append(address)
            return sorted(addresses)
        size *= 2
    raise ctypes.WinError(ctypes.get_last_error())


def query_gguf_residency(model_path: str | Path) -> dict:
    """Count resident working-set pages whose mapped filename resolves to model_path.

    This enumerates the current process working set and groups addresses by
    VirtualQueryEx regions; it does not read model bytes or alter page residency.
    """
    started = time.perf_counter()
    target = os.path.normcase(os.path.abspath(str(Path(model_path).expanduser())))
    if os.name != "nt":
        return {"status": "unsupported", "resident_bytes": None, "method": "Windows PSAPI only"}

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.VirtualQueryEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                       ctypes.POINTER(_MemoryBasicInformation), ctypes.c_size_t]
    kernel32.VirtualQueryEx.restype = ctypes.c_size_t
    kernel32.GetSystemInfo.argtypes = [ctypes.POINTER(_SystemInfo)]
    psapi.GetMappedFileNameW.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                         wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetMappedFileNameW.restype = wintypes.DWORD
    kernel32.QueryDosDeviceW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.QueryDosDeviceW.restype = wintypes.DWORD
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.QueryWorkingSet.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.QueryWorkingSet.restype = wintypes.BOOL

    process = kernel32.GetCurrentProcess()
    system_info = _SystemInfo()
    kernel32.GetSystemInfo(ctypes.byref(system_info))
    page_size = int(system_info.dwPageSize) or 4096
    addresses = _query_working_set_addresses(psapi, process, page_size)
    if not addresses:
        return {"status": "ok", "resident_bytes": 0, "resident_pages": 0,
                "mapped_virtual_bytes": 0, "matched_regions": 0,
                "scan_seconds": round(time.perf_counter() - started, 4),
                "method": "QueryWorkingSet + VirtualQueryEx + GetMappedFileNameW"}

    ranges: list[tuple[int, int]] = []
    cursor = addresses[0]
    end_address = addresses[-1] + page_size
    filename_buffer = ctypes.create_unicode_buffer(32768)
    mapped_virtual_bytes = 0
    matched_regions = 0
    while cursor < end_address:
        mbi = _MemoryBasicInformation()
        queried = kernel32.VirtualQueryEx(process, ctypes.c_void_p(cursor),
                                          ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not queried:
            cursor += page_size
            continue
        base = int(mbi.BaseAddress or cursor)
        region_end = max(cursor + page_size, base + int(mbi.RegionSize))
        if mbi.State == MEM_COMMIT and mbi.Type == MEM_MAPPED:
            filename_buffer.value = ""
            if psapi.GetMappedFileNameW(process, ctypes.c_void_p(base),
                                        filename_buffer, len(filename_buffer)):
                device_path = filename_buffer.value
                resolved = _device_path_to_drive(device_path, kernel32)
                normalized = os.path.normcase(os.path.abspath(resolved)) if resolved else None
                if normalized == target:
                    ranges.append((base, region_end))
                    mapped_virtual_bytes += region_end - base
                    matched_regions += 1
        cursor = region_end

    resident_pages = 0
    for start, end in ranges:
        first = bisect.bisect_left(addresses, start)
        last = bisect.bisect_left(addresses, end, first)
        resident_pages += last - first
    return {
        "status": "ok" if matched_regions else "mapping_not_found",
        "resident_pages": resident_pages if matched_regions else None,
        "resident_bytes": resident_pages * page_size if matched_regions else None,
        "page_size_bytes": page_size,
        "mapped_virtual_bytes": mapped_virtual_bytes if matched_regions else None,
        "matched_regions": matched_regions,
        "working_set_pages_scanned": len(addresses),
        "scan_seconds": round(time.perf_counter() - started, 4),
        "method": "QueryWorkingSet + VirtualQueryEx + GetMappedFileNameW; current-process pages only",
    }
