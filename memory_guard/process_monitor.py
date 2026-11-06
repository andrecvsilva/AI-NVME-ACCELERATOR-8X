"""Read-only Ollama process inventory and lightweight Windows memory metrics."""

from __future__ import annotations

import ctypes
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

MIB = 1024**2
GIB = 1024**3
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TH32CS_SNAPPROCESS = 0x00000002
_PROCESS_VM_COUNTERS = 3
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    name: str
    thread_count: int


@dataclass(frozen=True)
class OllamaProcessMetrics:
    pid: int
    name: str
    working_set_bytes: int | None = None
    private_working_set_bytes: int | None = None
    private_bytes: int | None = None
    virtual_bytes: int | None = None
    thread_count: int | None = None
    uptime_seconds: float | None = None
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in (
            "working_set_bytes", "private_working_set_bytes", "private_bytes", "virtual_bytes"
        ):
            value = result[key]
            result[key.replace("_bytes", "_mib")] = round(value / MIB, 2) if value is not None else None
        return result


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_ulong),
        ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class _ProcessMemoryCountersEx2(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
        ("PrivateWorkingSetSize", ctypes.c_size_t),
        ("SharedCommitUsage", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = _ProcessMemoryCountersEx2._fields_[:-2]


class _VmCounters(ctypes.Structure):
    """VM_COUNTERS layout used by NtQueryInformationProcess class 3."""

    _fields_ = [
        ("PeakVirtualSize", ctypes.c_size_t),
        ("VirtualSize", ctypes.c_size_t),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivatePageCount", ctypes.c_size_t),
    ]


def _matches_ollama(name: str) -> bool:
    return "ollama" in Path(name).name.casefold()


def enumerate_ollama_processes() -> list[ProcessIdentity]:
    """Take a Toolhelp snapshot; command lines are deliberately not collected."""
    if os.name != "nt":
        raise RuntimeError("Ollama process monitoring currently supports Windows only.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    handle = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    found: list[ProcessIdentity] = []
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            name = entry.szExeFile
            if _matches_ollama(name):
                found.append(ProcessIdentity(int(entry.th32ProcessID), name, int(entry.cntThreads)))
            ok = kernel32.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(handle)
    return found


def _filetime_value(filetime) -> int:
    return (int(filetime.dwHighDateTime) << 32) | int(filetime.dwLowDateTime)


def _read_virtual_size(process_handle) -> int | None:
    """Use the native VM counter for VirtualSize; never substitute private commit."""
    try:
        ntdll = ctypes.WinDLL("ntdll")
        query = ntdll.NtQueryInformationProcess
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
        query.restype = ctypes.c_long
        counters = _VmCounters()
        status = query(process_handle, _PROCESS_VM_COUNTERS, ctypes.byref(counters), ctypes.sizeof(counters), None)
        return int(counters.VirtualSize) if status >= 0 else None
    except (AttributeError, OSError):
        return None


def read_process_metrics(identity: ProcessIdentity) -> OllamaProcessMetrics:
    """Read one process and always close its handle; races/permissions are data."""
    if os.name != "nt":
        raise RuntimeError("Ollama process monitoring currently supports Windows only.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, identity.pid)
    if not handle:
        code = ctypes.get_last_error()
        state = "exited_during_collection" if code == _ERROR_INVALID_PARAMETER else "permission_denied" if code == _ERROR_ACCESS_DENIED else "unavailable"
        return OllamaProcessMetrics(identity.pid, identity.name, thread_count=identity.thread_count, status=state, error=f"WinError {code}")

    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_memory = getattr(psapi, "GetProcessMemoryInfo", None)
        if get_memory is None:
            get_memory = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        get_memory.restype = ctypes.c_int
        extended = _ProcessMemoryCountersEx2()
        extended.cb = ctypes.sizeof(extended)
        private_ws: int | None = None
        if get_memory(handle, ctypes.byref(extended), extended.cb):
            working_set = int(extended.WorkingSetSize)
            private_bytes = int(extended.PrivateUsage)
            private_ws = int(extended.PrivateWorkingSetSize)
        else:
            base = _ProcessMemoryCountersEx()
            base.cb = ctypes.sizeof(base)
            if not get_memory(handle, ctypes.byref(base), base.cb):
                code = ctypes.get_last_error()
                state = "exited_during_collection" if code == _ERROR_INVALID_PARAMETER else "permission_denied" if code == _ERROR_ACCESS_DENIED else "unavailable"
                return OllamaProcessMetrics(identity.pid, identity.name, thread_count=identity.thread_count, status=state, error=f"WinError {code}")
            working_set = int(base.WorkingSetSize)
            private_bytes = int(base.PrivateUsage)

        class _FileTime(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

        creation, exit_time, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
        kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime)]
        kernel32.GetProcessTimes.restype = ctypes.c_int
        uptime = None
        if kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
            elapsed_100ns = int(datetime.now(timezone.utc).timestamp() * 10_000_000) + 116444736000000000 - _filetime_value(creation)
            uptime = max(0.0, elapsed_100ns / 10_000_000)
        return OllamaProcessMetrics(
            pid=identity.pid,
            name=identity.name,
            working_set_bytes=working_set,
            private_working_set_bytes=private_ws,
            private_bytes=private_bytes,
            virtual_bytes=_read_virtual_size(handle),
            thread_count=identity.thread_count,
            uptime_seconds=round(uptime, 1) if uptime is not None else None,
        )
    finally:
        kernel32.CloseHandle(handle)


def collect_ollama_processes(
    *,
    enumerator: Callable[[], Iterable[ProcessIdentity]] = enumerate_ollama_processes,
    reader: Callable[[ProcessIdentity], OllamaProcessMetrics] = read_process_metrics,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        identities = list(enumerator())
    except (OSError, RuntimeError, PermissionError) as exc:
        return {"timestamp_utc": timestamp, "status": "unavailable", "error": str(exc), "processes": []}
    processes = []
    for identity in identities:
        try:
            processes.append(reader(identity).to_dict())
        except PermissionError as exc:
            processes.append(OllamaProcessMetrics(identity.pid, identity.name, thread_count=identity.thread_count, status="permission_denied", error=str(exc)).to_dict())
        except (ProcessLookupError, OSError) as exc:
            state = "exited_during_collection" if isinstance(exc, ProcessLookupError) else "unavailable"
            processes.append(OllamaProcessMetrics(identity.pid, identity.name, thread_count=identity.thread_count, status=state, error=str(exc)).to_dict())
    return {"timestamp_utc": timestamp, "status": "active", "error": None, "processes": processes}


def analyze_ollama_state(
    *,
    online: bool,
    loaded_model_count: int,
    process_report: dict,
    guard_state: str,
    residual_suspect_mib: int = 1536,
) -> dict:
    processes = process_report.get("processes", [])
    valid = [p for p in processes if p.get("status") == "ok"]
    unavailable = [p for p in processes if p.get("status") in {"permission_denied", "unavailable"}]
    exited = [p for p in processes if p.get("status") == "exited_during_collection"]
    total_private = sum(int(p["private_bytes"]) for p in valid if p.get("private_bytes") is not None)

    if not online:
        classification = "OFFLINE"
    elif loaded_model_count > 0:
        classification = "ONLINE_MODEL_LOADED"
    elif process_report.get("status") == "unavailable" or unavailable:
        classification = "PROCESS_INFO_UNAVAILABLE"
    elif exited:
        classification = "PROCESS_EXITED_DURING_COLLECTION"
    elif not processes:
        classification = "ONLINE_NO_MODEL_NO_PROCESS"
    elif total_private >= max(0, residual_suspect_mib) * MIB:
        classification = "MEMORY_RESIDUAL_SUSPECTED"
    else:
        classification = "ONLINE_NO_MODEL_LOW_MEMORY"
    return {
        "classification": classification,
        "memory_guard_state": guard_state,
        "ollama_online": online,
        "loaded_model_count": loaded_model_count,
        "process_count": len(processes),
        "private_bytes_total_for_valid_processes": total_private,
        "private_mib_total_for_valid_processes": round(total_private / MIB, 2),
        "residual_suspected_is_hypothesis": classification == "MEMORY_RESIDUAL_SUSPECTED",
        "process_monitor_status": process_report.get("status", "unavailable"),
    }


def record_process_snapshot(report: dict, logs_dir: str | Path = "logs") -> None:
    """Append bounded-memory JSON-array and CSV records; preserve existing history."""
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "process_monitor.json"
    csv_path = directory / "process_monitor.csv"
    processes = report.get("processes", [])
    rows = processes or [None]
    fields = [
        "timestamp_utc", "memory_guard_state", "ram_available_bytes", "ram_load_percent",
        "ollama_online", "loaded_model_count", "classification", "pid", "name",
        "working_set_bytes", "private_working_set_bytes", "private_bytes", "virtual_bytes",
        "thread_count", "uptime_seconds", "process_status",
    ]
    with _LOG_LOCK:
        if not json_path.exists():
            with json_path.open("w", encoding="utf-8") as handle:
                handle.write("[\n" + json.dumps(report, ensure_ascii=False) + "\n]\n")
        else:
            with json_path.open("r+b") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if end == 0:
                    raise RuntimeError(f"Refusing to append to empty existing process log: {json_path}")
                cursor = end - 1
                while cursor >= 0:
                    handle.seek(cursor)
                    char = handle.read(1)
                    if char not in b" \t\r\n":
                        break
                    cursor -= 1
                if cursor < 0 or char != b"]":
                    raise RuntimeError(f"Refusing to append to malformed process log: {json_path}")
                handle.seek(0)
                if handle.read(1) != b"[":
                    raise RuntimeError(f"Refusing to append to non-array process log: {json_path}")
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
                handle.write(separator + json.dumps(report, ensure_ascii=False).encode("utf-8") + b"\n]\n")

        new_csv = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if new_csv:
                writer.writeheader()
            for process in rows:
                row = {
                    "timestamp_utc": report.get("timestamp_utc", ""),
                    "memory_guard_state": report.get("memory_guard_state", ""),
                    "ram_available_bytes": report.get("ram_available_bytes", ""),
                    "ram_load_percent": report.get("ram_load_percent", ""),
                    "ollama_online": report.get("ollama_online", ""),
                    "loaded_model_count": report.get("loaded_model_count", ""),
                    "classification": report.get("classification", ""),
                }
                if process:
                    row.update({key: process.get(key, "") for key in fields if key in process})
                    row["process_status"] = process.get("status", "")
                writer.writerow(row)
