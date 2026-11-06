"""250 ms load profiler with bounded summaries and GGUF page-residency samples."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from engine.memory_guard import collect_memory_snapshot
from engine.gguf_residency import query_gguf_residency


def _process_memory_bytes() -> dict:
    """Return process working set/private bytes and cumulative I/O counters."""
    if os.name != "nt":
        return {"working_set_bytes": None, "private_bytes": None}
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        memory = {"working_set_bytes": None, "private_bytes": None, "page_fault_count": None}
    else:
        memory = {"working_set_bytes": int(counters.WorkingSetSize),
                  "private_bytes": int(counters.PrivateUsage),
                  "page_fault_count": int(counters.PageFaultCount)}
    io = IO_COUNTERS()
    kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IO_COUNTERS)]
    kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    if kernel32.GetProcessIoCounters(handle, ctypes.byref(io)):
        memory.update({"read_transfer_bytes": int(io.ReadTransferCount),
                       "read_operation_count": int(io.ReadOperationCount)})
    else:
        memory.update({"read_transfer_bytes": None, "read_operation_count": None})
    return memory


class ModelMemoryObserver:
    """Persist each sample as JSONL; retain only extrema and transition times in RAM."""

    def __init__(self, *, interval_seconds: float = 0.25,
                 residency_interval_seconds: float = 1.0,
                 memory_probe: Callable[..., dict] = collect_memory_snapshot,
                 process_probe: Callable[[], dict] = _process_memory_bytes,
                 residency_probe: Callable[[str | Path], dict] = query_gguf_residency):
        if not 0.25 <= interval_seconds <= 10:
            raise ValueError("interval_seconds must be between 0.25 and 10")
        if not 0.5 <= residency_interval_seconds <= 30:
            raise ValueError("residency_interval_seconds must be between 0.5 and 30")
        self.interval_seconds = interval_seconds
        self.residency_interval_seconds = residency_interval_seconds
        self.memory_probe = memory_probe
        self.process_probe = process_probe
        self.residency_probe = residency_probe
        self._stop = threading.Event()
        self._residency_stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._residency_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._summary: dict = {}
        self._budget_gib = 3.0
        self._reserve_mib = 512
        self._profile_path: Path | None = None
        self._model_path: Path | None = None
        self._run_id = ""
        self._started_at = 0.0
        self._latest_residency: dict | None = None
        self._last_pressure: str | None = None
        self._sample_index = 0

    def _append(self, event: dict) -> None:
        if self._profile_path is None:
            return
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self._profile_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _sample(self, *, track_interval: bool = True) -> None:
        sampled_at = time.perf_counter()
        try:
            mem = self.memory_probe(budget_gib=self._budget_gib,
                                    reserve_mib=self._reserve_mib,
                                    include_pagefile_query=False)
            physical = mem.get("physical_memory", {})
            if not physical and "physical_memory" not in mem:
                physical = mem
            commit = mem.get("virtual_commit", {})
            try:
                proc = self.process_probe()
            except Exception as exc:
                proc = {}
                with self._lock:
                    self._summary["process_probe_errors"] += 1
                    self._summary["last_process_probe_error"] = str(exc)

            elapsed = sampled_at - self._started_at
            with self._lock:
                residency = dict(self._latest_residency) if self._latest_residency else None

            pressure = (mem.get("guard") or {}).get("pressure", "unknown")
            transition = None
            if self._last_pressure is not None and pressure != self._last_pressure:
                transition = {"from": self._last_pressure, "to": pressure,
                              "elapsed_seconds": round(elapsed, 4),
                              "timestamp_utc": mem.get("timestamp_utc")}
            self._last_pressure = pressure
            self._sample_index += 1
            sample = {
                "event_type": "model_load_sample",
                "run_id": self._run_id,
                "sample_index": self._sample_index,
                "timestamp_utc": mem.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 4),
                "sampling_interval_target_ms": round(self.interval_seconds * 1000),
                "memory": {
                    "total_bytes": physical.get("total_bytes"),
                    "available_bytes": physical.get("available_bytes"),
                    "load_percent": physical.get("load_percent"),
                    "commit_limit_bytes": commit.get("limit_bytes"),
                    "committed_bytes": commit.get("committed_bytes"),
                    "commit_used_percent": commit.get("used_percent"),
                    "guard_state": pressure,
                    "guard_thresholds": (mem.get("guard") or {}).get("thresholds"),
                },
                "process": proc,
                "gguf_residency": residency,
                "guard_transition": transition,
            }
            with self._lock:
                self._summary["sample_count"] += 1
                previous_sample = self._summary.get("last_sample_perf")
                if previous_sample is not None and track_interval:
                    interval_ms = (sampled_at - previous_sample) * 1000
                    minimum = self._summary.get("sample_interval_ms_min")
                    maximum = self._summary.get("sample_interval_ms_max")
                    self._summary["sample_interval_ms_min"] = interval_ms if minimum is None else min(minimum, interval_ms)
                    self._summary["sample_interval_ms_max"] = interval_ms if maximum is None else max(maximum, interval_ms)
                    self._summary["sample_interval_ms_sum"] = self._summary.get("sample_interval_ms_sum", 0.0) + interval_ms
                    self._summary["sample_interval_count"] = self._summary.get("sample_interval_count", 0) + 1
                self._summary["last_sample_perf"] = sampled_at
                for key, value in {
                    "available_bytes": physical.get("available_bytes"),
                    "load_percent": physical.get("load_percent"),
                    "working_set_bytes": proc.get("working_set_bytes"),
                    "private_bytes": proc.get("private_bytes"),
                }.items():
                    if value is None:
                        continue
                    summary_key = "minimum_" + key if key == "available_bytes" else "peak_" + key
                    previous = self._summary.get(summary_key)
                    self._summary[summary_key] = value if previous is None else (
                        min(previous, value) if key == "available_bytes" else max(previous, value))
                for key in ("page_fault_count", "read_transfer_bytes", "read_operation_count"):
                    value = proc.get(key)
                    if value is not None:
                        self._summary.setdefault("start_" + key, value)
                        self._summary["latest_" + key] = value
                ranks = {"unknown": -1, "normal": 0, "warning": 1, "critical": 2}
                if ranks.get(pressure, -1) > ranks.get(self._summary.get("peak_pressure", "unknown"), -1):
                    self._summary["peak_pressure"] = pressure
                self._summary["last_pressure"] = pressure
                if transition:
                    self._summary.setdefault("guard_transitions", []).append(transition)
                if residency and residency.get("resident_bytes") is not None:
                    value = residency["resident_bytes"]
                    prev = self._summary.get("peak_gguf_resident_bytes")
                    self._summary["peak_gguf_resident_bytes"] = value if prev is None else max(prev, value)
            self._append(sample)
        except Exception as exc:
            with self._lock:
                self._summary["sample_errors"] += 1
                self._summary["last_sample_error"] = str(exc)

    def start(self, *, budget_gib: float, reserve_mib: int,
              model_path: str | Path | None = None,
              profile_path: str | Path | None = None) -> "ModelMemoryObserver":
        self._budget_gib, self._reserve_mib = budget_gib, reserve_mib
        self._profile_path = Path(profile_path) if profile_path else None
        self._model_path = Path(model_path).resolve() if model_path else None
        self._run_id = uuid.uuid4().hex
        self._started_at = time.perf_counter()
        self._latest_residency = None
        self._last_pressure = None
        self._sample_index = 0
        self._summary = {"sample_count": 0, "sample_errors": 0, "process_probe_errors": 0,
                         "minimum_available_bytes": None, "peak_load_percent": None,
                         "peak_working_set_bytes": None, "peak_private_bytes": None,
                         "peak_gguf_resident_bytes": None, "peak_pressure": "unknown",
                         "last_pressure": "unknown", "guard_transitions": []}
        self._stop.clear()
        self._residency_stop.clear()
        self._append({"event_type": "model_load_profile_started", "run_id": self._run_id,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "model_path": str(self._model_path) if self._model_path else None,
                      "memory_interval_ms": round(self.interval_seconds * 1000),
                      "residency_interval_ms": round(self.residency_interval_seconds * 1000)})
        self._sample(track_interval=False)
        self._thread = threading.Thread(target=self._run, name="model-memory-observer", daemon=True)
        self._thread.start()
        if self._model_path is not None:
            self._residency_thread = threading.Thread(
                target=self._run_residency, name="gguf-residency-observer", daemon=True
            )
            self._residency_thread.start()
        return self

    def _run(self) -> None:
        deadline = time.perf_counter() + self.interval_seconds
        while not self._stop.is_set():
            if self._stop.wait(max(0, deadline - time.perf_counter())):
                break
            self._sample()
            deadline += self.interval_seconds
            if deadline < time.perf_counter():
                deadline = time.perf_counter() + self.interval_seconds

    def _run_residency(self) -> None:
        while not self._residency_stop.is_set():
            try:
                result = self.residency_probe(self._model_path)
            except Exception as exc:
                result = {"status": "error", "error": str(exc), "resident_bytes": None}
            result = {**result, "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                      "observer_elapsed_seconds": round(time.perf_counter() - self._started_at, 4)}
            with self._lock:
                self._latest_residency = result
                if result.get("resident_bytes") is not None:
                    value = result["resident_bytes"]
                    previous = self._summary.get("peak_gguf_resident_bytes")
                    self._summary["peak_gguf_resident_bytes"] = value if previous is None else max(previous, value)
            if self._residency_stop.wait(self.residency_interval_seconds):
                break

    def summary(self) -> dict:
        with self._lock:
            return {key: value for key, value in self._summary.items() if key != "last_sample_perf"}

    def stop(self, *, status: str = "completed", error: str | None = None) -> dict:
        self._stop.set()
        self._residency_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5)
            self._thread = None
        if self._residency_thread is not None:
            self._residency_thread.join(timeout=self.residency_interval_seconds + 10)
            self._residency_thread = None
        self._sample(track_interval=False)
        with self._lock:
            for key in ("page_fault_count", "read_transfer_bytes", "read_operation_count"):
                start_value = self._summary.get("start_" + key)
                latest_value = self._summary.get("latest_" + key)
                self._summary["delta_" + key] = (
                    max(0, latest_value - start_value)
                    if start_value is not None and latest_value is not None else None
                )
            result = {key: value for key, value in self._summary.items() if key != "last_sample_perf"}
            count = result.get("sample_interval_count", 0)
            result["sample_interval_ms_mean"] = (
                result.get("sample_interval_ms_sum", 0.0) / count if count else None
            )
        self._append({"event_type": "model_load_profile_finished", "run_id": self._run_id,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": status,
                      "error": error, "summary": result})
        return result
