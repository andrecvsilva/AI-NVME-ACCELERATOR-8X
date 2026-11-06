"""Low-overhead physical-disk I/O sampling through Windows PDH counters.

These counters describe the whole physical disk instance. They cannot attribute
traffic to one process or file; use ETW/WPA for GGUF-specific attribution.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from typing import Callable

MIB = 1024 * 1024
_PDH_FMT_DOUBLE = 0x00000200


class _CounterValueUnion(ctypes.Union):
    _fields_ = [
        ("long_value", wintypes.LONG),
        ("double_value", ctypes.c_double),
        ("large_value", ctypes.c_longlong),
        ("ansi_string", ctypes.c_char_p),
        ("wide_string", wintypes.LPWSTR),
    ]


class _FormattedCounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", _CounterValueUnion)]


class _PdhPhysicalDiskSource:
    """Read English PDH counters; PdhAddEnglishCounter works on localized Windows."""

    def __init__(self, instance: str):
        if os.name != "nt":
            raise OSError("Physical disk PDH counters are available only on Windows.")
        self._pdh = ctypes.WinDLL("pdh.dll", use_last_error=True)
        self._query = ctypes.c_void_p()
        self._counters: dict[str, ctypes.c_void_p] = {}
        self._pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t,
                                            ctypes.POINTER(ctypes.c_void_p)]
        self._pdh.PdhOpenQueryW.restype = wintypes.DWORD
        self._pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR,
                                                    ctypes.c_size_t,
                                                    ctypes.POINTER(ctypes.c_void_p)]
        self._pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
        self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self._pdh.PdhCollectQueryData.restype = wintypes.DWORD
        self._pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                                          ctypes.POINTER(wintypes.DWORD),
                                                          ctypes.POINTER(_FormattedCounterValue)]
        self._pdh.PdhGetFormattedCounterValue.restype = wintypes.DWORD
        self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._pdh.PdhCloseQuery.restype = wintypes.DWORD

        self._check(self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)), "open PDH query")
        paths = {
            "read_bytes_per_second": "Disk Read Bytes/sec",
            "read_operations_per_second": "Disk Reads/sec",
            "average_read_latency_seconds": "Avg. Disk sec/Read",
        }
        try:
            for key, counter_name in paths.items():
                counter = ctypes.c_void_p()
                path = rf"\PhysicalDisk({instance})\{counter_name}"
                self._check(self._pdh.PdhAddEnglishCounterW(
                    self._query, path, 0, ctypes.byref(counter)), f"add PDH counter {counter_name}")
                self._counters[key] = counter
        except BaseException:
            self.close()
            raise

    def prime(self) -> None:
        """Establish the first PDH sample; rate counters become valid on the next poll."""
        self._check(self._pdh.PdhCollectQueryData(self._query), "prime counters")

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise OSError(f"PDH failed to {operation} (status 0x{status:08X}).")

    def collect(self) -> dict[str, float]:
        self._check(self._pdh.PdhCollectQueryData(self._query), "collect counters")
        values: dict[str, float] = {}
        for key, counter in self._counters.items():
            formatted = _FormattedCounterValue()
            self._check(self._pdh.PdhGetFormattedCounterValue(
                counter, _PDH_FMT_DOUBLE, None, ctypes.byref(formatted)), f"read {key}")
            if formatted.status != 0:
                raise OSError(f"PDH counter {key} is not ready (status 0x{formatted.status:08X}).")
            values[key] = max(0.0, float(formatted.value.double_value))
        return values

    def close(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()


class PhysicalDiskIoObserver:
    """Sample whole-disk read rates without spawning a helper process."""

    def __init__(self, *, instance: str = "0 C:", interval_seconds: float = 0.25,
                 source_factory: Callable[[str], object] = _PdhPhysicalDiskSource):
        if not 0.1 <= interval_seconds <= 10:
            raise ValueError("interval_seconds must be between 0.1 and 10 seconds")
        if not instance.strip():
            raise ValueError("PDH physical disk instance cannot be empty")
        self.instance = instance
        self.interval_seconds = interval_seconds
        self._source_factory = source_factory
        self._source = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._last_at: float | None = None
        self._last_values: dict[str, float] | None = None
        self._sample_count = 0
        self._errors = 0
        self._read_bytes_estimate = 0.0
        self._read_ops_estimate = 0.0
        self._read_seconds_estimate = 0.0
        self._throughput_max = 0.0
        self._elapsed = 0.0
        self._start_error: str | None = None

    def start(self) -> "PhysicalDiskIoObserver":
        if self._thread is not None:
            raise RuntimeError("Physical disk observer is already started")
        try:
            self._source = self._source_factory(self.instance)
            prime = getattr(self._source, "prime", None)
            if callable(prime):
                prime()
                baseline = None
            else:
                baseline = self._source.collect()
        except Exception as exc:
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._close_source()
            return self
        now = time.perf_counter()
        self._started_at = self._last_at = now
        self._last_values = baseline
        self._thread = threading.Thread(target=self._run, name="physical-disk-io-observer", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                values = self._source.collect()
                self._record(values, time.perf_counter())
            except Exception:
                with self._lock:
                    self._errors += 1

    def _record(self, values: dict[str, float], sampled_at: float) -> None:
        with self._lock:
            if self._last_at is None:
                self._last_at, self._last_values = sampled_at, values
                return
            elapsed = max(0.0, sampled_at - self._last_at)
            if elapsed <= 0:
                return
            rate_bytes = max(0.0, float(values["read_bytes_per_second"]))
            rate_ops = max(0.0, float(values["read_operations_per_second"]))
            latency_s = max(0.0, float(values["average_read_latency_seconds"]))
            self._read_bytes_estimate += rate_bytes * elapsed
            self._read_ops_estimate += rate_ops * elapsed
            self._read_seconds_estimate += latency_s * rate_ops * elapsed
            self._throughput_max = max(self._throughput_max, rate_bytes / MIB)
            self._elapsed += elapsed
            self._sample_count += 1
            self._last_at, self._last_values = sampled_at, values

    def _close_source(self) -> None:
        if self._source is not None:
            close = getattr(self._source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._source = None

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
            if self._thread.is_alive():
                with self._lock:
                    self._errors += 1
            else:
                self._thread = None
        self._close_source()
        with self._lock:
            elapsed = self._elapsed
            ops = self._read_ops_estimate
            return {
                "available": self._start_error is None,
                "source": "Windows PDH PhysicalDisk counters",
                "disk_instance": self.instance,
                "sample_interval_seconds": self.interval_seconds,
                "sample_count": self._sample_count,
                "sample_errors": self._errors,
                "measurement_seconds": round(elapsed, 3),
                "estimated_read_bytes": int(self._read_bytes_estimate),
                "estimated_read_mib": round(self._read_bytes_estimate / MIB, 3),
                "estimated_read_ops": int(self._read_ops_estimate),
                "estimated_iops": round(ops / elapsed, 2) if elapsed > 0 else None,
                "average_read_mib_s": round(self._read_bytes_estimate / MIB / elapsed, 3) if elapsed > 0 else None,
                "peak_sampled_read_mib_s": round(self._throughput_max, 3),
                "weighted_average_read_latency_ms": (
                    round(self._read_seconds_estimate / ops * 1000, 3) if ops > 0 else None
                ),
                "scope": "All read I/O on this physical disk during the window; background activity is included.",
                "per_file_attribution_available": False,
                "per_operation_latency_percentiles_available": False,
                "limitation": self._start_error or (
                    "Rates are integrated from PDH samples; these do not identify reads for the GGUF specifically."
                ),
            }

