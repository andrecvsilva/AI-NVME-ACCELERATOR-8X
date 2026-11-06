"""Measured throughput, IOPS, and nearest-rank latency summaries."""

from __future__ import annotations

import math
import statistics

MIB = 1024 * 1024


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(*, bytes_read: int, elapsed_seconds: float, operations: int = 0,
              latencies_ms: list[float] | None = None) -> dict:
    samples = latencies_ms or []
    seconds = max(0.0, elapsed_seconds)
    return {
        "bytes_read": int(bytes_read),
        "operations": int(operations),
        "elapsed_seconds": round(seconds, 6),
        "throughput_mb_s": round(bytes_read / 1_000_000 / seconds, 3) if seconds > 0 else None,
        "throughput_mib_s": round(bytes_read / MIB / seconds, 3) if seconds > 0 else None,
        "iops": round(operations / seconds, 3) if seconds > 0 else None,
        "latency_mean_ms": round(statistics.fmean(samples), 6) if samples else None,
        "latency_p50_ms": round(percentile(samples, 0.50), 6) if samples else None,
        "latency_p95_ms": round(percentile(samples, 0.95), 6) if samples else None,
        "latency_p99_ms": round(percentile(samples, 0.99), 6) if samples else None,
    }
