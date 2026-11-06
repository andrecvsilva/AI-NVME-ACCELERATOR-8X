"""High-resolution timing helpers based on the monotonic performance clock."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def measure_ms(operation: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000.0
