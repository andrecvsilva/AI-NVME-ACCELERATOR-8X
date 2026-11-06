"""In-memory cache behavior benchmark; it does not claim SSD/inference speedups."""

from __future__ import annotations

import argparse
import json

from engine.cache_manager import AdaptiveBlockCache, MIB


def run_hot_block_trace(*, block_bytes: int = 4096, repetitions: int = 100, capacity_mib: int = 1) -> dict:
    if block_bytes < 1 or not 1 <= repetitions <= 10000 or not 0 <= capacity_mib <= 16:
        raise ValueError("block_bytes must be positive, repetitions 1..10000, capacity 0..16 MiB")
    cache = AdaptiveBlockCache(max_capacity_mib=capacity_mib)
    cache.adjust_capacity(
        available_bytes=2 * MIB * 1024,
        configured_budget_bytes=3 * MIB * 1024,
        reserve_bytes=512 * MIB,
        pressure="normal",
    )
    source = {f"block-{index}": bytes([index % 251]) * block_bytes for index in range(8)}
    uncached_loads = 0
    cached_loads = 0
    for index in range(repetitions):
        key = f"block-{index % 8}"
        uncached_loads += 1
        def load() -> bytes:
            nonlocal cached_loads
            cached_loads += 1
            return source[key]
        cache.get_or_load(key, load)
    return {
        "workload": "eight repeated in-memory blocks; synthetic behavior test only",
        "repetitions": repetitions,
        "block_bytes": block_bytes,
        "uncached_load_calls": uncached_loads,
        "cache_loader_calls": cached_loads,
        "cache": cache.stats(),
        "limitations": ["No disk is read by this benchmark.", "No inference or token rate is measured."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--capacity-mib", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_hot_block_trace(block_bytes=args.block_bytes, repetitions=args.repetitions, capacity_mib=args.capacity_mib), indent=2))
