"""Safe settings planner; optimization requires later measured comparisons."""

from __future__ import annotations

from .config import AcceleratorConfig
from .scheduler import AdaptiveRamNvmeScheduler


def optimization_plan(config: AcceleratorConfig, snapshot: dict) -> dict:
    decision = AdaptiveRamNvmeScheduler(config).decide(snapshot)
    return {
        "admission": decision.to_dict(),
        "runtime_options": {
            "n_ctx": decision.context, "n_batch": decision.batch,
            "n_ubatch": decision.batch, "n_threads": decision.threads,
            "n_threads_batch": decision.threads, "n_gpu_layers": 0,
            "use_mmap": decision.mmap, "use_mlock": decision.mlock,
        },
        "optimization_status": "planned_not_benchmarked",
        "claim": "No speedup is claimed; settings are conservative limits pending measured validation.",
    }
