"""Integration facade for status/planning and explicit validation runs."""

from __future__ import annotations

from pathlib import Path

from engine.memory_guard import collect_memory_snapshot

from .config import AcceleratorConfig
from .model_streaming import plan_model_streaming
from .optimizer import optimization_plan
from .paging import paging_report
from .prefetch import prefetch_decision
from .validation import run_single_comparison


class AcceleratorRuntime:
    def __init__(self, config: AcceleratorConfig, *, project_root: str | Path):
        config.validate()
        self.config = config
        self.project_root = Path(project_root).resolve()

    def inspect(self, *, snapshot: dict | None = None) -> dict:
        snapshot = snapshot or collect_memory_snapshot(
            budget_gib=self.config.memory_budget_gib, reserve_mib=self.config.reserve_mib
        )
        schedule = optimization_plan(self.config, snapshot)
        model = Path(self.config.model_path).expanduser()
        try:
            streaming = plan_model_streaming(model, mmap_enabled=self.config.use_mmap,
                                             mlock_enabled=self.config.use_mlock).__dict__
        except (OSError, ValueError) as exc:
            streaming = {"path": str(model), "available": False, "detail": str(exc)}
        prefetch = prefetch_decision(
            schedule=schedule["admission"],
            requested_bytes=self.config.max_prefetch_mib * 1024**2,
            max_bytes=self.config.max_prefetch_mib * 1024**2,
        )
        return {
            "model_streaming": streaming,
            "paging": paging_report(None),
            "schedule": schedule,
            "prefetch": prefetch,
            "validation": {"status": "not_run", "requires_explicit_invocation": True},
            "limits": {
                "custom_tensor_pager": False,
                "custom_weight_prefetch": False,
                "adaptive_cache_applied_to_model_weights": False,
                "reason": "llama-cpp-python does not expose stable mapped tensor page control; Windows owns page residency.",
            },
        }

    def compare_once(self, prompt: str) -> dict:
        return run_single_comparison(self.config.model_path, prompt, config=self.config,
                                     logs_dir=self.project_root / "logs")
