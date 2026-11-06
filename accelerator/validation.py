"""Explicit, sequential profile comparison; never runs unless requested by caller."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading

from engine.memory_guard import collect_memory_snapshot
from engine.inference_engine import InferenceEngine

_RUN_LOCK = threading.Lock()


def _append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def _profile_metrics(result: dict) -> dict:
    load = result.get("load") or {}
    generation = result.get("generation") or {}
    paging = generation.get("paging_observation") or {}
    return {
        "status": result.get("status"),
        "load_seconds": load.get("load_seconds"),
        "generation_seconds": generation.get("generation_seconds"),
        "tokens": generation.get("completion_tokens"),
        "tokens_per_second": generation.get("tokens_per_second"),
        "minimum_system_available_ram_bytes": paging.get("minimum_system_available_bytes"),
        "peak_process_working_set_bytes": paging.get("peak_process_working_set_bytes"),
        "process_page_fault_count_delta": paging.get("process_page_fault_count_delta"),
        "process_read_transfer_bytes_delta": paging.get("process_read_transfer_bytes_delta"),
        "critical_observed": generation.get("critical_observed_during_generation"),
        "hard_page_faults": None,
        "measurement_limit": "PageFaultCount and read counters are process-wide; they do not isolate GGUF or prove physical SSD reads.",
    }


def run_single_comparison(model_path: str | Path, prompt: str, *, config,
                          logs_dir: str | Path) -> dict:
    """Run one guarded baseline then one adaptive profile, serially; explicit opt-in only."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("An accelerator comparison is already active")
    results = []
    try:
        for profile in ("baseline_mmap", "adaptive_mmap"):
            snapshot = collect_memory_snapshot(budget_gib=config.memory_budget_gib,
                                               reserve_mib=config.reserve_mib)
            if snapshot.get("guard", {}).get("pressure") != "normal":
                results.append({"profile": profile, "status": "blocked",
                                "reason": "Memory Guard is not NORMAL; no inference started."})
                break
            options = ({"n_ctx": config.max_context, "n_batch": config.max_batch,
                        "n_threads": config.max_threads} if profile == "baseline_mmap" else {})
            engine = InferenceEngine(
                model_path, memory_budget_gib=config.memory_budget_gib,
                reserve_mib=config.reserve_mib,
                observation_interval_seconds=config.observation_interval_seconds,
                adaptive_schedule=(profile == "adaptive_mmap"),
                use_extra_bufts=config.use_extra_bufts,
                physical_disk_instance=config.physical_disk_instance,
                **options,
            )
            try:
                loaded = engine.load_model()
                generated = engine.generate(prompt, max_tokens=config.max_generation_tokens, temperature=0)
                results.append({"profile": profile, "status": "completed", "load": loaded,
                                "generation": generated})
            except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                results.append({"profile": profile, "status": "failed", "error": str(exc)})
                break
            finally:
                engine.unload_model()
        by_profile = {item.get("profile"): item for item in results if item.get("status") == "completed"}
        baseline = by_profile.get("baseline_mmap", {})
        adaptive = by_profile.get("adaptive_mmap", {})
        baseline_metrics = _profile_metrics(baseline) if baseline else {}
        adaptive_metrics = _profile_metrics(adaptive) if adaptive else {}
        baseline_rate = (baseline.get("generation") or {}).get("tokens_per_second")
        adaptive_rate = (adaptive.get("generation") or {}).get("tokens_per_second")
        comparison = {
            "complete": bool(baseline and adaptive),
            "baseline_tokens_per_second": baseline_rate,
            "adaptive_tokens_per_second": adaptive_rate,
            "adaptive_over_baseline_tokens_per_second": (
                round(adaptive_rate / baseline_rate, 4)
                if baseline_rate and adaptive_rate else None
            ),
            "baseline_load_seconds": (baseline.get("load") or {}).get("load_seconds"),
            "adaptive_load_seconds": (adaptive.get("load") or {}).get("load_seconds"),
            "by_profile": {"baseline_mmap": baseline_metrics, "adaptive_mmap": adaptive_metrics},
            "adaptive_minus_baseline_minimum_ram_bytes": (
                adaptive_metrics.get("minimum_system_available_ram_bytes", 0)
                - baseline_metrics.get("minimum_system_available_ram_bytes", 0)
                if baseline and adaptive and adaptive_metrics.get("minimum_system_available_ram_bytes") is not None
                and baseline_metrics.get("minimum_system_available_ram_bytes") is not None else None
            ),
            "one_repetition_only": True,
            "hard_page_faults_measured": False,
        }
        event = {
            "event_type": "accelerator_profile_comparison",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(Path(model_path).expanduser().resolve()),
            "results": results,
            "comparison": comparison,
            "sequential": True,
            "repetitions": 1,
            "speedup_claimed": False,
            "note": "One observation per profile is exploratory; not a statistically validated benchmark.",
        }
        _append_jsonl(Path(logs_dir) / "accelerator_runs.jsonl", event)
        return event
    finally:
        _RUN_LOCK.release()
