"""AI NVMe Accelerator 8X: safe, read-only Stage 1 diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from engine.memory_guard import collect_memory_snapshot
from engine.nvme_io_engine import benchmark_block_sweep, benchmark_file
from engine.system_probe import probe_windows
from adaptive_cache import AdaptiveCacheConfig, AdaptiveCacheEngine
from memory_guard import MemoryGuard, guarded_chat, query_ollama_ps, read_memory_snapshot, stop_model
from memory_guard.logger import record_run
from memory_guard.process_monitor import analyze_ollama_state, collect_ollama_processes, record_process_snapshot
from nvme_io import NvmeBenchmarkManager, NvmeIoConfig, get_device_info
from nvme_io.report import latest_csv_event
from accelerator.config import AcceleratorConfig
from accelerator.runtime import AcceleratorRuntime

ROOT = Path(__file__).resolve().parent


def save_event(event: dict) -> None:
    log_path = ROOT / "logs" / "performance.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log = {"events": []}
    else:
        log = {"events": []}
    log.setdefault("events", []).append(event)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def run_diagnose() -> dict:
    report = probe_windows()
    report["event_type"] = "hardware_diagnostic"
    save_event(report)
    return report


def run_monitor(samples: int, interval: float, budget_gib: float, reserve_mib: int) -> dict:
    import time

    results = []
    for index in range(samples):
        results.append(collect_memory_snapshot(budget_gib=budget_gib, reserve_mib=reserve_mib))
        if index + 1 < samples:
            time.sleep(interval)
    report = {
        "event_type": "memory_storage_monitor",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": results,
        "read_only": True,
        "actions_taken": [],
    }
    save_event(report)
    return report


def _guard_for(args=None) -> MemoryGuard:
    return MemoryGuard()


def run_guard_dashboard(args) -> None:
    import time
    config_path = ROOT / "config" / "adaptive_cache.json"
    cache_engine = None
    cache_error = None
    accelerator_config = AcceleratorConfig.load(ROOT / "config" / "accelerator.json")
    try:
        cache_config = AdaptiveCacheConfig.load(config_path)
        cache_engine = AdaptiveCacheEngine(cache_config, project_root=ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        cache_error = str(exc)
    try:
        while True:
            snap = read_memory_snapshot()
            guard = _guard_for(args).inspect()
            schedule = AcceleratorRuntime(accelerator_config, project_root=ROOT).inspect(snapshot={
                "guard": {"pressure": guard.state.lower()},
                "physical_memory": {"available_bytes": snap.available_bytes},
            })["schedule"]["admission"]
            ollama = query_ollama_ps()
            process_report = collect_ollama_processes()
            process_state = analyze_ollama_state(
                online=ollama.online,
                loaded_model_count=len(ollama.models),
                process_report=process_report,
                guard_state=guard.state,
                residual_suspect_mib=args.residual_suspect_mib,
            )
            process_report.update({
                **process_state,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ram_available_bytes": snap.available_bytes,
                "ram_load_percent": snap.load_percent,
                "processes": process_report.get("processes", []),
            })
            record_process_snapshot(process_report, ROOT / "logs")
            try:
                cache_stats = cache_engine.stats() if cache_engine else None
            except (OSError, RuntimeError, ValueError) as exc:
                cache_stats = None
                cache_error = str(exc)
            loaded = ", ".join(str(item.get("NAME", "unknown")) for item in ollama.models) or "NENHUM"
            os.system("cls")
            print("AI NVME ACCELERATOR 8X — MEMORY GUARD\n")
            print(f"RAM TOTAL:       {snap.total_bytes / 1024**3:.2f} GB")
            print(f"RAM DISPONÍVEL:  {snap.available_bytes / 1024**3:.2f} GB")
            print(f"RAM UTILIZADA:   {snap.load_percent}%")
            print(f"COMMIT:          {snap.commit_used_percent:.1f}%")
            print(f"OLLAMA:          {'ONLINE' if ollama.online else 'OFFLINE'}")
            print(f"MODELO:          {loaded}")
            print()
            if process_report.get("processes"):
                for process in process_report["processes"]:
                    print(f"{process['name'].upper()} (PID {process['pid']}):")
                    print(f"  WORKING SET:      {_display_mib(process.get('working_set_mib'))}")
                    print(f"  WORKING SET PRIV.:{_display_mib(process.get('private_working_set_mib'))}")
                    print(f"  MEMÓRIA PRIVADA:  {_display_mib(process.get('private_mib'))}")
                    print(f"  MEMÓRIA VIRTUAL:  {_display_mib(process.get('virtual_mib'))}")
                    print(f"  THREADS:          {process.get('thread_count') if process.get('thread_count') is not None else 'indisponível'}")
                    uptime = process.get("uptime_seconds")
                    print(f"  TEMPO ATIVO:      {f'{uptime / 3600:.1f} h' if uptime is not None else 'indisponível'}")
                    if process.get("status") != "ok":
                        print(f"  ESTADO:           {process['status']} ({process.get('error') or 'sem detalhe'})")
            elif process_report.get("status") == "unavailable":
                print(f"PROCESSOS OLLAMA: INDISPONÍVEIS — {process_report.get('error')}")
            else:
                print("PROCESSOS OLLAMA: nenhum encontrado")
            print(f"PROCESS MONITOR: {process_state['classification']}")
            print(f"MEMORY GUARD:    {guard.state}")
            print(f"MODEL STRATEGY:  {'MMAP/OS-PAGED (when loaded)' if accelerator_config.use_mmap else 'DISABLED'}")
            print(f"RAM/NVME PLAN:   {'ADMITTED' if schedule['allowed'] else 'BLOCKED'} — ctx {schedule['context']}, batch {schedule['batch']}, threads {schedule['threads']}")
            print(f"PLAN DETAIL:     {schedule['reason']}")
            print(f"WEIGHT PREFETCH: {'ELIGIBLE, NOT EXECUTED' if schedule['allow_prefetch'] else 'DISABLED'} (mapping API unavailable)")
            print()
            if cache_stats:
                ram = cache_stats["ram"]
                nvme = cache_stats["nvme"]
                metrics = cache_stats["metrics"]
                print(f"ADAPTIVE CACHE:  {'ACTIVE' if cache_stats['cache_active'] else 'DISABLED'}")
                print(f"RAM CACHE:       {ram['used_accounted_bytes'] / 1024**2:.2f} / {ram['capacity_bytes'] / 1024**2:.2f} MiB")
                print(f"NVME CACHE:      {nvme['used_bytes'] / 1024**2:.2f} / {nvme['limit_bytes'] / 1024**2:.2f} MiB")
                print(f"NVME FREE DISK:  {nvme['free_bytes'] / 1024**3:.2f} GiB")
                print(f"CACHE HITS:      {metrics['hits']}  MISSES: {metrics['misses']}  HIT RATE: {metrics['hit_rate'] if metrics['hit_rate'] is not None else 'n/a'}")
                print(f"PREFETCH:        {'ENABLED' if cache_stats['prefetch_enabled'] else 'DISABLED'}")
            else:
                print(f"ADAPTIVE CACHE:  DISABLED — {cache_error or 'unavailable'}")
            latest_io = latest_csv_event(ROOT / "logs" / "nvme_io.csv")
            if latest_io:
                print(f"NVME I/O (ÚLTIMA MEDIÇÃO): {latest_io.get('timestamp_utc', 'data indisponível')} — "
                      f"{latest_io.get('operation', 'operação')} — {latest_io.get('throughput_mb_s') or 'n/a'} MB/s; "
                      f"IOPS {latest_io.get('iops') or 'n/a'}; P95 {latest_io.get('latency_p95_ms') or 'n/a'} ms")
                print(f"  CACHE: {latest_io.get('windows_cache_state') or 'estado desconhecido'}")
            else:
                print("NVME I/O:       sem medições registradas")
            if not ollama.online:
                print(f"DETALHE:         {ollama.error}")
            print("\nCtrl+C para sair; nenhuma ação automática encerra processos ou descarrega modelos.")
            time.sleep(args.interval)
    finally:
        if cache_engine:
            cache_engine.close()


def _display_mib(value) -> str:
    return f"{value:.2f} MiB" if value is not None else "indisponível"


def run_accelerator(args) -> dict:
    from dataclasses import replace

    config = AcceleratorConfig.load(ROOT / "config" / "accelerator.json")
    if args.model:
        config = replace(config, model_path=str(args.model))
        config.validate()
    model = Path(config.model_path).expanduser()
    runtime = AcceleratorRuntime(config, project_root=ROOT)
    if args.action == "profile-load":
        if not args.run:
            raise ValueError("Load profiling opens the GGUF but does not generate tokens; pass --run to start it explicitly.")
        from engine.inference_engine import InferenceEngine
        engine = InferenceEngine(
            model, n_ctx=config.max_context, n_batch=config.max_batch,
            n_threads=config.max_threads, memory_budget_gib=config.memory_budget_gib,
            reserve_mib=config.reserve_mib, observation_interval_seconds=0.25,
            use_extra_bufts=config.use_extra_bufts,
            physical_disk_instance=config.physical_disk_instance,
            profile_path=ROOT / "logs" / "model_load_profiles.jsonl",
        )
        try:
            metrics = engine.load_model()
            return {"event_type": "model_load_profile", "status": "completed",
                    "generation_started": False, **metrics}
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            return {"event_type": "model_load_profile", "status": "blocked_or_failed",
                    "generation_started": False, "error": str(exc),
                    "diagnostics": engine.last_load_diagnostics}
        finally:
            engine.unload_model()
    if args.action == "infer":
        if not args.run:
            raise ValueError("Inference requires explicit --run; it can create high memory pressure.")
        max_tokens = args.max_tokens or config.max_generation_tokens
        if not 1 <= max_tokens <= min(128, config.max_context - 1):
            raise ValueError("--max-tokens is outside the configured context limit")
        from engine.inference_engine import InferenceEngine
        engine = InferenceEngine(
            model, n_ctx=config.max_context, n_batch=config.max_batch,
            n_threads=config.max_threads, memory_budget_gib=config.memory_budget_gib,
            reserve_mib=config.reserve_mib,
            observation_interval_seconds=config.observation_interval_seconds,
            use_extra_bufts=config.use_extra_bufts,
            physical_disk_instance=config.physical_disk_instance,
            capture_etw=args.capture_etw,
            profile_path=ROOT / "logs" / "model_load_profiles.jsonl",
        )
        try:
            loaded = engine.load_model()
            generated = engine.generate(
                args.prompt or "Responda somente: teste de leitura do modelo.",
                max_tokens=max_tokens, temperature=0,
            )
            return {"event_type": "accelerator_single_inference", "load": loaded,
                    "generation": generated, "generation_started": True}
        finally:
            engine.unload_model()
    if args.action == "compare":
        if not args.run:
            raise ValueError("Comparison can load a model; pass --run only when ready to explicitly run inference.")
        return runtime.compare_once(args.prompt or "Responda somente: teste de validação.")

    result = runtime.inspect()
    result["event_type"] = "accelerator_status" if args.action == "status" else "accelerator_plan"
    result["notes"] = [
        "mmap maps file-backed model pages; it does not make SSD latency equivalent to RAM.",
        "The Python wrapper cannot identify GGUF-only resident pages or prefetch individual tensors.",
        "No inference or benchmark was run by this status/plan command.",
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe hardware diagnostics and read-only disk benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("diagnose", help="Report CPU, RAM, Windows, disks, volumes and pagefiles.")
    monitor = subparsers.add_parser("monitor", help="Sample RAM, commit, pagefile and C: free space without changing settings.")
    monitor.add_argument("--samples", type=int, default=5, help="Number of snapshots (1..300).")
    monitor.add_argument("--interval", type=float, default=1.0, help="Seconds between snapshots (0.25..60).")
    monitor.add_argument("--budget-gib", type=float, default=3.0, help="Configured model/cache budget; does not reserve RAM.")
    monitor.add_argument("--reserve-mib", type=int, default=512, help="Safety reserve subtracted from available/budget memory.")
    disk = subparsers.add_parser("disk-read", help="Read a user-selected existing file; never writes to it.")
    disk.add_argument("file", type=Path, help="Existing local file, ideally the GGUF model on the NVMe.")
    disk.add_argument("--max-mib", type=int, default=256, help="Maximum bytes to read per pass (1..1024 MiB).")
    disk.add_argument("--block-kib", type=int, default=1024, help="Sequential read block size (4..4096 KiB).")
    disk.add_argument("--random-ops", type=int, default=128, help="Number of 4 KiB random reads (0..4096).")
    sweep = subparsers.add_parser("disk-sweep", help="Compare sequential reads at five block sizes; total read is capped at 320 MiB.")
    sweep.add_argument("file", type=Path, help="Existing local file, ideally the GGUF model on the NVMe.")
    sweep.add_argument("--mib-per-block", type=int, default=32, help="MiB read per block size (1..64; 5 sizes maximum).")
    sweep.add_argument("--uncached", action="store_true", help="Use Windows aligned FILE_FLAG_NO_BUFFERING reads (bypasses system file cache).")
    dashboard = subparsers.add_parser("memory-guard", help="Show live RAM and Ollama status; Ctrl+C exits.")
    dashboard.add_argument("--interval", type=float, default=2.0)
    dashboard.add_argument("--residual-suspect-mib", type=int, default=1536,
                           help="Private-memory threshold for a diagnostic residual suspicion; never proves a leak.")
    nvme = subparsers.add_parser("nvme-io", help="Identify storage or explicitly run bounded read-only NVMe measurements.")
    nvme_actions = nvme.add_mutually_exclusive_group()
    nvme_actions.add_argument("--info", action="store_true", help="Read Windows device/volume information only (default).")
    nvme_actions.add_argument("--benchmark", action="store_true", help="Explicitly create bounded temporary test data and measure reads.")
    nvme.add_argument("--drive", help="Windows drive letter to inspect or test, e.g. C:")
    etw_test = subparsers.add_parser(
        "etw-file-test", help="Capture bounded unbuffered reads from a small owned file to validate ETW file attribution (requires elevated PowerShell)."
    )
    etw_test.add_argument("--size-mib", type=int, default=4,
                          help="Temporary test-file size, 1..16 MiB (default 4).")
    chat = subparsers.add_parser("ollama-chat", help="Send one chat only when memory is NORMAL and Ollama has no loaded model.")
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--sample-interval", type=float, default=1.0)
    stop = subparsers.add_parser("ollama-stop", help="Request normal unload after inference has finished.")
    stop.add_argument("--model", required=True)
    stop.add_argument("--confirm-idle", action="store_true", help="Confirm no inference is active in any client.")
    accelerator = subparsers.add_parser("accelerator", help="Show the integrated mmap, paging, scheduler, and prefetch plan.")
    accelerator.add_argument("action", nargs="?", choices=("status", "plan", "compare", "profile-load", "infer"), default="status")
    accelerator.add_argument("--model", type=Path, help="GGUF path (defaults to config/accelerator.json).")
    accelerator.add_argument("--run", action="store_true", help="Explicit opt-in: compare may generate; profile-load only loads and immediately unloads, without generation.")
    accelerator.add_argument("--prompt", help="Prompt for the explicit comparison run.")
    accelerator.add_argument("--max-tokens", type=int, help="Token cap for the explicit single inference action.")
    accelerator.add_argument("--capture-etw", action="store_true", help="Capture WPR DiskIO/FileIO ETW during `accelerator infer` (requires elevated PowerShell).")
    args = parser.parse_args()

    try:
        if args.command == "diagnose":
            result = run_diagnose()
        elif args.command == "monitor":
            if not 1 <= args.samples <= 300:
                raise ValueError("samples must be between 1 and 300")
            if not 0.25 <= args.interval <= 60:
                raise ValueError("interval must be between 0.25 and 60 seconds")
            if not 0.25 <= args.budget_gib <= 64:
                raise ValueError("budget_gib must be between 0.25 and 64")
            if not 0 <= args.reserve_mib <= 8192:
                raise ValueError("reserve_mib must be between 0 and 8192")
            result = run_monitor(args.samples, args.interval, args.budget_gib, args.reserve_mib)
        elif args.command == "memory-guard":
            if not 0.5 <= args.interval <= 60:
                raise ValueError("dashboard interval must be between 0.5 and 60 seconds")
            run_guard_dashboard(args)
            return 0
        elif args.command == "nvme-io":
            io_config = NvmeIoConfig.load(ROOT / "config" / "nvme_io.json")
            if args.drive:
                from dataclasses import replace
                io_config = replace(io_config, drive=args.drive)
            if args.benchmark:
                manager = NvmeBenchmarkManager(io_config, project_root=ROOT, memory_guard=MemoryGuard())
                result = manager.run_benchmark()
            else:
                result = get_device_info(io_config.drive)
        elif args.command == "etw-file-test":
            from engine.etw_file_test import run_etw_file_test
            result = run_etw_file_test(ROOT / "logs", size_mib=args.size_mib)
        elif args.command == "ollama-chat":
            guard = _guard_for(args)
            try:
                result = guarded_chat(
                    args.model, args.prompt, guard=guard,
                    sample_interval_seconds=args.sample_interval,
                )
                record_run(result, ROOT / "reports")
            except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                try:
                    snapshot = guard.inspect().snapshot.to_dict()
                except (OSError, RuntimeError):
                    snapshot = {}
                record_run({
                    "model": args.model,
                    "result": "blocked" if isinstance(exc, MemoryError) else "failed",
                    "memory_state_before": guard.inspect().state if snapshot else "UNKNOWN",
                    "memory_before": snapshot,
                    "error": str(exc),
                    "tokens": 0,
                    "critical_during_request": False,
                }, ROOT / "reports")
                raise
        elif args.command == "ollama-stop":
            if not args.confirm_idle:
                raise ValueError("Refusing to unload: verify no inference is active, then pass --confirm-idle")
            status = query_ollama_ps()
            if not status.online:
                raise ConnectionError(f"Ollama is unavailable: {status.error}")
            if not any(item.get("NAME") == args.model for item in status.models):
                raise ValueError(f"Model is not currently loaded: {args.model}")
            stop_model(args.model)
            result = {"event_type": "ollama_model_unloaded", "model": args.model, "requested_by_user": True}
        elif args.command == "accelerator":
            result = run_accelerator(args)
        elif args.command == "disk-sweep":
            result = benchmark_block_sweep(args.file, mib_per_block=args.mib_per_block, bypass_cache=args.uncached)
            result["event_type"] = "read_only_block_size_sweep"
            result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            save_event(result)
        else:
            result = benchmark_file(
                args.file,
                max_mib=args.max_mib,
                block_kib=args.block_kib,
                random_ops=args.random_ops,
            )
            result["event_type"] = "read_only_disk_benchmark"
            result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            save_event(result)
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
