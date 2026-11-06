"""Guarded llama.cpp integration using file-backed mmap and CPU-only execution."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Callable

from engine.memory_guard import GIB, MIB, collect_memory_snapshot
from engine.disk_io_observer import PhysicalDiskIoObserver
from engine.etw_io_trace import WprEtwCapture
from engine.model_memory_observer import ModelMemoryObserver
from accelerator.config import AcceleratorConfig
from accelerator.scheduler import AdaptiveRamNvmeScheduler
from accelerator.model_streaming import plan_model_streaming
from accelerator.paging import paging_report

_LOAD_LOCK = threading.Lock()
_LLAMA_DEFAULT_PARAMS_LOCK = threading.Lock()
_ACTIVE_ENGINE = None


def _construct_llama(llama_factory: Callable, options: dict, *, use_extra_bufts: bool):
    """Construct llama-cpp-python with an explicit native extra-buffer policy.

    llama-cpp-python 0.3.36 does not expose ``use_extra_bufts`` as a public
    constructor argument. Its native model params do expose it, so set the
    default only for the duration of construction and restore it immediately.
    The project load lock and this lock serialize that temporary module patch.
    """
    if use_extra_bufts:
        return llama_factory(**options)

    import llama_cpp.llama_cpp as llama_api

    with _LLAMA_DEFAULT_PARAMS_LOCK:
        original = llama_api.llama_model_default_params

        def configured_defaults():
            params = original()
            if not hasattr(params, "use_extra_bufts"):
                raise RuntimeError(
                    "Installed llama-cpp-python binding does not expose native use_extra_bufts; "
                    "cannot safely disable CPU weight repacking."
                )
            params.use_extra_bufts = False
            return params

        llama_api.llama_model_default_params = configured_defaults
        try:
            return llama_factory(**options)
        finally:
            llama_api.llama_model_default_params = original


@contextmanager
def _capture_native_load_output(path: Path):
    """Capture C/C++ stdout and stderr so llama.cpp buffer lines are retained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        c_runtime = __import__("ctypes").CDLL(None)
        c_runtime.fflush(None)
    except Exception:
        c_runtime = None
    original_out = original_err = target = None
    redirected = False
    try:
        original_out, original_err = os.dup(1), os.dup(2)
        target = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        os.dup2(target, 1)
        os.dup2(target, 2)
        redirected = True
    except OSError:
        # Some Windows terminals expose handles that the CRT cannot duplicate.
        # Keep loading functional and let native llama.cpp output go to console.
        if original_out is not None:
            try:
                os.dup2(original_out, 1)
            except OSError:
                pass
        if original_err is not None:
            try:
                os.dup2(original_err, 2)
            except OSError:
                pass
        for descriptor in (target, original_out, original_err):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        yield
        return
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            if c_runtime is not None:
                c_runtime.fflush(None)
            if redirected:
                os.dup2(original_out, 1)
                os.dup2(original_err, 2)
        finally:
            for descriptor in (target, original_out, original_err):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass


class InferenceEngine:
    """Loads one GGUF at a time with mmap and conservative context/batch defaults."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 512,
        n_batch: int = 16,
        n_threads: int = 4,
        memory_budget_gib: float = 3.0,
        reserve_mib: int = 512,
        memory_probe: Callable[..., dict] = collect_memory_snapshot,
        observation_interval_seconds: float = 1.0,
        process_memory_probe: Callable[[], dict] | None = None,
        adaptive_schedule: bool = True,
        use_extra_bufts: bool = False,
        physical_disk_instance: str = "0 C:",
        disk_io_observer_factory: Callable | None = None,
        capture_etw: bool = False,
        etw_capture_factory: Callable | None = None,
        model_memory_observer_factory: Callable | None = None,
        profile_path: str | Path | None = None,
    ):
        if not 128 <= n_ctx <= 4096:
            raise ValueError("n_ctx must be between 128 and 4096 for this memory-conscious prototype")
        if not 1 <= n_batch <= 128:
            raise ValueError("n_batch must be between 1 and 128")
        if not 1 <= n_threads <= 8:
            raise ValueError("n_threads must be between 1 and 8")
        if memory_budget_gib <= 0 or reserve_mib < 0:
            raise ValueError("memory budget must be positive and reserve cannot be negative")
        self.model_path = Path(model_path).expanduser().resolve()
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.memory_budget_gib = memory_budget_gib
        self.reserve_mib = reserve_mib
        self._memory_probe = memory_probe
        self.adaptive_schedule = adaptive_schedule
        self.use_extra_bufts = bool(use_extra_bufts)
        self.physical_disk_instance = physical_disk_instance
        self._disk_io_observer_factory = disk_io_observer_factory or PhysicalDiskIoObserver
        self.capture_etw = bool(capture_etw)
        self._etw_capture_factory = etw_capture_factory or WprEtwCapture
        self._model_memory_observer_factory = model_memory_observer_factory or ModelMemoryObserver
        self.profile_path = (Path(profile_path) if profile_path else
                             Path(__file__).resolve().parents[1] / "logs" / "model_load_profiles.jsonl")
        self._scheduler = AdaptiveRamNvmeScheduler(AcceleratorConfig(
            memory_budget_gib=memory_budget_gib,
            reserve_mib=max(256, reserve_mib),
            max_context=n_ctx,
            min_context=min(128, n_ctx),
            max_batch=n_batch,
            min_batch=min(4, n_batch),
            max_threads=n_threads,
            observation_interval_seconds=observation_interval_seconds,
        ))
        # Loading gets a fixed quarter-second sampling cadence as requested;
        # it is independent of the slower UI/generation polling setting.
        observer_args = {"interval_seconds": 0.25, "memory_probe": memory_probe}
        if process_memory_probe is not None:
            observer_args["process_probe"] = process_memory_probe
        self._observer_args = observer_args
        self._model = None
        self._etw_capture_instance = None
        self.load_metrics: dict | None = None
        self.last_load_diagnostics: dict | None = None
        self._generation_lock = threading.Lock()

    def load_model(self, llm_factory: Callable | None = None) -> dict:
        global _ACTIVE_ENGINE
        if self._model is not None:
            return self.load_metrics or {}
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")
        if self.capture_etw and self._etw_capture_instance is None:
            try:
                self._etw_capture_instance = self._etw_capture_factory(
                    self.profile_path.parent, gguf_path=self.model_path
                )
                etw_preflight = self._etw_capture_instance.preflight()
            except Exception as exc:
                raise RuntimeError(f"ETW preflight failed before model loading: {exc}") from exc
            if not etw_preflight.get("available"):
                self._etw_capture_instance = None
                raise RuntimeError(
                    "ETW prerequisites failed; model loading was not started. "
                    + str(etw_preflight.get("reason", "unknown WPR preflight error"))
                )
        streaming_plan = plan_model_streaming(self.model_path, mmap_enabled=True, mlock_enabled=False)
        with _LOAD_LOCK:
            if _ACTIVE_ENGINE is not None and _ACTIVE_ENGINE is not self:
                raise RuntimeError("Another model is already loaded; unload it before loading this model.")
            before = self._memory_probe(budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib)
            guard = before.get("guard", {})
            schedule = self._scheduler.decide(before)
            if not schedule.allowed:
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                blocked = {
                    "event_type": "model_load_profile_blocked_before_start",
                    "timestamp_utc": before.get("timestamp_utc"),
                    "model_path": str(self.model_path),
                    "guard_state": guard.get("pressure", "unknown"),
                    "physical_memory": before.get("physical_memory"),
                    "virtual_commit": before.get("virtual_commit"),
                    "thresholds": guard.get("thresholds"),
                    "reason": schedule.reason,
                    "load_started": False,
                    "generation_started": False,
                }
                with self.profile_path.open("a", encoding="utf-8") as stream:
                    stream.write(__import__("json").dumps(blocked, ensure_ascii=False, separators=(",", ":")) + "\n")
                self.last_load_diagnostics = blocked
                raise RuntimeError(f"Model loading blocked by Memory Guard: {schedule.reason}")
            native_llama_factory = llm_factory is None
            if llm_factory is None:
                # Some Windows installations retain a stale CUDA_PATH even when
                # the selected llama.cpp wheel is CPU-only. The package attempts
                # to register that path while importing, so suppress only a
                # nonexistent path for this import; never mutate system settings.
                cuda_path = os.environ.get("CUDA_PATH")
                clear_stale_cuda_path = bool(cuda_path and not Path(cuda_path, "bin").is_dir())
                if clear_stale_cuda_path:
                    os.environ.pop("CUDA_PATH", None)
                try:
                    from llama_cpp import Llama
                except ImportError as exc:
                    raise RuntimeError(
                        "llama-cpp-python is not installed. Follow the CPU installation steps in README.md."
                    ) from exc
                finally:
                    if clear_stale_cuda_path:
                        os.environ["CUDA_PATH"] = cuda_path
                llm_factory = Llama

            options = {
                "model_path": str(self.model_path),
                "n_ctx": schedule.context if self.adaptive_schedule else self.n_ctx,
                "n_batch": schedule.batch if self.adaptive_schedule else self.n_batch,
                "n_ubatch": schedule.batch if self.adaptive_schedule else self.n_batch,
                "n_threads": schedule.threads if self.adaptive_schedule else self.n_threads,
                "n_threads_batch": schedule.threads if self.adaptive_schedule else self.n_threads,
                "n_gpu_layers": 0,
                "use_mmap": True,
                "use_mlock": False,
                "verbose": True,
            }
            native_log = self.profile_path.with_name(
                f"llama_native_load_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
            )
            observer = self._model_memory_observer_factory(**self._observer_args).start(
                budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib,
                model_path=self.model_path, profile_path=self.profile_path,
            )
            if observer.summary().get("peak_pressure") != "normal":
                observation = observer.stop(status="blocked_before_load")
                self.last_load_diagnostics = {"status": "blocked_before_load", "observation": observation}
                raise RuntimeError(
                    f"Memory Guard changed to {observation.get('peak_pressure', 'unknown').upper()} "
                    "before model loading; model loading was not started."
                )
            started = time.perf_counter()
            model = None
            try:
                output_capture = _capture_native_load_output(native_log) if native_llama_factory else nullcontext()
                with output_capture:
                    if native_llama_factory:
                        model = _construct_llama(
                            llm_factory, options, use_extra_bufts=self.use_extra_bufts
                        )
                    else:
                        model = llm_factory(**options)
            except BaseException as exc:
                load_observation = observer.stop(status="loader_exception", error=str(exc))
                self.last_load_diagnostics = {
                    "status": "loader_exception", "load_seconds": round(time.perf_counter() - started, 3),
                    "memory_observation_during_load": load_observation,
                    "native_log_path": str(native_log) if native_llama_factory else None, "error": str(exc),
                }
                raise
            load_observation = observer.stop(status="loaded")
            after_load = self._memory_probe(
                budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib
            )
            after_load_pressure = (after_load.get("guard") or {}).get("pressure", "unknown")
            elapsed = time.perf_counter() - started
            peak_load_pressure = load_observation.get("peak_pressure", "unknown")
            if peak_load_pressure in {"critical", "unknown"} or after_load_pressure != "normal":
                close_method = getattr(model, "close", None) if model is not None else None
                if callable(close_method):
                    close_method()
                del model
                gc.collect()
                self.last_load_diagnostics = {
                    "status": "guard_pressure_after_load", "load_seconds": round(elapsed, 3),
                    "memory_observation_during_load": load_observation,
                    "memory_pressure_after_load": after_load_pressure,
                    "native_log_path": str(native_log) if native_llama_factory else None,
                }
                raise RuntimeError(
                    f"Memory Guard peak was {peak_load_pressure.upper()} during loading and current pressure is "
                    f"{after_load_pressure.upper()}; "
                    "model was closed after loading completed and generation was not started."
                )
            self._model = model
            _ACTIVE_ENGINE = self
            model_size = self.model_path.stat().st_size
            self.load_metrics = {
                "model_path": str(self.model_path),
                "model_file_bytes": model_size,
                "model_streaming_plan": streaming_plan.__dict__,
                "model_file_exceeds_configured_memory_budget": model_size > int(self.memory_budget_gib * GIB),
                "model_file_exceeds_available_ram_before_load": (
                    model_size > int((before.get("physical_memory") or {}).get("available_bytes", 0))
                ),
                "mmap_note": "A mapped file can exceed RAM capacity, but pages may be repeatedly read from SSD and generation can crawl.",
                "load_seconds": round(elapsed, 3),
                "memory_pressure_before_load": guard.get("pressure", "unknown"),
                "memory_observation_during_load": load_observation,
                "memory_pressure_after_load": after_load_pressure,
                "native_log_path": str(native_log) if native_llama_factory else None,
                "paging_observation": paging_report(load_observation),
                "observation_scope_note": (
                    "System RAM is sampled during load/generation. Process working set/private bytes include all process memory; "
                    "Windows does not attribute these totals exclusively to GGUF mapped pages here. SSD bytes read are not measured."
                ),
                "options": {key: value for key, value in options.items() if key != "model_path"},
                "use_extra_bufts": self.use_extra_bufts,
                "native_buffer_policy_applied": native_llama_factory,
                "cpu_weight_repack_disabled": native_llama_factory and not self.use_extra_bufts,
                "mmap_enabled": schedule.mmap,
                "gpu_layers": 0,
                "adaptive_schedule": schedule.to_dict() if self.adaptive_schedule else {**schedule.to_dict(), "applied": False},
            }
            self.last_load_diagnostics = self.load_metrics
            return self.load_metrics

    def generate(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.2) -> dict:
        if self._model is None:
            raise RuntimeError("Load the model before generating.")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not 1 <= max_tokens <= min(128, self.n_ctx - 1):
            raise ValueError(f"max_tokens must be between 1 and {min(128, self.n_ctx - 1)}")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        before = self._memory_probe(budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib)
        schedule = self._scheduler.decide(before)
        if not schedule.allowed:
            raise RuntimeError(f"Generation blocked by Memory Guard: {schedule.reason}")
        observer = ModelMemoryObserver(**self._observer_args).start(
            budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib
        )
        if observer.summary().get("peak_pressure") != "normal":
            observation = observer.stop()
            raise RuntimeError(
                f"Memory Guard changed to {observation.get('peak_pressure', 'unknown').upper()} "
                "before generation; generation was not started."
            )

        etw_capture = None
        etw_start = {"available": False, "started": False,
                     "reason": "ETW capture was not requested."}
        if self.capture_etw:
            try:
                etw_capture = self._etw_capture_instance or self._etw_capture_factory(
                    self.profile_path.parent, gguf_path=self.model_path
                )
                self._etw_capture_instance = etw_capture
                etw_start = etw_capture.start()
            except Exception as exc:
                observer.stop()
                raise RuntimeError(f"ETW capture failed before generation: {exc}") from exc
            if not etw_start.get("started"):
                observer.stop()
                raise RuntimeError(
                    "ETW capture was requested but could not start; generation was not started. "
                    + str(etw_start.get("reason", "unknown WPR error"))
                )
        disk_observer = None
        try:
            disk_observer = self._disk_io_observer_factory(
                instance=self.physical_disk_instance, interval_seconds=0.25
            ).start()
        except Exception:
            # Disk counters are diagnostic only and must never prevent inference.
            disk_observer = None

        started = time.perf_counter()
        try:
            with self._generation_lock:
                result = self._model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        finally:
            generation_observation = observer.stop()
            disk_observation = disk_observer.stop() if disk_observer is not None else {
                "available": False,
                "limitation": "Physical disk observer could not start; no disk counters were collected.",
            }
            if etw_capture is not None:
                try:
                    etw_observation = etw_capture.stop()
                except Exception as exc:
                    etw_observation = {"available": False, "started": True,
                                       "reason": f"ETW stop/export failed: {type(exc).__name__}: {exc}"}
            else:
                etw_observation = etw_start
        elapsed = time.perf_counter() - started
        after = self._memory_probe(budget_gib=self.memory_budget_gib, reserve_mib=self.reserve_mib)
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if completion_tokens == 0 and text:
            try:
                completion_tokens = len(self._model.tokenize(text.encode("utf-8"), add_bos=False, special=False))
            except Exception:
                completion_tokens = 0
        paging = paging_report(generation_observation)
        etw_gguf = (etw_observation.get("gguf_analysis")
                    if isinstance(etw_observation, dict) else None)
        correlation = {
            "generation_seconds": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(completion_tokens / elapsed, 3)
                if elapsed > 0 and completion_tokens else None,
            "memory_minimum_available_bytes": generation_observation.get("minimum_available_bytes"),
            "memory_peak_load_percent": generation_observation.get("peak_load_percent"),
            "process_page_fault_count_delta": paging.get("process_page_fault_count_delta"),
            "process_read_transfer_bytes_delta": paging.get("process_read_transfer_bytes_delta"),
            "physical_disk_io": disk_observation,
            "gguf_physical_io": etw_gguf,
            "scope_note": (
                "Metrics share the generation interval, but page faults and process reads are process-wide; "
                "only gguf_physical_io is ETW-attributed by FileObject. Correlation does not prove storage wait caused token latency."
            ),
        }
        return {
            "text": text,
            "completion_tokens": completion_tokens,
            "generation_seconds": round(elapsed, 3),
            "tokens_per_second": round(completion_tokens / elapsed, 3) if elapsed > 0 and completion_tokens else None,
            "memory_before": before,
            "memory_after": after,
            "memory_observation_during_generation": generation_observation,
            "physical_disk_io_during_generation": disk_observation,
            "etw_io_capture": etw_observation,
            "generation_io_memory_correlation": correlation,
            "memory_pressure_peak_during_generation": generation_observation.get("peak_pressure", "unknown"),
            "critical_observed_during_generation": generation_observation.get("peak_pressure") == "critical",
            "paging_observation": paging,
        }

    def unload_model(self) -> None:
        global _ACTIVE_ENGINE
        with _LOAD_LOCK:
            if self._model is not None:
                model = self._model
                self._model = None
                close_method = getattr(model, "close", None)
                if callable(close_method):
                    close_method()
                del model
                gc.collect()
            if _ACTIVE_ENGINE is self:
                _ACTIVE_ENGINE = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.unload_model()
