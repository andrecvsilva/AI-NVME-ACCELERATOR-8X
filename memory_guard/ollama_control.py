"""Read-only Ollama status and guarded synchronous chat requests."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .monitor import MemorySnapshot, read_memory_snapshot
from .protection import MemoryGuard


@dataclass(frozen=True)
class OllamaStatus:
    online: bool
    models: tuple[dict, ...]
    error: str | None = None


def query_ollama_ps(timeout_seconds: float = 3.0) -> OllamaStatus:
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=timeout_seconds, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return OllamaStatus(False, (), str(exc))
    if result.returncode != 0:
        return OllamaStatus(False, (), (result.stderr or "ollama ps failed").strip())
    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return OllamaStatus(True, ())
    headers = lines[0].split()
    models = []
    for line in lines[1:]:
        values = line.split()
        if values:
            models.append({headers[index]: values[index] for index in range(min(len(headers), len(values)))})
    return OllamaStatus(True, tuple(models))


def stop_model(model: str, *, timeout_seconds: float = 10.0) -> None:
    """Request normal Ollama unload; caller must invoke this only after inference."""
    if not model.strip():
        raise ValueError("model cannot be empty")
    result = subprocess.run(
        ["ollama", "stop", model], capture_output=True, text=True,
        timeout=timeout_seconds, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ollama stop failed").strip())


def guarded_chat(
    model: str,
    prompt: str,
    *,
    guard: MemoryGuard,
    status_provider: Callable[[], OllamaStatus] = query_ollama_ps,
    snapshot_provider: Callable[[], MemorySnapshot] = read_memory_snapshot,
    sample_interval_seconds: float = 1.0,
    base_url: str = "http://localhost:11434",
) -> dict:
    """Run one request only when memory is NORMAL and Ollama has no resident model.

    A critical sample during a request is recorded. The HTTP request is not killed;
    no process or active model is forcibly terminated.
    """
    if not model.strip() or not prompt.strip():
        raise ValueError("model and prompt cannot be empty")
    if sample_interval_seconds < 0.25:
        raise ValueError("sample_interval_seconds must be at least 0.25")
    admission = guard.require_admission()
    status = status_provider()
    if not status.online:
        raise ConnectionError(f"Ollama is unavailable: {status.error or 'unknown error'}")
    if status.models:
        raise RuntimeError("Ollama already has a model loaded; refusing to start another request.")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 128},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    before = admission.snapshot
    samples = [before.to_dict()]
    start = time.perf_counter()
    finished = threading.Event()

    def sample_while_running() -> None:
        while not finished.wait(sample_interval_seconds):
            try:
                samples.append(snapshot_provider().to_dict())
            except Exception:
                # A transient monitor failure must not terminate an active generation.
                continue

    sampler = threading.Thread(target=sample_while_running, name="memory-guard-sampler", daemon=True)
    sampler.start()
    response_data = bytearray()
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            response_data.extend(response.read())
    finally:
        finished.set()
        sampler.join(timeout=sample_interval_seconds + 0.5)
    wall_seconds = time.perf_counter() - start
    after = snapshot_provider()
    samples.append(after.to_dict())
    result = json.loads(response_data.decode("utf-8"))
    total_seconds = float(result.get("total_duration", 0)) / 1e9
    eval_seconds = float(result.get("eval_duration", 0)) / 1e9
    eval_count = int(result.get("eval_count", 0))
    return {
        "model": model,
        "result": "completed",
        "memory_state_before": admission.state,
        "memory_before": before.to_dict(),
        "memory_samples_during": samples[1:-1],
        "memory_after": after.to_dict(),
        "memory_state_after": guard.inspect().state,
        "load_seconds": round(float(result.get("load_duration", 0)) / 1e9, 3),
        "inference_seconds": round(eval_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds > 0 else None,
        "done_reason": result.get("done_reason"),
        "response": (result.get("message") or {}).get("content", ""),
        "ollama_loaded_after": list(status_provider().models),
        "critical_during_request": any(
            guard.thresholds.critical_available_gib * (1024**3) > row["available_bytes"]
            or row["load_percent"] >= guard.thresholds.critical_load_percent
            or row["commit_used_percent"] >= guard.thresholds.critical_commit_percent
            for row in samples[1:-1]
        ),
    }
