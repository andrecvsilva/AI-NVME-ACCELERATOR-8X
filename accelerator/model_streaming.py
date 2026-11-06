"""GGUF file-backed loading plan delegated to llama.cpp mmap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelStreamingPlan:
    path: str
    file_bytes: int
    mmap_enabled: bool
    mlock_enabled: bool
    load_strategy: str
    note: str


def plan_model_streaming(path: str | Path, *, mmap_enabled: bool = True,
                         mlock_enabled: bool = False) -> ModelStreamingPlan:
    model = Path(path).expanduser().resolve(strict=True)
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("Model streaming currently requires an existing GGUF file")
    if not mmap_enabled:
        raise ValueError("This prototype requires mmap; eager loading is outside its memory policy")
    if mlock_enabled:
        raise ValueError("mlock can pin pages in RAM and is disallowed by the safe profile")
    return ModelStreamingPlan(
        path=str(model), file_bytes=model.stat().st_size, mmap_enabled=True, mlock_enabled=False,
        load_strategy="llama.cpp file-backed mmap; Windows faults mapped pages as needed",
        note=("This avoids an eager Python copy. It does not provide a custom tensor pager, guarantee "
              "that all weights fit in RAM, or guarantee one-pass SSD streaming during token generation."),
    )
