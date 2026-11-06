"""Bounded llama.cpp inference benchmark for the selected local GGUF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    Path.home() / "Downloads" / "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Responda somente: teste concluído.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reserve-mib", type=int, default=512)
    args = parser.parse_args()

    engine = InferenceEngine(
        args.model,
        n_ctx=args.context,
        n_batch=args.batch,
        n_threads=args.threads,
        reserve_mib=args.reserve_mib,
    )
    try:
        load_metrics = engine.load_model()
        generation = engine.generate(args.prompt, max_tokens=args.max_tokens)
        result = {"load": load_metrics, "generation": generation}
        log_path = ROOT / "logs" / "performance.json"
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"events": []}
        log.setdefault("events", []).append({"event_type": "inference_benchmark", **result})
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    finally:
        engine.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
