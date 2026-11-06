"""Append-only JSON and CSV records for Memory Guard operations."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOCK = Lock()
_CSV_FIELDS = [
    "timestamp_utc", "model", "result", "guard_state", "ram_total_bytes",
    "ram_available_before_bytes", "ram_available_during_min_bytes",
    "ram_available_after_bytes", "load_seconds", "inference_seconds",
    "tokens", "tokens_per_second", "critical_during_request",
]


def record_run(record: dict, reports_dir: str | Path = "reports") -> None:
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    enriched = dict(record)
    enriched.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    snapshots = [enriched.get("memory_before", {}), *enriched.get("memory_samples_during", []), enriched.get("memory_after", {})]
    available = [int(item["available_bytes"]) for item in snapshots if item.get("available_bytes") is not None]
    row = {
        "timestamp_utc": enriched["timestamp_utc"],
        "model": enriched.get("model", ""),
        "result": enriched.get("result", ""),
        "guard_state": enriched.get("memory_state_before", ""),
        "ram_total_bytes": snapshots[0].get("total_bytes", "") if snapshots else "",
        "ram_available_before_bytes": snapshots[0].get("available_bytes", "") if snapshots else "",
        "ram_available_during_min_bytes": min(available) if len(available) > 2 else "",
        "ram_available_after_bytes": snapshots[-1].get("available_bytes", "") if snapshots else "",
        "load_seconds": enriched.get("load_seconds", ""),
        "inference_seconds": enriched.get("inference_seconds", ""),
        "tokens": enriched.get("tokens", ""),
        "tokens_per_second": enriched.get("tokens_per_second", ""),
        "critical_during_request": enriched.get("critical_during_request", ""),
    }
    json_path = directory / "memory_report.json"
    csv_path = directory / "memory_history.csv"
    with _LOCK:
        events = {"events": []}
        if json_path.exists():
            try:
                events = json.loads(json_path.read_text(encoding="utf-8"))
                if not isinstance(events, dict) or not isinstance(events.get("events"), list):
                    raise ValueError("Invalid memory report schema")
            except (OSError, json.JSONDecodeError, ValueError):
                raise RuntimeError(f"Refusing to overwrite invalid report: {json_path}")
        events["events"].append(enriched)
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(json_path)
        new_csv = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if new_csv:
                writer.writeheader()
            writer.writerow(row)
