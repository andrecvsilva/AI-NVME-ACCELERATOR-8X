"""Append-only NVMe I/O logs and latest-run text report."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

_FIELDS = ("timestamp_utc", "status", "drive", "device_model", "bus_type_reported", "memory_guard_state",
          "operation", "file_size_bytes", "block_bytes", "operations", "bytes_read", "elapsed_seconds",
          "throughput_mb_s", "throughput_mib_s", "iops", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
          "windows_cache_state", "error", "result_json")


class NvmeIoReporter:
    def __init__(self, logs_dir: str | Path, reports_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.reports_dir = Path(reports_dir)
        self.json_path = self.logs_dir / "nvme_io.json"
        self.csv_path = self.logs_dir / "nvme_io.csv"
        self.report_path = self.reports_dir / "nvme_io_report.txt"
        self._lock = threading.Lock()

    def append(self, event: dict) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock:
            if not self.json_path.exists():
                self.json_path.write_text("[\n" + json.dumps(record, ensure_ascii=False) + "\n]\n", encoding="utf-8")
            else:
                with self.json_path.open("r+b") as handle:
                    handle.seek(0, os.SEEK_END)
                    cursor = handle.tell() - 1
                    while cursor >= 0:
                        handle.seek(cursor)
                        char = handle.read(1)
                        if char not in b" \t\r\n":
                            break
                        cursor -= 1
                    if cursor < 0 or char != b"]":
                        raise RuntimeError(f"Refusing to append to malformed NVMe I/O log: {self.json_path}")
                    handle.seek(0)
                    if handle.read(1) != b"[":
                        raise RuntimeError(f"Refusing to append to non-array NVMe I/O log: {self.json_path}")
                    previous = cursor - 1
                    while previous >= 0:
                        handle.seek(previous)
                        prior = handle.read(1)
                        if prior not in b" \t\r\n":
                            break
                        previous -= 1
                    handle.seek(cursor)
                    handle.truncate()
                    handle.write((b"\n" if prior == b"[" else b",\n") +
                                 json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n]\n")
            row = {field: record.get(field, "") for field in _FIELDS}
            row["result_json"] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            new_csv = not self.csv_path.exists()
            with self.csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_FIELDS, extrasaction="ignore")
                if new_csv:
                    writer.writeheader()
                writer.writerow(row)

    def write_report(self, event: dict) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        lines = ["AI NVMe Accelerator 8X — NVMe I/O Engine 1.0", "Relatório da última execução",
                 f"Data UTC: {event.get('timestamp_utc', 'indisponível')}", f"Estado: {event.get('status', 'unknown')}",
                 f"Memory Guard: {event.get('memory_guard_state', 'indisponível')}",
                 f"Dispositivo: {event.get('device', {}).get('device_model') or 'indisponível'}",
                 f"Unidade/barramento: {event.get('device', {}).get('drive')} / {event.get('device', {}).get('bus_type_reported')}",
                 "Cache do Windows: primeira passagem não é garantidamente fria; a passagem repetida pode ser atendida pelo cache.",
                 "Medição de armazenamento separada de inferência; não representa tokens/s.", ""]
        for file_result in event.get("files", []):
            lines.append(f"Arquivo {file_result.get('file_size_bytes', 0) / (1024**2):.2f} MiB — {file_result.get('status')}")
            for label in ("sequential_first", "sequential_repeat"):
                measure = file_result.get(label) or {}
                lines.append(f"  {label}: bytes={measure.get('bytes_read')} tempo_s={measure.get('elapsed_seconds')} "
                             f"MB/s={measure.get('throughput_mb_s')} MiB/s={measure.get('throughput_mib_s')} cache={measure.get('windows_cache_state')}")
            for key, measure in file_result.get("random", {}).items():
                lines.append(f"  random_{key}: ops={measure.get('operations')} IOPS={measure.get('iops')} "
                             f"latência média/P50/P95/P99 ms={measure.get('latency_mean_ms')}/"
                             f"{measure.get('latency_p50_ms')}/{measure.get('latency_p95_ms')}/{measure.get('latency_p99_ms')}")
        if event.get("error"):
            lines.extend(["", f"Erro/interrupção: {event['error']}"])
        lines.extend(["", "Limites: leitura em blocos; arquivos temporários próprios; nenhuma limpeza forçada do cache do Windows.",
                      "Nenhuma inferência Ollama é iniciada por este mecanismo.", ""])
        fd, temp_name = tempfile.mkstemp(prefix=".nvme-io-report-", suffix=".tmp", dir=self.reports_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(lines))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.report_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)


def latest_csv_event(csv_path: str | Path) -> dict | None:
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        if position <= 0:
            return None
        handle.seek(position - 1)
        if handle.read(1) in {b"\n", b"\r"}:
            position -= 1
        while position > 0:
            position -= 1
            handle.seek(position)
            if handle.read(1) == b"\n":
                break
        handle.seek(position + 1 if position > 0 else 0)
        raw = handle.readline().decode("utf-8-sig").strip()
    if not raw:
        return None
    try:
        # CSV values contain no literal newlines; use header plus last row only.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
        return next(csv.DictReader([header, raw]))
    except (csv.Error, StopIteration, UnicodeError, OSError):
        return None
