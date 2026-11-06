"""Optional short WPR DiskIO/FileIO capture for later ETW analysis."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from engine.etw_io_analyzer import analyze_tracerpt_xml


def _is_elevated() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


class WprEtwCapture:
    """Capture whole-system disk and file I/O while an explicit inference runs.

    The trace can include unrelated process/file paths. It stays in the project
    logs directory and is never uploaded or automatically deleted.
    """

    def __init__(self, logs_dir: str | Path, *, gguf_path: str | Path | None = None,
                 target_file_path: str | Path | None = None,
                 export_csv: bool = True,
                 runner: Callable = subprocess.run,
                 elevated_probe: Callable[[], bool] = _is_elevated):
        self.logs_dir = Path(logs_dir).resolve()
        self.runner = runner
        self.elevated_probe = elevated_probe
        requested_target = target_file_path or gguf_path
        self.target_file_path = Path(requested_target).resolve() if requested_target else None
        self.export_csv = export_csv
        self._started = False
        self._trace_path: Path | None = None
        self._start_error: str | None = None

    def _run(self, command: list[str], *, timeout: int = 30):
        return self.runner(command, check=False, capture_output=True, text=True, timeout=timeout)

    def preflight(self) -> dict:
        """Check WPR prerequisites without starting a trace or loading a model."""
        if os.name != "nt":
            self._start_error = "WPR ETW capture is available only on Windows."
            return {"available": False, "started": False, "reason": self._start_error}
        if not self.elevated_probe():
            self._start_error = "WPR requires an elevated PowerShell session for kernel DiskIO/FileIO capture."
            return {"available": False, "started": False, "reason": self._start_error}

        status = self._run(["wpr.exe", "-status"])
        status_text = f"{status.stdout}\n{status.stderr}".casefold()
        if status.returncode != 0 or not any(phrase in status_text for phrase in (
            "not recording", "não está gravando", "nao esta gravando", "not currently recording"
        )):
            self._start_error = "WPR is active or its status is uncertain; refusing to take over another trace."
            return {"available": False, "started": False, "reason": self._start_error}
        self._start_error = None
        return {"available": True, "started": False, "reason": None,
                "status": "WPR is idle and capture privileges are available."}

    def start(self) -> dict:
        check = self.preflight()
        if not check.get("available"):
            return check

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._trace_path = self.logs_dir / f"model_io_{run_id}.etl"
        started = self._run(["wpr.exe", "-start", "DiskIO", "-start", "FileIO", "-filemode"])
        if started.returncode != 0:
            self._start_error = (started.stderr or started.stdout or "WPR failed to start.").strip()
            return {"available": False, "started": False, "reason": self._start_error}
        self._started = True
        return {
            "available": True,
            "started": True,
            "profiles": ["DiskIO", "FileIO"],
            "storage_mode": "filemode",
            "trace_path": str(self._trace_path),
            "scope": "Whole-system ETW; may contain unrelated process and file activity.",
        }

    def stop(self) -> dict:
        if not self._started or self._trace_path is None:
            return {"available": False, "started": False, "reason": self._start_error or "No trace was started."}
        stopped = self._run(["wpr.exe", "-stop", str(self._trace_path)], timeout=60)
        self._started = False
        if stopped.returncode != 0:
            return {"available": False, "started": True, "trace_path": str(self._trace_path),
                    "reason": (stopped.stderr or stopped.stdout or "WPR failed to stop the trace.").strip()}

        csv_path = self._trace_path.with_suffix(".csv")
        xml_path = self._trace_path.with_suffix(".xml")
        summary_path = self._trace_path.with_suffix(".summary.txt")
        exported_xml = self._run([
            "tracerpt.exe", str(self._trace_path), "-o", str(xml_path), "-of", "XML", "-y",
        ], timeout=120)
        exported_csv = None
        if self.export_csv:
            exported_csv = self._run([
                "tracerpt.exe", str(self._trace_path), "-o", str(csv_path), "-of", "CSV", "-y",
                "-summary", str(summary_path),
            ], timeout=120)
        analysis = None
        analysis_error = None
        if exported_xml.returncode == 0 and self.target_file_path is not None:
            try:
                analysis = analyze_tracerpt_xml(xml_path, self.target_file_path)
            except Exception as exc:
                analysis_error = f"{type(exc).__name__}: {exc}"
        return {
            "available": exported_xml.returncode == 0 and (exported_csv is None or exported_csv.returncode == 0),
            "started": True,
            "trace_path": str(self._trace_path),
            "events_csv_path": str(csv_path) if exported_csv is not None and exported_csv.returncode == 0 else None,
            "events_xml_path": str(xml_path) if exported_xml.returncode == 0 else None,
            "summary_path": str(summary_path) if exported_csv is not None and exported_csv.returncode == 0 else None,
            "reason": None if exported_xml.returncode == 0 and (exported_csv is None or exported_csv.returncode == 0) else (
                exported_xml.stderr or exported_xml.stdout
                or (exported_csv.stderr or exported_csv.stdout if exported_csv is not None else "")
                or "tracerpt failed to export the ETL."
            ).strip(),
            "gguf_analysis": analysis,
            "target_file_analysis": analysis,
            "gguf_analysis_error": analysis_error,
            "interpretation": (
                "XML preserves raw ETW events; CSV is optional. gguf_analysis separates filesystem read requests from DiskIO "
                "completion events attributed via FileObject; inspect the trace if either export or correlation is unavailable."
            ),
        }
