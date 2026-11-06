import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_guard.monitor import GIB
from memory_guard.process_monitor import (
    OllamaProcessMetrics,
    ProcessIdentity,
    analyze_ollama_state,
    collect_ollama_processes,
    record_process_snapshot,
)


def process(pid=101, name="ollama.exe", private_mib=100, status="ok"):
    return OllamaProcessMetrics(
        pid=pid,
        name=name,
        working_set_bytes=private_mib * 1024**2,
        private_working_set_bytes=private_mib * 1024**2 // 2,
        private_bytes=private_mib * 1024**2,
        virtual_bytes=private_mib * 1024**2 * 3,
        thread_count=8,
        uptime_seconds=180.0,
        status=status,
    ).to_dict()


def report(*processes):
    return {"status": "active", "error": None, "processes": list(processes)}


class ProcessMonitorTests(unittest.TestCase):
    def test_online_without_model_low_consumption(self):
        result = analyze_ollama_state(
            online=True, loaded_model_count=0, process_report=report(process()), guard_state="NORMAL"
        )
        self.assertEqual(result["classification"], "ONLINE_NO_MODEL_LOW_MEMORY")

    def test_online_with_loaded_model(self):
        result = analyze_ollama_state(
            online=True, loaded_model_count=1, process_report=report(process()), guard_state="WARNING"
        )
        self.assertEqual(result["classification"], "ONLINE_MODEL_LOADED")
        self.assertEqual(result["memory_guard_state"], "WARNING")

    def test_ollama_offline(self):
        result = analyze_ollama_state(
            online=False, loaded_model_count=0, process_report=report(), guard_state="NORMAL"
        )
        self.assertEqual(result["classification"], "OFFLINE")

    def test_high_private_memory_is_only_a_suspicion(self):
        result = analyze_ollama_state(
            online=True, loaded_model_count=0,
            process_report=report(process(private_mib=2048)), guard_state="WARNING"
        )
        self.assertEqual(result["classification"], "MEMORY_RESIDUAL_SUSPECTED")
        self.assertTrue(result["residual_suspected_is_hypothesis"])

    def test_low_private_memory_is_not_suspected(self):
        result = analyze_ollama_state(
            online=True, loaded_model_count=0,
            process_report=report(process(private_mib=32)), guard_state="NORMAL"
        )
        self.assertEqual(result["classification"], "ONLINE_NO_MODEL_LOW_MEMORY")

    def test_process_terminated_during_collection(self):
        identity = ProcessIdentity(8, "ollama.exe", 2)
        collected = collect_ollama_processes(
            enumerator=lambda: [identity], reader=lambda _identity: (_ for _ in ()).throw(ProcessLookupError("gone"))
        )
        self.assertEqual(collected["processes"][0]["status"], "exited_during_collection")
        analyzed = analyze_ollama_state(online=True, loaded_model_count=0, process_report=collected, guard_state="NORMAL")
        self.assertEqual(analyzed["classification"], "PROCESS_EXITED_DURING_COLLECTION")

    def test_permission_error(self):
        identity = ProcessIdentity(8, "ollama app.exe", 2)
        collected = collect_ollama_processes(
            enumerator=lambda: [identity], reader=lambda _identity: (_ for _ in ()).throw(PermissionError("denied"))
        )
        self.assertEqual(collected["processes"][0]["status"], "permission_denied")
        analyzed = analyze_ollama_state(online=True, loaded_model_count=0, process_report=collected, guard_state="NORMAL")
        self.assertEqual(analyzed["classification"], "PROCESS_INFO_UNAVAILABLE")

    def test_no_ollama_processes(self):
        result = analyze_ollama_state(online=True, loaded_model_count=0, process_report=report(), guard_state="NORMAL")
        self.assertEqual(result["classification"], "ONLINE_NO_MODEL_NO_PROCESS")

    def test_metrics_unavailable(self):
        identity = ProcessIdentity(9, "ollama.exe", 1)
        unavailable = OllamaProcessMetrics(9, "ollama.exe", thread_count=1, status="unavailable")
        collected = collect_ollama_processes(enumerator=lambda: [identity], reader=lambda _identity: unavailable)
        self.assertIsNone(collected["processes"][0]["private_mib"])
        result = analyze_ollama_state(online=True, loaded_model_count=0, process_report=collected, guard_state="CRITICAL")
        self.assertEqual(result["classification"], "PROCESS_INFO_UNAVAILABLE")
        self.assertEqual(result["memory_guard_state"], "CRITICAL")

    def test_json_and_csv_records_append_without_touching_performance_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            performance = logs / "performance.json"
            performance.write_text('{"events": [{"keep": true}]}', encoding="utf-8")
            sample = {
                "timestamp_utc": "2026-11-06T00:00:00+00:00",
                "memory_guard_state": "WARNING",
                "ram_available_bytes": 3 * GIB,
                "ram_load_percent": 80,
                "ollama_online": True,
                "loaded_model_count": 0,
                "classification": "MEMORY_RESIDUAL_SUSPECTED",
                "processes": [process(private_mib=1700)],
            }
            record_process_snapshot(sample, logs)
            record_process_snapshot(sample, logs)
            saved = json.loads((logs / "process_monitor.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 2)
            self.assertEqual(performance.read_text(encoding="utf-8"), '{"events": [{"keep": true}]}')
            with (logs / "process_monitor.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pid"], "101")
            self.assertEqual(rows[0]["classification"], "MEMORY_RESIDUAL_SUSPECTED")

    def test_empty_process_json_log_appends_valid_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_process_snapshot({"processes": []}, tmp)
            record_process_snapshot({"processes": []}, tmp)
            content = json.loads((Path(tmp) / "process_monitor.json").read_text(encoding="utf-8"))
            self.assertEqual(len(content), 2)

    def test_memory_guard_states_are_preserved_in_diagnosis(self):
        for state in ("NORMAL", "WARNING", "CRITICAL"):
            result = analyze_ollama_state(
                online=True, loaded_model_count=0, process_report=report(process()), guard_state=state
            )
            self.assertEqual(result["memory_guard_state"], state)

    def test_units_are_explicit_and_virtual_is_not_private_commit(self):
        values = process(private_mib=64)
        self.assertEqual(values["working_set_mib"], 64)
        self.assertEqual(values["private_mib"], 64)
        self.assertEqual(values["virtual_mib"], 192)
        self.assertNotEqual(values["virtual_bytes"], values["private_bytes"])


if __name__ == "__main__":
    unittest.main()
