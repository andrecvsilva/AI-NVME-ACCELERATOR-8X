import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_guard.logger import record_run
from memory_guard.monitor import GIB, MemorySnapshot
from memory_guard.ollama_control import OllamaStatus, guarded_chat, query_ollama_ps
from memory_guard.protection import MemoryGuard, classify_memory
from memory_guard.thresholds import GuardThresholds


def snapshot(available_gib=8, load=50, commit=50):
    total = 16 * GIB
    available = int(available_gib * GIB)
    return MemorySnapshot(
        "2026-11-06T00:00:00+00:00", total, available, total - available,
        load, 40 * GIB, int(40 * GIB * (100 - commit) / 100),
        int(40 * GIB * commit / 100), commit,
    )


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        time.sleep(0.32)
        return b'{"model":"qwen3:4b-instruct","message":{"content":"ok"},"done_reason":"stop","eval_count":2,"eval_duration":1000000000,"load_duration":2000000000,"total_duration":3000000000}'


class MemoryGuardPackageTests(unittest.TestCase):
    def test_normal_warning_critical_states(self):
        thresholds = GuardThresholds()
        self.assertEqual(classify_memory(snapshot(8), thresholds), "NORMAL")
        self.assertEqual(classify_memory(snapshot(1), thresholds), "WARNING")
        self.assertEqual(classify_memory(snapshot(0.25), thresholds), "CRITICAL")

    def test_ollama_without_loaded_models(self):
        with patch("memory_guard.ollama_control.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "NAME ID SIZE PROCESSOR CONTEXT UNTIL\n"
            status = query_ollama_ps()
        self.assertTrue(status.online)
        self.assertEqual(status.models, ())

    def test_loaded_model_blocks_new_request(self):
        guard = MemoryGuard(snapshot_provider=lambda: snapshot())
        with self.assertRaisesRegex(RuntimeError, "already has a model loaded"):
            guarded_chat(
                "m", "p", guard=guard,
                status_provider=lambda: OllamaStatus(True, ({"NAME": "other"},)),
            )

    def test_ollama_communication_failure(self):
        with patch("memory_guard.ollama_control.subprocess.run", side_effect=OSError("missing")):
            status = query_ollama_ps()
        self.assertFalse(status.online)
        self.assertIn("missing", status.error)
        guard = MemoryGuard(snapshot_provider=lambda: snapshot())
        with self.assertRaises(ConnectionError):
            guarded_chat("m", "p", guard=guard, status_provider=lambda: status)

    def test_warning_and_critical_block_before_ollama(self):
        for current in (snapshot(1), snapshot(0.25)):
            guard = MemoryGuard(snapshot_provider=lambda current=current: current)
            with self.assertRaises(MemoryError):
                guarded_chat("m", "p", guard=guard, status_provider=lambda: self.fail("must not query ollama"))

    def test_drop_during_inference_is_sampled_and_recorded(self):
        sequence = iter([snapshot(0.25), snapshot(0.25)])
        guard = MemoryGuard(snapshot_provider=lambda: snapshot(8))
        with patch("memory_guard.ollama_control.urllib.request.urlopen", return_value=Response()):
            result = guarded_chat(
                "m", "p", guard=guard,
                status_provider=lambda: OllamaStatus(True, ()),
                snapshot_provider=lambda: next(sequence),
                sample_interval_seconds=0.25,
            )
        self.assertTrue(result["critical_during_request"])
        self.assertEqual(result["memory_state_after"], "NORMAL")
        self.assertEqual(result["tokens_per_second"], 2.0)

    def test_json_and_csv_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_run({
                "model": "test", "result": "completed", "memory_state_before": "NORMAL",
                "memory_before": snapshot().to_dict(), "memory_after": snapshot().to_dict(),
                "tokens": 1,
            }, tmp)
            self.assertTrue((Path(tmp) / "memory_report.json").is_file())
            self.assertTrue((Path(tmp) / "memory_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
