from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from engine.etw_file_test import run_etw_file_test


class FakeGuard:
    def require_admission(self):
        return SimpleNamespace(state="NORMAL")


class FakeCapture:
    preflight_available = True

    def __init__(self, _logs, *, target_file_path):
        self.target_file_path = Path(target_file_path)

    def preflight(self):
        return {"available": self.preflight_available, "reason": "admin required"}

    def start(self):
        return {"started": True}

    def stop(self):
        return {"available": True, "target_file_analysis": {
            "available": True, "physical_file_read_bytes": self.target_file_path.stat().st_size,
        }}


class EtwFileTestTests(unittest.TestCase):
    def test_bounded_file_test_checks_hash_and_removes_only_its_probe_file(self):
        with TemporaryDirectory() as temp:
            reader_calls = []

            def reader(path, expected):
                reader_calls.append(expected)
                data = path.read_bytes()
                import hashlib
                return len(data), hashlib.sha256(data).hexdigest()

            result = run_etw_file_test(
                temp, size_mib=1, capture_factory=FakeCapture,
                unbuffered_reader=reader, memory_guard=FakeGuard(), windows_probe=lambda: True,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(reader_calls, [1024 * 1024])
            self.assertFalse(Path(result["test_file_path"]).exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_preflight_block_does_not_create_probe_file(self):
        with TemporaryDirectory() as temp:
            class DeniedCapture(FakeCapture):
                preflight_available = False

            result = run_etw_file_test(
                temp, size_mib=1, capture_factory=DeniedCapture,
                memory_guard=FakeGuard(), windows_probe=lambda: True,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["test_file_created"])
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_invalid_file_size_is_rejected(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_etw_file_test(
                    temp, size_mib=17, capture_factory=FakeCapture,
                    memory_guard=FakeGuard(), windows_probe=lambda: True,
                )


if __name__ == "__main__":
    unittest.main()
