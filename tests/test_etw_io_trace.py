from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from engine.etw_io_trace import WprEtwCapture


class WprEtwCaptureTests(unittest.TestCase):
    def test_refuses_capture_without_elevation(self):
        calls = []
        capture = WprEtwCapture("logs", runner=lambda *a, **k: calls.append(a), elevated_probe=lambda: False)
        result = capture.start()
        self.assertFalse(result["started"])
        self.assertIn("elevated", result["reason"])
        self.assertEqual(calls, [])

    def test_start_export_and_stop_uses_owned_paths(self):
        with TemporaryDirectory() as temp:
            calls = []

            def runner(command, **_kwargs):
                calls.append(command)
                if command[:2] == ["wpr.exe", "-status"]:
                    return SimpleNamespace(returncode=0, stdout="WPR is not recording", stderr="")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            capture = WprEtwCapture(temp, runner=runner, elevated_probe=lambda: True)
            started = capture.start()
            self.assertTrue(started["started"])
            self.assertIn("DiskIO", calls[1])
            self.assertIn("FileIO", calls[1])
            stopped = capture.stop()
            self.assertTrue(stopped["available"])
            self.assertTrue(Path(stopped["trace_path"]).parent == Path(temp).resolve())
            self.assertTrue(stopped["events_csv_path"].endswith(".csv"))
            self.assertEqual(calls[2][1], "-stop")
            self.assertEqual(calls[3][0], "tracerpt.exe")
            self.assertIn("XML", calls[3])
            self.assertEqual(calls[4][0], "tracerpt.exe")
            self.assertIn("CSV", calls[4])

    def test_can_skip_large_csv_export(self):
        with TemporaryDirectory() as temp:
            calls = []

            def runner(command, **_kwargs):
                calls.append(command)
                if command[:2] == ["wpr.exe", "-status"]:
                    return SimpleNamespace(returncode=0, stdout="WPR is not recording", stderr="")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            capture = WprEtwCapture(temp, runner=runner, elevated_probe=lambda: True,
                                    export_csv=False)
            self.assertTrue(capture.start()["started"])
            stopped = capture.stop()
            self.assertTrue(stopped["available"])
            self.assertIsNone(stopped["events_csv_path"])
            self.assertIsNone(stopped["summary_path"])
            self.assertEqual(len(calls), 4)
            self.assertIn("XML", calls[3])

    def test_refuses_to_take_over_an_existing_trace(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="WPR is recording", stderr="")

        capture = WprEtwCapture("logs", runner=runner, elevated_probe=lambda: True)
        result = capture.start()
        self.assertFalse(result["started"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
