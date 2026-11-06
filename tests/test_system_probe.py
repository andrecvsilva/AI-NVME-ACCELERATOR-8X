import unittest
from unittest.mock import patch

from engine.system_probe import _to_list, probe_windows


class SystemProbeTests(unittest.TestCase):
    def test_to_list_normalizes_powershell_singletons(self):
        self.assertEqual(_to_list(None), [])
        self.assertEqual(_to_list({"Name": "disk"}), [{"Name": "disk"}])
        self.assertEqual(_to_list([1, 2]), [1, 2])

    @patch("engine.system_probe.os.name", "posix")
    def test_probe_rejects_non_windows(self):
        # Avoid starting PowerShell on non-Windows systems.
        with self.assertRaises(RuntimeError):
            probe_windows()


if __name__ == "__main__":
    unittest.main()
