from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.nvme_io_engine import _unbuffered_sequential_read, benchmark_block_sweep


class BlockSweepTests(unittest.TestCase):
    def test_sweep_respects_per_size_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(b"x" * (2 * 1024 * 1024))
            result = benchmark_block_sweep(path, mib_per_block=1, block_sizes_kib=(4, 64))
        self.assertEqual(result["maximum_total_bytes_read"], 2 * 1024 * 1024)
        self.assertEqual([item["block_kib"] for item in result["sequential_block_sweep"]], [4, 64])
        self.assertTrue(all(item["bytes_read"] == 1024 * 1024 for item in result["sequential_block_sweep"]))

    def test_sweep_rejects_oversized_total_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(b"x" * 4096)
            with self.assertRaises(ValueError):
                benchmark_block_sweep(path, mib_per_block=65)

    @patch("engine.nvme_io_engine.os.name", "posix")
    def test_unbuffered_mode_is_windows_only(self):
        with self.assertRaises(RuntimeError):
            _unbuffered_sequential_read(Path("unused"), 4096, 4096)


if __name__ == "__main__":
    unittest.main()
