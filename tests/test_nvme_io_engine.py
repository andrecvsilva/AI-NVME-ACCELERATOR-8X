from pathlib import Path
import tempfile
import unittest

from engine.nvme_io_engine import benchmark_file


class DiskBenchmarkTests(unittest.TestCase):
    def test_reads_existing_file_and_respects_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(bytes(range(256)) * 64)
            result = benchmark_file(path, max_mib=1, block_kib=4, random_ops=8)
        self.assertEqual(result["sequential"]["bytes_read"], 16_384)
        self.assertEqual(result["random_4k"]["operations"], 4)
        self.assertTrue(any("Read-only" in item for item in result["limitations"]))

    def test_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            benchmark_file("Z:/this-file-should-not-exist.gguf")

    def test_rejects_unbounded_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(b"x" * 8192)
            with self.assertRaises(ValueError):
                benchmark_file(path, max_mib=1025)
            with self.assertRaises(ValueError):
                benchmark_file(path, random_ops=4097)


if __name__ == "__main__":
    unittest.main()
