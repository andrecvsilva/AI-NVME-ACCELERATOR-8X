from pathlib import Path
import tempfile
import unittest

from engine.prefetch_engine import PredictivePrefetchEngine, SequentialAccessPredictor


class PredictorTests(unittest.TestCase):
    def test_requires_stable_positive_stride(self):
        predictor = SequentialAccessPredictor()
        self.assertIsNone(predictor.observe(0))
        self.assertIsNone(predictor.observe(1))
        self.assertEqual(predictor.observe(2), 3)
        self.assertEqual(predictor.observe(3), 4)

    def test_pattern_change_resets_confidence(self):
        predictor = SequentialAccessPredictor()
        for block in (0, 1, 2):
            result = predictor.observe(block)
        self.assertEqual(result, 3)
        self.assertIsNone(predictor.observe(8))
        self.assertIsNone(predictor.observe(10))


class PrefetchEngineTests(unittest.TestCase):
    def test_does_not_schedule_under_memory_pressure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.part"
            path.write_bytes(b"a" * 16384)
            with PredictivePrefetchEngine() as engine:
                self.assertFalse(engine.prefetch(path, 0, 4096, pressure="warning", available_bytes=2**30, reserve_bytes=0))
                self.assertEqual(engine.stats()["scheduled"], 0)

    def test_async_read_returns_exact_requested_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.part"
            content = bytes(range(256)) * 32
            path.write_bytes(content)
            with PredictivePrefetchEngine() as engine:
                self.assertTrue(engine.prefetch(path, 1, 4096, pressure="normal", available_bytes=2**30, reserve_bytes=512 * 2**20))
                result = engine.take(path, 1, 4096, wait_timeout_s=2)
                self.assertEqual(result, content[4096:8192])
                self.assertEqual(engine.stats()["useful_bytes"], 4096)

    def test_rejects_unbounded_block_size(self):
        with PredictivePrefetchEngine() as engine:
            with self.assertRaises(ValueError):
                engine.prefetch("unused", 0, 5 * 1024 * 1024, pressure="normal", available_bytes=2**30, reserve_bytes=0)


if __name__ == "__main__":
    unittest.main()
