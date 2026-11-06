from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.inference_engine import InferenceEngine, _capture_native_load_output


def memory_sample(pressure="normal"):
    available = {"normal": 8 * 1024**3, "warning": 3 * 1024**3,
                 "critical": 256 * 1024**2}[pressure]
    return {
        "guard": {"pressure": pressure},
        "physical_memory": {
            "total_bytes": 16 * 1024**3,
            "available_bytes": available,
            "load_percent": 100 - int(available / (16 * 1024**3) * 100),
        },
    }


class FakeLlama:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False

    def create_chat_completion(self, **kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 2},
        }

    def close(self):
        self.closed = True


class FakeDiskObserver:
    def __init__(self, **_kwargs):
        pass

    def start(self):
        return self

    def stop(self):
        return {"available": True, "estimated_read_mib": 1.5,
                "per_file_attribution_available": False}


class FakeEtwPreflight:
    def __init__(self, *_args, **_kwargs):
        pass

    def preflight(self):
        return {"available": False, "started": False, "reason": "admin required"}


class WarningRecoveredObserver:
    def __init__(self, **_kwargs):
        pass

    def start(self, **_kwargs):
        return self

    def summary(self):
        return {"peak_pressure": "normal"}

    def stop(self, **_kwargs):
        return {"peak_pressure": "warning", "last_pressure": "normal", "sample_count": 2}


class InferenceEngineTests(unittest.TestCase):
    def test_native_log_redirection_failure_does_not_block_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "llama.log"
            with patch("engine.inference_engine.os.dup", side_effect=OSError(6, "invalid handle")):
                with _capture_native_load_output(log):
                    reached_loader = True
            self.assertTrue(reached_loader)

    def test_load_uses_mmap_cpu_and_low_context_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"not a real model; factory is mocked")
            engine = InferenceEngine(
                model, memory_probe=lambda **_: memory_sample("normal"),
                disk_io_observer_factory=FakeDiskObserver,
            )
            created = []
            def factory(**kwargs):
                fake = FakeLlama(**kwargs)
                created.append(fake)
                return fake
            metrics = engine.load_model(llm_factory=factory)
            self.assertTrue(metrics["mmap_enabled"])
            self.assertEqual(created[0].options["n_gpu_layers"], 0)
            self.assertEqual(created[0].options["n_ctx"], 512)
            self.assertEqual(metrics["memory_pressure_before_load"], "normal")
            generated = engine.generate("hi", max_tokens=4)
            self.assertEqual(generated["completion_tokens"], 2)
            self.assertEqual(generated["physical_disk_io_during_generation"]["estimated_read_mib"], 1.5)
            self.assertEqual(generated["generation_io_memory_correlation"]["completion_tokens"], 2)
            self.assertEqual(generated["generation_io_memory_correlation"]["physical_disk_io"]["estimated_read_mib"], 1.5)
            engine.unload_model()
            self.assertTrue(created[0].closed)

    def test_critical_memory_guard_stops_before_model_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            engine = InferenceEngine(model, memory_probe=lambda **_: memory_sample("critical"))
            called = []
            with self.assertRaises(RuntimeError):
                engine.load_model(llm_factory=lambda **_: called.append(True))
            self.assertEqual(called, [])

    def test_etw_preflight_blocks_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            engine = InferenceEngine(
                model, memory_probe=lambda **_: memory_sample("normal"),
                capture_etw=True, etw_capture_factory=FakeEtwPreflight,
            )
            called = []
            with self.assertRaisesRegex(RuntimeError, "ETW prerequisites"):
                engine.load_model(llm_factory=lambda **_: called.append(True))
            self.assertEqual(called, [])

    def test_transient_warning_during_load_is_allowed_only_after_recovery_to_normal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            engine = InferenceEngine(
                model, memory_probe=lambda **_: memory_sample("normal"),
                model_memory_observer_factory=WarningRecoveredObserver,
            )
            metrics = engine.load_model(llm_factory=FakeLlama)
            self.assertEqual(metrics["memory_observation_during_load"]["peak_pressure"], "warning")
            self.assertEqual(metrics["memory_pressure_after_load"], "normal")
            engine.unload_model()

    def test_mmap_low_ram_profile_uses_minimum_context_batch_and_threads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            snapshot = memory_sample("normal")
            snapshot["physical_memory"]["available_bytes"] = int(3.1 * 1024**3)
            engine = InferenceEngine(
                model, n_ctx=512, n_batch=16, n_threads=4,
                memory_probe=lambda **_: snapshot,
            )
            created = []
            metrics = engine.load_model(llm_factory=lambda **kwargs: created.append(FakeLlama(**kwargs)) or created[-1])
            self.assertEqual(created[0].options["n_ctx"], 128)
            self.assertEqual(created[0].options["n_batch"], 4)
            self.assertEqual(created[0].options["n_threads"], 1)
            self.assertTrue(metrics["mmap_enabled"])
            engine.unload_model()

    def test_warning_memory_guard_blocks_before_model_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            engine = InferenceEngine(model, memory_probe=lambda **_: memory_sample("warning"))
            called = []
            with self.assertRaisesRegex(RuntimeError, "Memory Guard"):
                engine.load_model(llm_factory=lambda **_: called.append(True))
            self.assertEqual(called, [])

    def test_generation_validates_token_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fake.gguf"
            model.write_bytes(b"fake")
            engine = InferenceEngine(model, memory_probe=lambda **_: memory_sample())
            engine.load_model(llm_factory=FakeLlama)
            with self.assertRaises(ValueError):
                engine.generate("hi", max_tokens=512)
            engine.unload_model()


if __name__ == "__main__":
    unittest.main()
