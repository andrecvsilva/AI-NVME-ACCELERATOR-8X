import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from memory_guard.monitor import MemorySnapshot
from memory_guard.protection import GuardDecision
from adaptive_cache.memory_policy import CacheMemoryDecision
from nvme_io.benchmark_manager import BenchmarkBusyError, MIB, NvmeBenchmarkManager
from nvme_io.config import NvmeIoConfig
from nvme_io.device_info import get_device_info
from nvme_io.metrics import percentile, summarize
from nvme_io.random_benchmark import random_read
from nvme_io.report import NvmeIoReporter, latest_csv_event
from nvme_io.sequential_benchmark import sequential_read


def snapshot():
    return MemorySnapshot("2026-11-06T00:00:00+00:00", 16 * MIB, 12 * MIB, 4 * MIB, 25,
                          32 * MIB, 24 * MIB, 8 * MIB, 25.0)


class FakeGuard:
    def __init__(self, states=None):
        self.states = list(states or ["NORMAL"])
        self.calls = 0

    def inspect(self):
        index = min(self.calls, len(self.states) - 1)
        state = self.states[index]
        self.calls += 1
        if state == "UNAVAILABLE":
            raise RuntimeError("simulated memory API failure")
        return GuardDecision(state, state == "NORMAL", "simulated", snapshot())


class FakePolicy:
    def __init__(self, state="NORMAL", available=True):
        self.state = state
        self.available = available
        self.calls = 0

    def evaluate(self):
        self.calls += 1
        return CacheMemoryDecision(self.available, self.state, 1024, 12 * MIB, MIB, 0,
                                   self.state == "NORMAL", self.state == "NORMAL", self.state == "NORMAL",
                                   False, "simulated policy")


def config(**overrides):
    values = dict(drive="C:", test_sizes_mib=(1,), sequential_block_kib=4,
                  random_operations=4, random_block_kib=(4, 64), warning_max_file_mib=1,
                  warning_random_operations=2, maximum_total_temp_mib=1, minimum_free_mib=0,
                  memory_poll_seconds=60, temp_directory="benchmark/nvme_io_runs")
    values.update(overrides)
    return NvmeIoConfig(**values)


class MetricsTests(unittest.TestCase):
    def test_throughput_iops_and_average_latency_are_calculated(self):
        result = summarize(bytes_read=2_000_000, elapsed_seconds=2, operations=10, latencies_ms=[1, 3])
        self.assertEqual(result["throughput_mb_s"], 1.0)
        self.assertGreater(result["throughput_mib_s"], 0)
        self.assertEqual(result["iops"], 5.0)
        self.assertEqual(result["latency_mean_ms"], 2.0)

    def test_nearest_rank_percentiles(self):
        values = [9, 1, 8, 2, 7, 3, 6, 4, 5, 10]
        self.assertEqual(percentile(values, 0.50), 5)
        self.assertEqual(percentile(values, 0.95), 10)
        self.assertEqual(percentile(values, 0.99), 10)

    def test_empty_latency_samples_are_unavailable(self):
        result = summarize(bytes_read=0, elapsed_seconds=0, operations=0)
        self.assertIsNone(result["throughput_mb_s"])
        self.assertIsNone(result["latency_p99_ms"])

    def test_percentile_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            percentile([1], 1.1)


class FileBenchmarkTests(unittest.TestCase):
    def test_sequential_reads_in_chunks_and_reports_actual_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"a" * 16384)
            result = sequential_read(path, block_bytes=4096)
        self.assertTrue(result["complete"])
        self.assertEqual(result["bytes_read"], 16384)
        self.assertEqual(result["block_bytes"], 4096)

    def test_sequential_block_below_minimum_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"x" * 4096)
            with self.assertRaises(ValueError):
                sequential_read(path, block_bytes=1024)

    def test_random_4k_and_64k_reads_return_latency_percentiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(bytes(range(256)) * 1024)
            for size in (4096, 65536):
                result = random_read(path, block_bytes=size, operations=8, seed=8)
                self.assertEqual(result["operations"], 8)
                self.assertIsNotNone(result["latency_p50_ms"])
                self.assertIsNotNone(result["latency_p95_ms"])
                self.assertIsNotNone(result["latency_p99_ms"])

    def test_random_read_rejects_non_supported_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"x" * 8192)
            with self.assertRaises(ValueError):
                random_read(path, block_bytes=8192, operations=1)

    def test_config_rejects_total_write_over_cap(self):
        with self.assertRaises(ValueError):
            config(test_sizes_mib=(1, 2), maximum_total_temp_mib=2)


class DeviceInfoTests(unittest.TestCase):
    def test_info_is_read_only_and_raid_does_not_claim_non_nvme(self):
        completed = SimpleNamespace(returncode=0, stdout=json.dumps({
            "Volume": {"FileSystem": "NTFS", "Size": 1000, "SizeRemaining": 500},
            "Disk": {"Number": 0, "FriendlyName": "Intel SSD", "BusType": "RAID", "Size": 1000},
            "PhysicalDisks": []}), stderr="")
        with unittest.mock.patch("nvme_io.device_info.os.name", "nt"):
            result = get_device_info("c:", runner=Mock(return_value=completed))
        self.assertEqual(result["filesystem"], "NTFS")
        self.assertFalse(result["nvme_confirmed_by_windows"])
        self.assertIn("cannot be inferred", result["nvme_status_note"])


class ManagerTests(unittest.TestCase):
    def manager(self, root, *, state="NORMAL", available=True, guard=None, config_value=None,
                device_provider=None, disk_free=100 * MIB, file_writer=None,
                sequential_runner=None, random_runner=None):
        return NvmeBenchmarkManager(
            config_value or config(), project_root=root, memory_guard=guard or FakeGuard(),
            adaptive_policy=FakePolicy(state, available),
            device_provider=device_provider or (lambda drive: {"drive": drive, "device_model": "simulated",
                                                               "bus_type_reported": "NVMe", "free_bytes": disk_free}),
            disk_usage=lambda _path: SimpleNamespace(free=disk_free), file_writer=file_writer,
            sequential_runner=sequential_runner or sequential_read,
            random_runner=random_runner or random_read,
        )

    def test_small_synthetic_run_records_first_repeat_and_random_results_then_cleans_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.manager(root)
            result = manager.run_benchmark()
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["inference_started"])
            self.assertEqual(len(result["files"]), 1)
            self.assertIn("sequential_first", result["files"][0])
            self.assertIn("sequential_repeat", result["files"][0])
            self.assertEqual(len(result["files"][0]["random"]), 2)
            self.assertFalse(list((root / "benchmark" / "nvme_io_runs").glob("run-*")))
            self.assertTrue((root / "logs" / "nvme_io.json").exists())
            self.assertTrue((root / "logs" / "nvme_io.csv").exists())
            self.assertTrue((root / "reports" / "nvme_io_report.txt").exists())

    def test_critical_blocks_before_device_query_or_file_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = Mock(side_effect=AssertionError("device query should not run"))
            result = self.manager(Path(tmp), state="CRITICAL", device_provider=provider).run_benchmark()
            self.assertEqual(result["status"], "blocked")
            provider.assert_not_called()
            self.assertFalse((Path(tmp) / "benchmark" / "nvme_io_runs").exists())

    def test_unavailable_policy_blocks_before_file_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.manager(Path(tmp), state="UNAVAILABLE", available=False).run_benchmark()
            self.assertEqual(result["status"], "blocked")
            self.assertFalse((Path(tmp) / "benchmark" / "nvme_io_runs").exists())

    def test_warning_reduces_size_and_random_operation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(test_sizes_mib=(1, 2), maximum_total_temp_mib=3, warning_max_file_mib=1,
                         random_operations=4, warning_random_operations=1)
            result = self.manager(Path(tmp), state="WARNING", config_value=cfg).run_benchmark()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["preflight"]["selected_file_sizes_mib"], [1])
            self.assertTrue(all(item["operations"] <= 1 for item in result["files"][0]["random"].values()))

    def test_critical_memory_transition_stops_additional_io_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = FakeGuard(["NORMAL", "NORMAL", "CRITICAL"])
            sequence = Mock(side_effect=AssertionError("sequential read must not start after CRITICAL"))
            result = self.manager(Path(tmp), guard=guard, sequential_runner=sequence).run_benchmark()
            self.assertEqual(result["status"], "blocked")
            self.assertIn("CRITICAL", result["error"])
            sequence.assert_not_called()
            self.assertEqual(result["files"][0]["status"], "incomplete")
            self.assertTrue((Path(tmp) / "reports" / "nvme_io_report.txt").exists())

    def test_short_sequential_read_is_reported_incomplete_and_stops_later_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = {"bytes_read": 0, "complete": False, "block_bytes": 4096,
                     "elapsed_seconds": 0.0, "throughput_mb_s": None}
            sequence = Mock(return_value=short)
            random = Mock(side_effect=AssertionError("random phases must not start after short read"))
            result = self.manager(Path(tmp), sequential_runner=sequence, random_runner=random).run_benchmark()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["files"][0]["status"], "incomplete")
            self.assertIn("short read", result["files"][0]["error"])
            sequence.assert_called_once()
            random.assert_not_called()

    def test_memory_guard_is_rechecked_between_file_creation_and_first_read_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = FakeGuard(["NORMAL", "NORMAL", "NORMAL", "NORMAL"])
            observed_calls = []
            def runner(path, **kwargs):
                observed_calls.append(guard.calls)
                size = path.stat().st_size
                return {"file_size_bytes": size, "bytes_read": size, "complete": True,
                        "block_bytes": kwargs["block_bytes"], "operations": 0,
                        "elapsed_seconds": 1.0, "throughput_mb_s": 1.0}
            result = self.manager(Path(tmp), guard=guard, sequential_runner=runner,
                                  random_runner=lambda _path, **kwargs: {"complete": True,
                                      "operations": kwargs["operations"], "bytes_read": kwargs["operations"] * kwargs["block_bytes"],
                                      "block_bytes": kwargs["block_bytes"], "elapsed_seconds": 1.0}) .run_benchmark()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(observed_calls), 2)
            self.assertGreaterEqual(observed_calls[0], 3)  # policy, before file, before first read
            self.assertGreater(observed_calls[1], observed_calls[0])  # forced check before repeat

    def test_low_free_space_refuses_test_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = Mock(side_effect=AssertionError("test file should not be created"))
            result = self.manager(Path(tmp), disk_free=100, file_writer=writer).run_benchmark()
            self.assertEqual(result["status"], "failed")
            writer.assert_not_called()
            self.assertIn("Insufficient free disk space", result["error"])

    def test_temp_directory_outside_project_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            cfg = config(temp_directory=outside)
            result = self.manager(Path(tmp), config_value=cfg).run_benchmark()
            self.assertEqual(result["status"], "failed")
            self.assertIn("inside the project", result["error"])

    def test_file_writer_uses_exclusive_creation_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "owned.bin"
            path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                NvmeBenchmarkManager._write_test_file(path, 4096)
            self.assertEqual(path.read_bytes(), b"preserve")

    def test_project_lock_rejects_a_second_simultaneous_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".nvme-io.lock"
            with path.open("a+b") as first, path.open("a+b") as second:
                if path.stat().st_size == 0:
                    first.write(b"0")
                    first.flush()
                NvmeBenchmarkManager._lock_file(first, acquire=True)
                try:
                    with self.assertRaises(BenchmarkBusyError):
                        NvmeBenchmarkManager._lock_file(second, acquire=True)
                finally:
                    NvmeBenchmarkManager._lock_file(first, acquire=False)

    def test_stale_cleanup_removes_only_marked_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "owned"
            run_id = "a" * 32
            run_dir = base / f"run-{run_id}-old"
            run_dir.mkdir(parents=True)
            marker = run_dir / ".nvme-io-owned"
            marker.write_text(run_id, encoding="utf-8")
            generated = run_dir / f"nvmeio-{run_id}-16mib.bin"
            generated.write_bytes(b"module data")
            unknown = run_dir / "user.txt"
            unknown.write_text("preserve", encoding="utf-8")
            NvmeBenchmarkManager._cleanup_stale_runs(base)
            self.assertFalse(generated.exists())
            self.assertTrue(unknown.exists())
            self.assertTrue(marker.exists())

    def test_reporter_appends_new_json_and_csv_without_touching_other_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            old = logs / "performance.json"
            old.write_text('{"events": [1]}', encoding="utf-8")
            reporter = NvmeIoReporter(logs, root / "reports")
            reporter.append({"operation": "seq", "result": {"throughput_mb_s": 123}})
            reporter.append({"operation": "rand", "result": {"iops": 4}})
            self.assertEqual(len(json.loads((logs / "nvme_io.json").read_text(encoding="utf-8"))), 2)
            self.assertEqual(old.read_text(encoding="utf-8"), '{"events": [1]}')
            with (logs / "nvme_io.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_reporter_refuses_malformed_existing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            path = logs / "nvme_io.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                NvmeIoReporter(logs, Path(tmp) / "reports").append({"operation": "x"})

    def test_latest_csv_event_reads_last_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = NvmeIoReporter(Path(tmp), Path(tmp) / "reports")
            reporter.append({"operation": "old", "throughput_mb_s": 1})
            reporter.append({"operation": "new", "throughput_mb_s": 2})
            latest = latest_csv_event(reporter.csv_path)
            self.assertEqual(latest["operation"], "new")
            self.assertEqual(latest["throughput_mb_s"], "2")

    def test_info_method_does_not_construct_or_run_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = Mock(return_value={"drive": "C:\\", "device_model": "simulated"})
            manager = self.manager(Path(tmp), device_provider=provider)
            self.assertEqual(manager.device_info()["device_model"], "simulated")
            self.assertFalse((Path(tmp) / "benchmark" / "nvme_io_runs").exists())

    def test_memory_query_failure_during_run_stops_and_preserves_error_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = FakeGuard(["NORMAL", "UNAVAILABLE"])
            result = self.manager(Path(tmp), guard=guard).run_benchmark()
            self.assertEqual(result["status"], "blocked")
            self.assertIn("simulated memory API failure", result["error"])
            self.assertTrue((Path(tmp) / "reports" / "nvme_io_report.txt").exists())

    def test_user_interrupt_preserves_report_and_cleans_owned_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            def interrupt(*_args, **_kwargs):
                raise KeyboardInterrupt()
            manager = NvmeBenchmarkManager(config(), project_root=tmp, memory_guard=FakeGuard(),
                adaptive_policy=FakePolicy(), device_provider=lambda drive: {"drive": drive},
                disk_usage=lambda _path: SimpleNamespace(free=100 * MIB), sequential_runner=interrupt)
            result = manager.run_benchmark()
            self.assertEqual(result["status"], "interrupted")
            self.assertTrue((Path(tmp) / "reports" / "nvme_io_report.txt").exists())
            self.assertFalse(list((Path(tmp) / "benchmark" / "nvme_io_runs").glob("run-*")))


if __name__ == "__main__":
    unittest.main()
