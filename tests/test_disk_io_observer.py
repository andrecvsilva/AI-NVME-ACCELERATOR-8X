import unittest

from engine.disk_io_observer import MIB, PhysicalDiskIoObserver


class DiskIoObserverTests(unittest.TestCase):
    def test_integrates_disk_rates_and_latency_without_claiming_file_attribution(self):
        observer = PhysicalDiskIoObserver(source_factory=lambda _: None)
        observer._record({
            "read_bytes_per_second": 4 * MIB,
            "read_operations_per_second": 100,
            "average_read_latency_seconds": 0.002,
        }, 1.0)
        observer._record({
            "read_bytes_per_second": 4 * MIB,
            "read_operations_per_second": 100,
            "average_read_latency_seconds": 0.002,
        }, 2.0)

        result = observer.stop()
        self.assertEqual(result["estimated_read_bytes"], 4 * MIB)
        self.assertEqual(result["estimated_read_ops"], 100)
        self.assertEqual(result["estimated_iops"], 100)
        self.assertEqual(result["average_read_mib_s"], 4)
        self.assertEqual(result["weighted_average_read_latency_ms"], 2)
        self.assertFalse(result["per_file_attribution_available"])
        self.assertFalse(result["per_operation_latency_percentiles_available"])

    def test_unavailable_counter_is_reported_without_raising(self):
        observer = PhysicalDiskIoObserver(
            source_factory=lambda _: (_ for _ in ()).throw(OSError("counter unavailable"))
        ).start()
        result = observer.stop()
        self.assertFalse(result["available"])
        self.assertIn("counter unavailable", result["limitation"])

    def test_rejects_unbounded_or_invalid_poll_frequency(self):
        with self.assertRaises(ValueError):
            PhysicalDiskIoObserver(interval_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
