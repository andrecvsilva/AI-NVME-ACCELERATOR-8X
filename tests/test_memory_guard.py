import unittest
from unittest.mock import patch

from engine.memory_guard import GIB, MIB, evaluate_pressure


class MemoryGuardTests(unittest.TestCase):
    def test_critical_when_available_memory_is_low(self):
        result = evaluate_pressure(
            available_bytes=400 * MIB,
            total_bytes=16 * GIB,
            physical_memory_load_percent=98,
            commit_used_percent=40,
            budget_bytes=3 * GIB,
            reserve_bytes=512 * MIB,
        )
        self.assertEqual(result["pressure"], "critical")
        self.assertEqual(result["recommended_cache_headroom_bytes"], 0)
        self.assertTrue(result["physical_memory_exceeds_configured_budget"])

    def test_warning_when_commit_pressure_is_high(self):
        result = evaluate_pressure(
            available_bytes=2 * GIB,
            total_bytes=4 * GIB,
            physical_memory_load_percent=50,
            commit_used_percent=90,
            budget_bytes=3 * GIB,
            reserve_bytes=512 * MIB,
        )
        self.assertEqual(result["pressure"], "warning")
        self.assertEqual(result["recommended_cache_headroom_bytes"], int(1.5 * GIB))

    def test_normal_is_recommendation_only(self):
        result = evaluate_pressure(
            available_bytes=8 * GIB,
            total_bytes=16 * GIB,
            physical_memory_load_percent=50,
            commit_used_percent=45,
            budget_bytes=3 * GIB,
            reserve_bytes=512 * MIB,
        )
        self.assertEqual(result["pressure"], "normal")
        self.assertTrue(result["recommendation_only"])

    def test_invalid_memory_percent_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_pressure(
                available_bytes=1,
                total_bytes=1,
                physical_memory_load_percent=0,
                commit_used_percent=101,
                budget_bytes=1,
                reserve_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
