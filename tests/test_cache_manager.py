import unittest

from engine.cache_manager import AdaptiveBlockCache, ENTRY_OVERHEAD_BYTES, MIB


class AdaptiveCacheTests(unittest.TestCase):
    def test_disables_cache_under_memory_warning_and_bypasses(self):
        cache = AdaptiveBlockCache(max_capacity_mib=1)
        capacity = cache.adjust_capacity(
            available_bytes=2 * MIB,
            configured_budget_bytes=3 * MIB,
            reserve_bytes=512 * 1024,
            pressure="warning",
        )
        self.assertEqual(capacity, 0)
        self.assertFalse(cache.put("a", b"data"))
        self.assertEqual(cache.get("a"), None)
        self.assertEqual(cache.stats()["disabled_bypasses"], 2)

    def test_adapts_capacity_and_evicts_least_used_on_shrink(self):
        cache = AdaptiveBlockCache(max_capacity_mib=1)
        cache.adjust_capacity(
            available_bytes=4 * MIB,
            configured_budget_bytes=4 * MIB,
            reserve_bytes=0,
            pressure="normal",
        )
        cache.put("cold", b"c" * 400)
        cache.get("cold")
        cache.get("cold")
        cache.put("hot", b"h" * 400)
        cache.get("hot")
        cache.get("hot")
        cache.get("hot")
        cache.put("new", b"n" * 400)
        cache.adjust_capacity(
            available_bytes=ENTRY_OVERHEAD_BYTES + 400,
            configured_budget_bytes=4 * MIB,
            reserve_bytes=0,
            pressure="normal",
        )
        self.assertIsNone(cache.get("cold"))
        self.assertEqual(cache.get("hot"), b"h" * 400)
        self.assertLessEqual(cache.used_bytes, cache.capacity_bytes)

    def test_get_or_load_and_hit_rate(self):
        cache = AdaptiveBlockCache(max_capacity_mib=1)
        cache.adjust_capacity(available_bytes=MIB, configured_budget_bytes=MIB, reserve_bytes=0, pressure="normal")
        self.assertEqual(cache.get_or_load("x", lambda: b"block"), b"block")
        self.assertEqual(cache.get_or_load("x", lambda: b"wrong"), b"block")
        self.assertEqual(cache.stats()["hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
