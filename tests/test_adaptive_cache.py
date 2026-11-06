import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_cache.cache_manager import AdaptiveCacheEngine
from adaptive_cache.config import AdaptiveCacheConfig
from adaptive_cache.eviction import rank_entries
from adaptive_cache.memory_policy import AdaptiveMemoryPolicy
from adaptive_cache.nvme_cache import NvmeCache
from engine.cache_manager import AdaptiveBlockCache, MIB
from memory_guard.monitor import GIB, MemorySnapshot
from memory_guard.protection import MemoryGuard


def mem_snapshot(state="NORMAL", available_gib=8):
    load = {"NORMAL": 40, "WARNING": 85, "CRITICAL": 97}.get(state, 40)
    commit = {"NORMAL": 40, "WARNING": 85, "CRITICAL": 97}.get(state, 40)
    total = 16 * GIB
    available = int(available_gib * GIB)
    return MemorySnapshot("2026-11-06T00:00:00+00:00", total, available, total-available,
                          load, 40*GIB, int(40*GIB*(100-commit)/100), int(40*GIB*commit/100), commit)


def make_config(**changes):
    defaults = AdaptiveCacheConfig(
        ram_cache_max_mib=1, ram_budget_mib=2, ram_reserve_mib=512,
        nvme_directory="cache/nvme", nvme_max_mib=2, nvme_min_free_mib=0,
        nvme_item_max_mib=1, entry_ttl_seconds=3600, prefetch_enabled=False,
        prefetch_block_kib=4,
    )
    return replace(defaults, **changes)


def memory_policy(state="NORMAL", available_gib=8, process_report=None):
    snapshot = mem_snapshot(state, available_gib)
    guard = MemoryGuard(snapshot_provider=lambda: snapshot)
    return AdaptiveMemoryPolicy(
        make_config(), memory_guard=guard,
        process_provider=lambda: process_report or {"status": "active", "processes": []},
    )


class AdaptiveMemoryPolicyTests(unittest.TestCase):
    def test_normal_uses_bounded_adaptive_budget(self):
        decision = memory_policy("NORMAL").evaluate()
        self.assertEqual(decision.guard_state, "NORMAL")
        self.assertTrue(decision.allow_ram_insert)
        self.assertGreater(decision.capacity_bytes, 0)
        self.assertLessEqual(decision.capacity_bytes, MIB)

    def test_warning_reduces_ram_and_blocks_nvme_and_prefetch(self):
        decision = memory_policy("WARNING", 3).evaluate()
        self.assertEqual(decision.guard_state, "WARNING")
        self.assertLessEqual(decision.capacity_bytes, 4*MIB)
        self.assertFalse(decision.allow_nvme_read)
        self.assertFalse(decision.allow_nvme_write)
        self.assertFalse(decision.allow_prefetch)

    def test_critical_blocks_new_cache_allocations(self):
        decision = memory_policy("CRITICAL", 1).evaluate()
        self.assertEqual(decision.capacity_bytes, 0)
        self.assertFalse(decision.allow_ram_insert)
        self.assertFalse(decision.allow_nvme_read)
        self.assertFalse(decision.allow_prefetch)

    def test_memory_guard_failure_fails_closed(self):
        guard = MemoryGuard(snapshot_provider=lambda: (_ for _ in ()).throw(OSError("probe failed")))
        decision = AdaptiveMemoryPolicy(make_config(), memory_guard=guard).evaluate()
        self.assertEqual(decision.guard_state, "UNAVAILABLE")
        self.assertEqual(decision.capacity_bytes, 0)
        self.assertFalse(decision.allow_nvme_write)

    def test_relevant_process_working_set_lowers_capacity(self):
        small = memory_policy("NORMAL", process_report={"status":"active", "processes":[]}).evaluate()
        large = memory_policy("NORMAL", process_report={"status":"active", "processes":[
            {"status":"ok", "private_working_set_bytes": 8*GIB}
        ]}).evaluate()
        self.assertLess(large.capacity_bytes, small.capacity_bytes)


class EvictionPolicyTests(unittest.TestCase):
    def test_lru_order(self):
        ranked = rank_entries([{"key":"old", "accessed_at":1,"frequency":8},
                               {"key":"new", "accessed_at":9,"frequency":1}], "LRU")
        self.assertEqual(ranked[0]["key"], "old")

    def test_lfu_order(self):
        ranked = rank_entries([{"key":"rare", "accessed_at":9,"frequency":1},
                               {"key":"frequent", "accessed_at":1,"frequency":8}], "LFU")
        self.assertEqual(ranked[0]["key"], "rare")

    def test_hybrid_combines_frequency_and_recency(self):
        ranked = rank_entries([{"key":"stale", "accessed_at":1,"frequency":2},
                               {"key":"recent_rare", "accessed_at":10,"frequency":1}], "HYBRID")
        self.assertEqual(ranked[0]["key"], "stale")


class NvmeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "nvme"
        self.config = make_config()
        self.cache = NvmeCache(self.config, root=self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_disk_cache_round_trip_and_hash_named_blob(self):
        self.assertTrue(self.cache.put("small-key", b"payload"))
        self.assertEqual(self.cache.get("small-key"), b"payload")
        blobs = list(self.root.glob("[0-9a-f]*.blob"))
        self.assertEqual(len(blobs), 1)

    def test_maximum_ssd_size_evicts_old_entries(self):
        config = make_config(nvme_max_mib=1, nvme_item_max_mib=1)
        cache = NvmeCache(config, root=self.root)
        self.assertTrue(cache.put("first", b"a"*(700*1024)))
        self.assertTrue(cache.put("second", b"b"*(700*1024)))
        self.assertLessEqual(cache.stats()["blob_bytes"], 1024*1024)
        self.assertIsNone(cache.get("first"))

    def test_low_free_space_refuses_write(self):
        usage = lambda _path: SimpleNamespace(total=10000, used=9500, free=500)
        cache = NvmeCache(make_config(nvme_min_free_mib=1), root=self.root, disk_usage=usage)
        self.assertFalse(cache.put("blocked", b"data"))
        self.assertEqual(list(self.root.glob("*.blob")), [])

    def test_corrupt_blob_is_detected_and_removed(self):
        self.cache.put("corrupt", b"actual")
        blob = next(self.root.glob("*.blob"))
        blob.write_bytes(b"tampered")
        self.assertIsNone(self.cache.get("corrupt"))
        self.assertEqual(self.cache.corruptions, 1)
        self.assertFalse(blob.exists())

    def test_invalid_hash_in_index_is_a_safe_miss(self):
        import time
        now = time.time()
        self.cache.index.put(self.cache._key_hash("bad"), "not-a-hash", 5, now, now + 10000)
        self.assertIsNone(self.cache.get("bad"))
        self.assertEqual(self.cache.corruptions, 1)

    def test_duplicate_content_uses_one_blob(self):
        self.assertTrue(self.cache.put("key-1", b"shared"))
        self.assertTrue(self.cache.put("key-2", b"shared"))
        self.assertEqual(len(list(self.root.glob("*.blob"))), 1)
        self.assertEqual(self.cache.index.count(), 2)
        self.assertEqual(self.cache.index.total_size(), len(b"shared"))

    def test_atomic_write_failure_cleans_temporary_cache_file(self):
        with patch("adaptive_cache.nvme_cache.os.replace", side_effect=OSError("simulated write error")):
            with self.assertRaises(OSError):
                self.cache.put("write-fail", b"data")
        self.assertEqual(list(self.root.glob(".adaptive-cache-tmp-*")), [])
        self.assertEqual(list(self.root.glob("*.blob")), [])

    def test_read_failure_becomes_miss_and_removes_mapping(self):
        self.cache.put("read-fail", b"data")
        with patch("pathlib.Path.read_bytes", side_effect=PermissionError("denied")):
            self.assertIsNone(self.cache.get("read-fail"))
        self.assertEqual(self.cache.index.count(), 0)

    def test_corrupt_index_recovers_and_prunes_only_owned_blobs(self):
        self.cache.put("lost", b"orphan")
        sentinel = self.root / "notes.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        (self.root / "index.sqlite3").write_bytes(b"not sqlite")
        recovered = NvmeCache(self.config, root=self.root)
        self.assertTrue(recovered.recovered_index)
        self.assertTrue(sentinel.exists())
        self.assertEqual(list(self.root.glob("*.blob")), [])
        self.assertTrue(list(self.root.glob("index.sqlite3.corrupt*")))

    def test_expiration_removes_only_cache_blob(self):
        sentinel = self.root / "user.txt"
        sentinel.write_text("keep", encoding="utf-8")
        expiring = NvmeCache(make_config(entry_ttl_seconds=1), root=self.root)
        with patch("adaptive_cache.nvme_cache.time.time", side_effect=[100.0, 102.0]):
            self.assertTrue(expiring.put("expires", b"old",))
            self.assertEqual(expiring.remove_expired(), 1)
        self.assertTrue(sentinel.exists())
        self.assertEqual(list(self.root.glob("*.blob")), [])

    def test_nvme_feature_can_be_disabled_without_creating_files(self):
        root = self.root / "disabled"
        cache = NvmeCache(make_config(nvme_enabled=False), root=root)
        self.assertFalse(cache.put("a", b"b"))
        self.assertIsNone(cache.get("a"))
        self.assertFalse(root.exists())


class AdaptiveCacheManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manager(self, state="NORMAL", **config_changes):
        config = make_config(**config_changes)
        snap = mem_snapshot(state)
        guard = MemoryGuard(snapshot_provider=lambda: snap)
        policy = AdaptiveMemoryPolicy(config, memory_guard=guard,
                                      process_provider=lambda: {"status":"active", "processes":[]})
        return AdaptiveCacheEngine(config, project_root=self.root, policy=policy)

    def test_cache_hit_and_miss_across_ram_nvme(self):
        manager = self.manager()
        try:
            self.assertTrue(manager.put("key", b"payload"))
            self.assertEqual(manager.get("key"), b"payload")
            manager.clear_ram()
            self.assertEqual(manager.get("key"), b"payload")
            self.assertIsNone(manager.get("absent"))
            stats = manager.stats()
            self.assertEqual(stats["metrics"]["hits"], 2)
            self.assertEqual(stats["metrics"]["misses"], 1)
        finally:
            manager.close()

    def test_entire_cache_can_be_disabled_without_creating_nvme_files(self):
        manager = self.manager(enabled=False, nvme_enabled=True)
        try:
            self.assertIsNone(manager.get("key"))
            self.assertFalse(manager.put("key", b"value"))
            stats = manager.stats()
            self.assertFalse(stats["cache_active"])
            self.assertFalse(stats["nvme"]["enabled"])
            self.assertFalse((self.root / "cache" / "nvme").exists())
        finally:
            manager.close()

    def test_repeated_dashboard_stats_do_not_append_redundant_policy_events(self):
        manager = self.manager()
        try:
            manager.stats()
            manager.stats()
            events = json.loads(manager.logger.json_path.read_text(encoding="utf-8"))
            self.assertEqual(sum(event["operation"] == "policy" for event in events), 1)
        finally:
            manager.close()

    def test_ram_limit_is_never_exceeded(self):
        manager = self.manager(ram_cache_max_mib=1, ram_budget_mib=1)
        try:
            manager.put("one", b"a"*(600*1024))
            manager.put("two", b"b"*(600*1024))
            stats = manager.stats()
            self.assertLessEqual(stats["ram"]["used_accounted_bytes"], stats["ram"]["capacity_bytes"])
            self.assertLessEqual(stats["ram"]["capacity_bytes"], MIB)
        finally:
            manager.close()

    def test_warning_and_critical_prevent_nvme_writes_and_prefetch(self):
        for state in ("WARNING", "CRITICAL"):
            manager = self.manager(state, prefetch_enabled=True)
            try:
                stored = manager.put("blocked", b"payload")
                if state == "CRITICAL":
                    self.assertFalse(stored)
                else:
                    self.assertTrue(stored)  # bounded RAM cache remains available in WARNING
                self.assertIsNone(manager.observe_prefetch(manager.nvme.root / "missing.blob", 0))
                self.assertEqual(manager.nvme.stats()["entries"], 0)
                self.assertEqual(manager.prefetcher.stats()["scheduled"], 0)
            finally:
                manager.close()

    def test_prefetch_rejects_files_outside_owned_nvme_directory(self):
        manager = self.manager(prefetch_enabled=True)
        outside = self.root / "outside.blob"
        outside.write_bytes(b"x"*4096)
        try:
            self.assertIsNone(manager.observe_prefetch(outside, 0))
            self.assertEqual(manager.prefetcher.stats()["scheduled"], 0)
        finally:
            manager.close()

    def test_cache_events_are_json_and_csv(self):
        manager = self.manager()
        try:
            manager.put("event", b"data")
            manager.get("event")
            self.assertTrue((self.root / "logs" / "adaptive_cache.json").exists())
            self.assertTrue((self.root / "logs" / "adaptive_cache.csv").exists())
            events = json.loads((self.root / "logs" / "adaptive_cache.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(events), 3)
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
