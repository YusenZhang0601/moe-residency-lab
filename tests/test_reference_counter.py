"""Bounded CPU checks of the preserved deployment source, with synthetic state."""

import atexit
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("historical_counter", ROOT / "reference/routing_counter_20261104.py")
counter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = counter
spec.loader.exec_module(counter)


class ReferenceCounterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.spec = counter.CounterSpec(
            n_layers=2, n_experts=16, top_k=10, layer_blob_bytes=(256, 512), gpu_slots=4,
            counts_bin=root / "counts.bin", stats_json=root / "stats.json", profile_path=root / "profile.bin",
            identity={"model": "synthetic-test"}, bind_mode="strict")
        self.manager = self.make_manager(self.spec)
        self.record = struct.pack("<ii10i10f", 0, 10, *range(10), *([1.0] * 10))

    def make_manager(self, spec):
        manager = counter.RoutingCounterManager(spec)
        self.addCleanup(atexit.unregister, manager.save_if_dirty)
        return manager

    def test_fragment_boundary_and_persistence(self):
        self.assertEqual(self.manager.ingest_raw_bytes(self.record[:30]), 0)
        self.assertEqual(self.manager.ingest_raw_bytes(self.record[:30] + self.record[30:]), 88)
        self.manager.save(force_rebuild_profile=True)
        restored = self.make_manager(self.spec)
        self.assertEqual(restored.live_records, 1)
        self.assertEqual(sum(restored.live_counts), 10)
        self.assertEqual(self.spec.counts_bin.stat().st_size, self.spec.expected_bin_bytes)

    def test_identity_mismatch_and_corrupt_ledger(self):
        self.manager.save()
        wrong = counter.CounterSpec(**{**self.spec.__dict__, "identity": {"model": "other"}})
        with self.assertRaises(counter.CounterIdentityError):
            self.make_manager(wrong)
        self.spec.counts_bin.write_bytes(b"broken")
        with self.assertRaises(counter.CounterIdentityError):
            self.make_manager(self.spec)

    def test_failed_save_remains_dirty(self):
        self.manager.ingest_raw_bytes(self.record)
        with mock.patch.object(self.manager, "_atomic_write", return_value=False):
            self.manager.save()
        self.assertTrue(self.manager._dirty)
        self.manager.save()
        self.assertFalse(self.manager._dirty)

    def test_parallel_ingest_and_save_with_final_flush(self):
        errors = []

        def worker():
            try:
                for _ in range(8):
                    self.manager.ingest_raw_bytes(self.record)
                    self.manager.save()
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.manager.save()
        restored = self.make_manager(self.spec)
        self.assertEqual((restored.live_records, sum(restored.live_counts)), (24, 240))
        self.assertEqual(list(self.spec.counts_bin.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
