import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from moe_residency.analysis import (
    LEDGER_HEADER, coverage, decode_profile, encode_profile, rank_experts, read_ledger,
)


def ledger_blob(counts=(30, 10, 20, 40), experts=2, top_k=2):
    return (LEDGER_HEADER.pack(b"STRC", 1, 2, experts, top_k, 0, 50, 1, 0)
            + struct.pack(f"<{len(counts) * 2}Q", *([0] * len(counts) + list(counts))))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.ledger = read_ledger(ledger_blob())
        self.sizes = [100, 20]

    def test_combines_historical_and_live_arrays(self):
        blob = LEDGER_HEADER.pack(b"STRC", 1, 1, 2, 1, 3, 7, 1, 0)
        self.assertEqual(read_ledger(blob + struct.pack("<4Q", 1, 2, 3, 4)).counts, (4, 6))

    def test_rejects_truncation_and_trailing_bytes(self):
        for blob in (b"", ledger_blob()[:-1], ledger_blob() + b"x"):
            with self.subTest(size=len(blob)), self.assertRaises(ValueError):
                read_ledger(blob)

    def test_rejects_header_mismatch(self):
        for magic, version in ((b"NOPE", 1), (b"STRC", 2)):
            blob = bytearray(ledger_blob())
            struct.pack_into("<4sI", blob, 0, magic, version)
            with self.assertRaises(ValueError):
                read_ledger(blob)

    def test_rejects_invalid_geometry(self):
        blob = bytearray(ledger_blob())
        struct.pack_into("<I", blob, 8, 0)
        with self.assertRaises(ValueError):
            read_ledger(blob)

    def test_density_and_frequency_have_distinct_orders(self):
        self.assertEqual(rank_experts(self.ledger, self.sizes), [3, 2, 0, 1])
        self.assertEqual(rank_experts(self.ledger, self.sizes, "frequency"), [3, 0, 2, 1])

    def test_exact_large_integer_ratios(self):
        ledger = read_ledger(ledger_blob((2**63, 2**63 + 1, 0, 0)))
        self.assertEqual(rank_experts(ledger, [1, 1])[:2], [1, 0])

    def test_ties_use_stable_flat_id(self):
        ledger = read_ledger(ledger_blob((0, 0, 0, 0)))
        self.assertEqual(rank_experts(ledger, [1, 1]), [0, 1, 2, 3])
        self.assertIsNone(coverage(ledger, [1, 1], [0, 1, 2, 3], 4)["selection_share"])

    def test_prefix_barrier_does_not_backfill(self):
        result = coverage(self.ledger, self.sizes, [0, 1, 2, 3], 110)
        self.assertEqual((result["experts"], result["resident_blob_bytes"]), (1, 100))

    def test_zero_exact_and_overfull_budgets(self):
        order = [3, 2, 0, 1]
        for budget, members in ((0, 0), (20, 1), (40, 2), (240, 4), (1000, 4)):
            self.assertEqual(coverage(self.ledger, self.sizes, order, budget)["experts"], members)

    def test_invalid_layout_policy_order_or_budget(self):
        for sizes in ([1], [1, 0], [1, -1], [1, True]):
            with self.assertRaises(ValueError):
                rank_experts(self.ledger, sizes)
        with self.assertRaises(ValueError):
            rank_experts(self.ledger, self.sizes, "unknown")
        with self.assertRaises(ValueError):
            coverage(self.ledger, self.sizes, [0, 0, 2, 3], 1)
        with self.assertRaises(ValueError):
            coverage(self.ledger, self.sizes, [0, 1, 2, 3], -1)

    def test_profile_roundtrip_non_power_of_two_geometry(self):
        ledger = read_ledger(ledger_blob((1, 2, 3, 4, 5, 6), experts=3))
        order = rank_experts(ledger, [10, 20])
        self.assertEqual(decode_profile(encode_profile(ledger, order), ledger), order)

    def test_profile_rejects_bad_inverse_and_duplicates(self):
        blob = bytearray(encode_profile(self.ledger, [0, 1, 2, 3]))
        blob[-1] = 1
        with self.assertRaises(ValueError):
            decode_profile(blob, self.ledger)
        with self.assertRaises(ValueError):
            encode_profile(self.ledger, [0, 0, 2, 3])

    def test_strp_geometry_mismatch(self):
        blob = bytearray(encode_profile(self.ledger, [0, 1, 2, 3]))
        struct.pack_into("<I", blob, 8, 3)
        with self.assertRaises(ValueError):
            decode_profile(blob, self.ledger)


class CommandTests(unittest.TestCase):
    def test_readonly_bound_inputs_and_exclusive_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = ledger_blob()
            ledger = root / "counts.bin"
            layout = root / "layout.json"
            profile = root / "profile.bin"
            ledger.write_bytes(blob)
            metadata = {"model_identity": "test", "capture_coverage": "synthetic",
                        "ledger_sha256": hashlib.sha256(blob).hexdigest(), "layer_blob_bytes": [100, 20]}
            layout.write_text(json.dumps(metadata))
            command = [sys.executable, "-B", "-m", "moe_residency", "--ledger", str(ledger),
                       "--layout", str(layout), "--budgets-gib", "0", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["geometry"]["layers"], 2)
            self.assertEqual(set(root.iterdir()), {ledger, layout})
            exported = subprocess.run(command + ["--export-profile", str(profile)], capture_output=True)
            self.assertEqual(exported.returncode, 0, exported.stderr)
            before = profile.read_bytes()
            duplicate = subprocess.run(command + ["--export-profile", str(profile)], capture_output=True)
            self.assertEqual(duplicate.returncode, 2)
            self.assertEqual(profile.read_bytes(), before)
            metadata["ledger_sha256"] = "0" * 64
            layout.write_text(json.dumps(metadata))
            mismatch = subprocess.run(command, capture_output=True)
            self.assertEqual(mismatch.returncode, 2)
            self.assertEqual(ledger.read_bytes(), blob)

    def test_demo_and_reject_nonfinite_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            fixture = subprocess.run([sys.executable, "-B", "examples/make_fixture.py", str(output)], capture_output=True)
            self.assertEqual(fixture.returncode, 0, fixture.stderr)
            result = subprocess.run([sys.executable, "-B", "-m", "moe_residency", "--ledger", str(output / "counts.bin"),
                                     "--layout", str(output / "layout.json"), "--budgets-gib", "NaN"], capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
