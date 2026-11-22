"""Tests for v0.2 first-principles hardware ledger helpers and published datasets."""

import json
from pathlib import Path
import unittest

from moe_residency.ledger_v2 import (
    classify_arm_validity,
    estimate_k3a_crossover_tokens,
    marginal_row_slope,
    recompute_v2_rc_d1_summary,
)

ROOT = Path(__file__).resolve().parents[1]
V2_DATA = ROOT / "data" / "2026-11-22_a6000-v2"


class LedgerV2HelperTest(unittest.TestCase):
    def test_marginal_row_slope_removes_intercept_bias(self):
        # Identical underlying linear law: win_ms = 6.64 + 5.90 * rows
        s_low = marginal_row_slope(6.64 + 5.90 * 4.0, 4.0)
        s_high = marginal_row_slope(6.64 + 5.90 * 6.0, 6.0)
        self.assertAlmostEqual(s_low, 5.90, places=6)
        self.assertAlmostEqual(s_high, 5.90, places=6)
        with self.assertRaises(ValueError):
            marginal_row_slope(20.0, 0.0)

    def test_cuda_p2_state_and_thermal_gate(self):
        # Cold A6000 runs at 7601 MHz in CUDA P2 with 405 MHz idle tail -> VALID
        ok, reasons = classify_arm_validity({
            "thermal_throttle_hits": 0,
            "temp_max_c": 74,
            "mem_mhz_p50": 7601,
            "cpu_interference_hits": 0,
            "mem_free_min_mib": 4703,
        })
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

        # Real thermal throttle or low VRAM floor -> INVALID
        ok_bad, reasons_bad = classify_arm_validity({
            "thermal_throttle_hits": 2,
            "temp_max_c": 88,
            "mem_mhz_p50": 7200,
            "cpu_interference_hits": 1,
            "mem_free_min_mib": 1800,
        })
        self.assertFalse(ok_bad)
        self.assertEqual(len(reasons_bad), 5)

    def test_k3a_crossover_solver_matches_measured_range(self):
        # Clean CHECK=0 measured ~4.4 ms per sub_b=15 -> ~365 tok crossover
        t_cross = estimate_k3a_crossover_tokens(sub_batch_ms=4.4, sub_b=15)
        self.assertGreater(t_cross, 340.0)
        self.assertLess(t_cross, 400.0)


class V2PublishedDatasetTest(unittest.TestCase):
    def test_v2_rc_grand_exam_recomputation_and_gates(self):
        exam = json.loads((V2_DATA / "v2_rc_grand_exam.json").read_text(encoding="utf-8"))
        summary = recompute_v2_rc_d1_summary(exam)

        # Pure decode window gain (strictly separated from end-to-end burst speedup)
        self.assertEqual(summary["off_win_ms"], 41.915)
        self.assertEqual(summary["on_win_ms"], 34.315)
        self.assertEqual(summary["win_delta_ms"], -7.6)
        self.assertEqual(summary["win_reduction_pct"], 18.1)
        self.assertEqual(summary["gpu_wait_reduction_pct"], 17.5)
        self.assertEqual(summary["emit_reduction_pct"], 33.7)
        self.assertEqual(summary["off_slope_win_adj"], 5.90)
        self.assertEqual(summary["on_slope_win_adj"], 4.66)
        self.assertEqual(summary["slope_reduction_pct"], 21.0)

        # End-to-end 4x100K burst speedup (decode + non-decode prefill/park/restore/IPC)
        self.assertEqual(summary["off_wall_s"], 20.55)
        self.assertEqual(summary["on_wall_s"], 11.92)
        self.assertEqual(summary["wall_reduction_pct"], 42.0)
        self.assertEqual(summary["wall_speedup_x"], 1.72)
        self.assertEqual(summary["decode_only_reduction_pct"], 17.8)
        self.assertEqual(summary["non_decode_reduction_pct"], 51.8)

        # Every D1 arm must pass validity gates and share exact 918 completion tokens
        for arm_name, row in exam["D1_four_lane_100k_abab"].items():
            ok, reasons = classify_arm_validity(row)
            self.assertTrue(ok, f"{arm_name} failed validity: {reasons}")
            self.assertEqual(row["total_completion_tokens"], 918)
            self.assertEqual(row["completion_tokens_per_lane"], [150, 256, 256, 256])

    def test_s1_l1_recorded_as_falsified_and_excluded(self):
        abl = json.loads((V2_DATA / "phase2_rfc_ablations.json").read_text(encoding="utf-8"))
        falsified = json.loads((V2_DATA / "falsified_registry.json").read_text(encoding="utf-8"))
        s1_l1 = abl["rfcs"]["RFC-S1-L1"]
        self.assertEqual(s1_l1["status"], "falsified_veto_merge")
        self.assertTrue(s1_l1["excluded_from_release"])
        self.assertTrue(any(item["id"] == "F7" for item in falsified["items"]))

        # Verify K3-A clean crossover flips sign between T=284 and T=424
        xover = abl["rfcs"]["RFC-K3-v2"]["clean_check0_crossover_table"]
        by_tok = {row["tokens"]: row["k3a_delta_ms"] for row in xover}
        self.assertLess(by_tok[183], 0)
        self.assertLess(by_tok[284], 0)
        self.assertGreater(by_tok[424], 0)
        self.assertGreater(by_tok[566], 0)


if __name__ == "__main__":
    unittest.main()
