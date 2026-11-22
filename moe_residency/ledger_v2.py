"""Offline physics and gate-validation helpers for the v0.2 A6000 MoE study.

Standard-library only; no network, engine, or host-control side effects.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_DECODE_INTERCEPT_MS = 6.64
CUDA_P2_MEM_MHZ_FLOOR = 7500.0
VRAM_SAFETY_FLOOR_MIB = 2048
THERMAL_THROTTLE_TEMP_C = 87.0


def marginal_row_slope(
    metric_ms: float,
    avg_rows: float,
    intercept_ms: float = DEFAULT_DECODE_INTERCEPT_MS,
) -> float:
    """Compute intercept-adjusted marginal latency per active decode row.

    Comparing raw ``win_ms / avg_rows`` across arms with different ``avg_rows``
    biases higher-row arms because the fixed dense-weight read intercept
    (``6.64 ms`` on RTX A6000 for Qwen3.8-Flash-Next IQ3_XXS) amortizes over
    more rows. Subtracting the measured intercept isolates marginal per-row cost.
    """
    if avg_rows <= 0.0:
        raise ValueError(f"avg_rows must be positive, got {avg_rows}")
    return (metric_ms - intercept_ms) / avg_rows


def classify_arm_validity(telemetry: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate telemetry against v0.2 hardware-measurement validity gates.

    Accounts for NVIDIA CUDA P2 state on RTX A6000 (7601 MHz active compute vs
    405 MHz idle tail), so ``mem_mhz_min == 405`` is not misclassified as
    thermal throttling unless ``mem_mhz_p50 < 7500`` or thermal flags/temp trip.
    """
    reasons: list[str] = []
    if int(telemetry.get("thermal_throttle_hits", 0)) > 0:
        reasons.append("thermal_throttle_hits > 0")
    if float(telemetry.get("temp_max_c", 0.0)) >= THERMAL_THROTTLE_TEMP_C:
        reasons.append(f"temp_max_c >= {THERMAL_THROTTLE_TEMP_C}")
    mem_p50 = telemetry.get("mem_mhz_p50")
    if mem_p50 is not None and float(mem_p50) < CUDA_P2_MEM_MHZ_FLOOR:
        reasons.append(f"mem_mhz_p50 < {CUDA_P2_MEM_MHZ_FLOOR}")
    if int(telemetry.get("cpu_interference_hits", 0)) > 0:
        reasons.append("cpu_interference_hits > 0")
    mem_free = telemetry.get("mem_free_min_mib")
    if mem_free is not None and float(mem_free) < VRAM_SAFETY_FLOOR_MIB:
        reasons.append(f"mem_free_min_mib < {VRAM_SAFETY_FLOOR_MIB}")
    return (len(reasons) == 0), reasons


def estimate_k3a_crossover_tokens(
    sub_batch_ms: float,
    sub_b: int,
    baseline_stage_ms: float = 98.5,
    baseline_attn_intercept_ms: float = 2.0,
    baseline_attn_slope_ms_per_tok: float = 0.0235,
) -> float:
    """Solve the continuous token count where sparse KV breaks even with full staging.

    Sparse KV time scales as ``sub_batch_ms * ceil(T / sub_b) ~= (sub_batch_ms / sub_b) * T``.
    Full-prefix staging scales as ``(baseline_stage_ms + baseline_attn_intercept_ms) + slope * T``.
    """
    if sub_b <= 0 or sub_batch_ms <= 0.0:
        raise ValueError("sub_b and sub_batch_ms must be positive")
    sparse_slope = sub_batch_ms / float(sub_b)
    if sparse_slope <= baseline_attn_slope_ms_per_tok:
        return math.inf
    return (baseline_stage_ms + baseline_attn_intercept_ms) / (
        sparse_slope - baseline_attn_slope_ms_per_tok
    )


def recompute_v2_rc_d1_summary(grand_exam: Mapping[str, Any]) -> dict[str, float]:
    """Recompute D1 pure-decode and end-to-end burst deltas from grand-exam JSON."""
    d1 = grand_exam["D1_four_lane_100k_abab"]
    offs = [d1["v2rc_off1"], d1["v2rc_off2"]]
    ons = [d1["v2rc_on1"], d1["v2rc_on2"]]

    def mean(rows: list[Mapping[str, Any]], key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    off_win = mean(offs, "win_ms")
    on_win = mean(ons, "win_ms")
    off_gpu = mean(offs, "gpu_wait_ms")
    on_gpu = mean(ons, "gpu_wait_ms")
    off_emit = mean(offs, "emit_ms")
    on_emit = mean(ons, "emit_ms")
    off_slope = sum(marginal_row_slope(r["win_ms"], r["avg_rows"]) for r in offs) / len(offs)
    on_slope = sum(marginal_row_slope(r["win_ms"], r["avg_rows"]) for r in ons) / len(ons)

    off_wall = mean(offs, "wall_ms")
    on_wall = mean(ons, "wall_ms")
    off_dec = mean(offs, "decode_only_ms")
    on_dec = mean(ons, "decode_only_ms")
    off_nondec = mean(offs, "non_decode_ms")
    on_nondec = mean(ons, "non_decode_ms")

    return {
        "off_win_ms": round(off_win, 3),
        "on_win_ms": round(on_win, 3),
        "win_delta_ms": round(on_win - off_win, 3),
        "win_reduction_pct": round((off_win - on_win) / off_win * 100.0, 1),
        "gpu_wait_reduction_pct": round((off_gpu - on_gpu) / off_gpu * 100.0, 1),
        "emit_reduction_pct": round((off_emit - on_emit) / off_emit * 100.0, 1),
        "off_slope_win_adj": round(off_slope, 2),
        "on_slope_win_adj": round(on_slope, 2),
        "slope_reduction_pct": round((off_slope - on_slope) / off_slope * 100.0, 1),
        "off_wall_s": round(off_wall / 1000.0, 2),
        "on_wall_s": round(on_wall / 1000.0, 2),
        "wall_reduction_pct": round((off_wall - on_wall) / off_wall * 100.0, 1),
        "wall_speedup_x": round(off_wall / on_wall, 2),
        "decode_only_reduction_pct": round((off_dec - on_dec) / off_dec * 100.0, 1),
        "non_decode_reduction_pct": round((off_nondec - on_nondec) / off_nondec * 100.0, 1),
    }
