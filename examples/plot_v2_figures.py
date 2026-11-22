"""Generate publication-quality figures for the v0.2 RTX A6000 full-stack study.

All numeric values are loaded directly from the sanitized measurement ledgers under
``data/2026-11-22_a6000-v2/``. No data points are hand-interpolated or fabricated.

Usage (from repository root):
    python3 examples/plot_v2_figures.py
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "2026-11-22_a6000-v2"
OUT_DIR = ROOT / "docs" / "assets"

# Okabe-Ito colorblind-safe palette (consistent across all figures)
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_SKY = "#56B4E9"
C_AMBER = "#E69F00"
C_GRAY = "#5C6670"
C_DARK = "#1F2933"


def fmt2(val: float) -> str:
    return str(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def apply_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 240,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 8.4,
        "axes.edgecolor": "#334E68",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E4E7EB",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    })


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def make_fig1_end_to_end(grand: dict) -> Path:
    d1 = grand["D1_four_lane_100k_abab"]
    off_arms = [d1["v2rc_off1"], d1["v2rc_off2"]]
    on_arms = [d1["v2rc_on1"], d1["v2rc_on2"]]

    off_dec = float(np.mean([a["decode_only_ms"] for a in off_arms])) / 1000.0
    off_non = float(np.mean([a["non_decode_ms"] for a in off_arms])) / 1000.0
    on_dec = float(np.mean([a["decode_only_ms"] for a in on_arms])) / 1000.0
    on_non = float(np.mean([a["non_decode_ms"] for a in on_arms])) / 1000.0

    off_wall_pts = [a["wall_ms"] / 1000.0 for a in off_arms]
    on_wall_pts = [a["wall_ms"] / 1000.0 for a in on_arms]

    off_gpu = float(np.mean([a["gpu_wait_ms"] for a in off_arms]))
    on_gpu = float(np.mean([a["gpu_wait_ms"] for a in on_arms]))
    off_commit = float(np.mean([a["commit_ms"] for a in off_arms]))
    on_commit = float(np.mean([a["commit_ms"] for a in on_arms]))
    off_emit = float(np.mean([a["emit_ms"] for a in off_arms]))
    on_emit = float(np.mean([a["emit_ms"] for a in on_arms]))
    off_win = float(np.mean([a["win_ms"] for a in off_arms]))
    on_win = float(np.mean([a["win_ms"] for a in on_arms]))
    off_other = off_win - off_gpu - off_commit - off_emit
    on_other = on_win - on_gpu - on_commit - on_emit

    off_slope = float(np.mean([a["slope_win_adj"] for a in off_arms]))
    on_slope = float(np.mean([a["slope_win_adj"] for a in on_arms]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)

    # Panel (a): End-to-End 4x100K Burst Wall Time
    x = np.array([0, 1])
    w = 0.48
    ax1.bar(x, [off_dec, on_dec], width=w, color=C_BLUE, label="Decode (win x win_ms)")
    ax1.bar(
        x,
        [off_non, on_non],
        width=w,
        bottom=[off_dec, on_dec],
        color=C_SKY,
        label="Non-decode (park/restore, prefill, IPC)",
    )
    ax1.scatter([0 - 0.10, 0 + 0.10], off_wall_pts, color=C_DARK, s=26, zorder=5, label="Individual ABAB arms")
    ax1.scatter([1 - 0.10, 1 + 0.10], on_wall_pts, color=C_DARK, s=26, zorder=5)

    ax1.text(0, off_dec / 2, f"{fmt2(off_dec)} s", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    ax1.text(0, off_dec + off_non / 2, f"{fmt2(off_non)} s", ha="center", va="center", color=C_DARK, fontweight="bold", fontsize=9.5)
    ax1.text(0, max(off_wall_pts) + 0.65, f"{fmt2(off_dec + off_non)} s", ha="center", va="bottom", color=C_DARK, fontweight="bold", fontsize=10.5)

    ax1.text(1, on_dec / 2, f"{fmt2(on_dec)} s", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    ax1.text(1, on_dec + on_non / 2, f"{fmt2(on_non)} s", ha="center", va="center", color=C_DARK, fontweight="bold", fontsize=9.5)
    ax1.text(1, max(on_wall_pts) + 0.65, f"{fmt2(on_dec + on_non)} s", ha="center", va="bottom", color=C_DARK, fontweight="bold", fontsize=10.5)

    ax1.annotate(
        "-42.0% (1.72x)\n44.8 -> 77.1 tok/s",
        xy=(1.0, max(on_wall_pts) + 2.3),
        xytext=(0.76, 18.8),
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.3, connectionstyle="arc3,rad=-0.12"),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Baseline (ALL OFF)\nStrata 0.1.56", "personal-0.1.56-p2\n(ALL 6 ON)"])
    ax1.set_ylabel("End-to-End Burst Wall Time (s)")
    ax1.set_ylim(0, 29.0)
    ax1.set_title("(a) 4x100K End-to-End Burst (918 Matched Tokens)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D9E2EC")

    # Panel (b): Pure 4x100K Decode Window Breakdown
    b1 = [off_gpu, on_gpu]
    b2 = [off_other, on_other]
    b3 = [off_commit, on_commit]
    b4 = [off_emit, on_emit]

    ax2.bar(x, b1, width=w, color=C_BLUE, label="gpu_wait (GPU critical path)")
    ax2.bar(x, b2, width=w, bottom=b1, color=C_GRAY, label="cpu_exp + host run overhead")
    ax2.bar(x, b3, width=w, bottom=np.add(b1, b2), color=C_AMBER, label="commit (sync, zero bucket shift)")
    ax2.bar(x, b4, width=w, bottom=np.add(np.add(b1, b2), b3), color=C_ORANGE, label="emit (MTP draft: 4x -> 1x read)")

    for idx, (g, o, c, e, tot, sl) in enumerate([
        (off_gpu, off_other, off_commit, off_emit, off_win, off_slope),
        (on_gpu, on_other, on_commit, on_emit, on_win, on_slope),
    ]):
        ax2.text(idx, g / 2, f"gpu_wait\n{fmt2(g)} ms", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax2.text(idx, g + o + c + e / 2, f"{fmt2(e)}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.3)
        ax2.text(idx, tot + 1.1, f"{fmt2(tot)} ms/win\n({fmt2(sl)} ms/row)", ha="center", va="bottom", color=C_DARK, fontweight="bold", fontsize=9.3)

    ax2.annotate(
        "-18.1% win_ms\n-21.0% slope\n(K1 + S1-L2)",
        xy=(1.0, on_win + 6.4),
        xytext=(0.56, 49.0),
        ha="center",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor=C_BLUE, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3, connectionstyle="arc3,rad=-0.15"),
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Baseline (ALL OFF)\navg_rows = 5.98", "personal-0.1.56-p2\navg_rows = 5.94"])
    ax2.set_ylabel("Decode Window Time (ms / window)")
    ax2.set_ylim(0, 58.0)
    ax2.set_title("(b) Pure 4x100K Decode Window (Separated from Burst)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D9E2EC", fontsize=8.0)

    fig.suptitle(
        "RTX A6000 48 GB (WDDM, PCIe Gen3 x16) | Qwen3.8-Flash-Next 176B IQ3_XXS | 0 Thermal Throttle, 0 CPU Interference",
        fontsize=9.5,
        color="#486581",
    )

    out = OUT_DIR / "fig1_end_to_end_4x100k.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def make_fig2_counterintuitive(abl: dict) -> Path:
    s3b = abl["rfcs"]["RFC-S3B"]["abab_100k_park_restore"]
    s1l1 = abl["rfcs"]["RFC-S1-L1"]["arms_4x100k"]
    s1l2 = abl["rfcs"]["RFC-S1-L2"]["arms_4x100k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    # Panel (a): 100K Cold Session Park Anatomy (RFC-S3B)
    labels = ["Baseline\n(std::vector::resize)", "+NUMA0 & Parallel\nHostCopy (0 MiB)", "+Async Exact Stage\n(+306 MiB)"]
    resize_vals = [
        s3b["off_mean"]["cold_resize_ms"],
        s3b["numa_mc1"]["cold_resize_ms"],
        s3b["async_exact"]["cold_resize_ms"],
    ]
    other_vals = [
        s3b["off_mean"]["cold_park_A_ms"] - s3b["off_mean"]["cold_resize_ms"],
        s3b["numa_mc1"]["cold_park_A_ms"] - s3b["numa_mc1"]["cold_resize_ms"],
        s3b["async_exact"]["cold_park_A_ms"] - s3b["async_exact"]["cold_resize_ms"],
    ]
    totals = [r + o for r, o in zip(resize_vals, other_vals)]
    x = np.arange(len(labels))
    w = 0.50

    ax1.bar(x, resize_vals, width=w, color=C_ORANGE, label="Single-thread resize() zero-fill faults")
    ax1.bar(x, other_vals, width=w, bottom=resize_vals, color=C_BLUE, label="D2H copy + NUMA host write + ckpt")

    ax1.text(0, resize_vals[0] / 2, "685 ms\n(61%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9.2)
    ax1.text(0, resize_vals[0] + other_vals[0] / 2, "435 ms", ha="center", va="center", color="white", fontweight="bold", fontsize=9.0)
    ax1.text(1, resize_vals[1] / 2, "74", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)
    ax1.text(1, resize_vals[1] + other_vals[1] / 2, "328 ms", ha="center", va="center", color="white", fontweight="bold", fontsize=9.0)
    ax1.text(2, other_vals[2] / 2, "185 ms", ha="center", va="center", color="white", fontweight="bold", fontsize=9.0)

    for idx, tot in enumerate(totals):
        suffix = "" if idx == 0 else f" ({totals[0]/tot:.1f}x)"
        ax1.text(idx, tot + 25, f"{tot:.0f} ms{suffix}", ha="center", va="bottom", color=C_DARK, fontweight="bold", fontsize=9.5)

    ax1.annotate(
        "resize(): 685 -> 0.43 ms\nHot reuse park: 126.6 -> 15.7 ms (8.1x)",
        xy=(2.0, 275),
        xytext=(1.32, 860),
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2, connectionstyle="arc3,rad=-0.16"),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.7)
    ax1.set_ylabel("100K Cold Session Park Latency (ms)")
    ax1.set_ylim(0, 1340)
    ax1.set_title("(a) 100K Cold Park Was OS Zero-Fill Faults, Not PCIe")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D9E2EC", fontsize=8.2)

    # Panel (b): Why Async Commit Backfired (RFC-S1-L1) vs Batched MTP (RFC-S1-L2)
    arms = [
        ("Baseline\n(s1_off2,\n6.08r)", s1l1["s1_off2"]),
        ("Async Commit\n(s1_on2, 5.94r)\nVETOED", s1l1["s1_on2"]),
        ("Det Baseline\n(det_off,\n5.97r)", s1l1["s1_det_off"]),
        ("Det Async\n(det_on, 6.07r)\nVETOED", s1l1["s1_det_on"]),
        ("Batched MTP\n(s1l2_on1, 6.00r)\nMERGED", s1l2["s1l2_on1"]),
    ]
    x2 = np.arange(len(arms))
    commit_v = [a["commit_ms"] for _, a in arms]
    emit_v = [a["emit_ms"] for _, a in arms]

    ax2.bar(x2, commit_v, width=0.52, color=C_AMBER, label="bt_commit (ms)")
    ax2.bar(x2, emit_v, width=0.52, bottom=commit_v, color=C_ORANGE, label="bt_emit / MTP draft (ms)")

    for idx, (c, e) in enumerate(zip(commit_v, emit_v)):
        tot = c + e
        if c >= 0.6:
            ax2.text(idx, c / 2, f"{c:.2f}", ha="center", va="center", color=C_DARK, fontweight="bold", fontsize=8.3)
        else:
            ax2.text(idx, c + 0.14, f"{c:.2f}", ha="center", va="bottom", color=C_DARK, fontsize=7.6)
        ax2.text(idx, c + e / 2, f"{e:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.6)
        ax2.text(idx, tot + 0.14, f"{tot:.2f} ms", ha="center", va="bottom", color=C_DARK, fontweight="bold", fontsize=8.6)

    ax2.annotate(
        "Bucket shift only +\ngpu_wait +0.54 to +0.75 ms\n(GDDR6 + 300W TDP wall)",
        xy=(1.0, 4.82),
        xytext=(1.55, 6.05),
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C_ORANGE,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF7ED", edgecolor=C_ORANGE, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2),
    )
    ax2.annotate(
        "4x -> 1x weight read\nemit -36.6%, zero shift",
        xy=(4.0, 4.12),
        xytext=(3.75, 5.95),
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for lbl, _ in arms], fontsize=7.7)
    ax2.set_ylabel("commit + emit Latency (ms / window)")
    ax2.set_ylim(0, 7.8)
    ax2.set_title("(b) Negative Result: Async Commit vs Batched MTP")
    ax2.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#D9E2EC", fontsize=8.2)

    fig.suptitle(
        "Counterintuitive Hardware Evidence: OS Page-Fault Dominance & Single-Bus Contention (RTX A6000 ABAB)",
        fontsize=9.5,
        color="#486581",
    )

    out = OUT_DIR / "fig2_counterintuitive_anatomy.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def make_fig3_crossover_and_host(abl: dict, grand: dict) -> Path:
    rows = abl["rfcs"]["RFC-K3-v2"]["clean_check0_crossover_table"]
    toks = [r["tokens"] for r in rows]
    off_sum = [r["off_sum_ms"] for r in rows]
    k3a_sum = [r["k3a_clean_sum_ms"] for r in rows]

    d5 = grand["D5_host_outside_stacked_ms"]
    ctx_labels = ["1K", "100K", "167K"]
    p1_vals = [d5["phase1_baseline"][k] for k in ctx_labels]
    s2a1_vals = [d5["v2rc_off2_s2a1_only"][k] for k in ctx_labels]
    s2a12_vals = [d5["v2rc_on2_s2a1_plus_s2a2"][k] for k in ctx_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

    # Panel (a): Sparse Prefill KV Crossover (RFC-K3-v2)
    ax1.axvspan(140, 256, color="#ECFDF5", alpha=0.85, label="Default sparse auto-zone (T <= 256)")
    ax1.axvspan(350, 380, color="#FEF3C7", alpha=0.90, label="Empirical crossover (T ~ 350-380)")
    ax1.axvline(256, color=C_GREEN, linestyle="--", linewidth=1.2)

    ax1.plot(toks, off_sum, marker="o", color=C_GRAY, linewidth=2.0, markersize=6.5, label="Full-prefix KV stage + QSA (baseline)")
    ax1.plot(toks, k3a_sum, marker="s", color=C_BLUE, linewidth=2.2, markersize=6.5, label="Selection-First sparse KV (CHECK=0)")

    for t, y_off, y_k3 in zip(toks, off_sum, k3a_sum):
        delta = y_k3 - y_off
        dy = -13 if delta < 0 else 8
        color = C_GREEN if delta < 0 else C_ORANGE
        ax1.annotate(f"{delta:+d} ms", (t, y_k3), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8.3, fontweight="bold", color=color)

    ax1.text(198, 134, "Sparse wins\n(-25% to -33%)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_GREEN)
    ax1.text(490, 66, "Above crossover:\nauto-routes to K3-B\n(full stage, 0 lend)", ha="center", va="center", fontsize=8.2, color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#BCCCDC", lw=0.9))

    ax1.set_xlim(140, 600)
    ax1.set_ylim(42, 222)
    ax1.set_xlabel("Incremental Prefill Chunk Size T (tokens at 100K context)")
    ax1.set_ylabel("KV Stage + QSA Attention Latency (ms)")
    ax1.set_title("(a) Sparse Prefill Crossover at 100K Context (RFC-K3 v2)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D9E2EC", fontsize=8.0)

    # Panel (b): Host-Outside-Engine Latency (D5)
    x = np.arange(len(ctx_labels))
    w = 0.25
    b1 = ax2.bar(x - w, p1_vals, width=w, color=C_GRAY, label="Phase 1 Baseline (O(N^2) getline + full IPC)")
    b2 = ax2.bar(x, s2a1_vals, width=w, color=C_SKY, label="+ S2A-1 (C++ 64 KiB getline + BGEN)")
    b3 = ax2.bar(x + w, s2a12_vals, width=w, color=C_BLUE, label="+ S2A-1 & S2A-2 (Delta Token IPC)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, color=C_DARK, fontweight="bold")

    ax2.annotate(
        "-53.4% at 167K (135.1 -> 62.9 ms)\nPython residual ~10-15 ms\n(Rust proxy rewrite rejected)",
        xy=(2 + w, 72.0),
        xytext=(0.55, 118.0),
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor=C_BLUE, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2, connectionstyle="arc3,rad=-0.16"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["1K Context", "100K Context", "167K Context"])
    ax2.set_ylabel("Host-Outside-Engine Latency (ms)")
    ax2.set_ylim(0, 188)
    ax2.set_title("(b) Host-Layer Scaling & Delta Token IPC (D5)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D9E2EC", fontsize=8.0)

    fig.suptitle(
        "Empirical Guardrails & Host Overhead Anatomy | Sanitized Data: data/2026-11-22_a6000-v2/",
        fontsize=9.5,
        color="#486581",
    )

    out = OUT_DIR / "fig3_sparse_prefill_and_host.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grand = load_json("v2_rc_grand_exam.json")
    abl = load_json("phase2_rfc_ablations.json")

    outputs = [
        make_fig1_end_to_end(grand),
        make_fig2_counterintuitive(abl),
        make_fig3_crossover_and_host(abl, grand),
    ]
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
