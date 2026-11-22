"""Recompute the reported figures from the public, allowlisted measurements."""

import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_residency.ledger_v2 import classify_arm_validity, recompute_v2_rc_d1_summary


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def main():
    provenance = load("provenance.json")
    source = provenance["counter_source"]
    assert hashlib.sha256((ROOT / source["file"]).read_bytes()).hexdigest() == source["sha256"]
    rows = load("gongwei-single-frontier.json")["jobs"]
    expected = {"confirm-grow": (172.895, 153.210, 51.8), "confirm-stream64": (175.664, 155.337, 54.4)}
    for arm, target in expected.items():
        samples = [row for row in rows if row["arm"] == arm]
        assert len(samples) == 5
        assert all(row["cold_usage"]["completion_tokens"] == 1024 for row in samples)
        decode = [float(re.search(r"generated in \d+ ms \(([\d.]+) tok/s\)", row["cold_engine_line"])[1])
                  for row in samples]
        actual = (round(statistics.median(row["cold_wall_seconds"] for row in samples), 3),
                  round(statistics.median(row["cold_first_output_seconds"] for row in samples), 3),
                  statistics.median(decode))
        assert actual == target, (arm, actual)
        print(arm, "median wall / first output / decode", actual)

    arms = load("a6000-budget-B1-summary.json")
    expected = {"AUTO_DYNAMIC": (6.406, 17.079, 22.484, 1.498),
                "GPU28_DYNAMIC": (7.156, 17.359, 21.407, 8.896),
                "GPU25_DYNAMIC": (6.469, 17.156, 21.375, 11.937),
                "GPU28_STATIC": (7.219, 17.719, 22.297, 8.978),
                "GPU28_PERIODIC64": (6.797, 17.390, 21.844, 9.017)}
    for name, target in expected.items():
        arm = arms[name]
        cases = arm["cases"]
        actual = tuple(round(cases[case]["wall_s"], 3) for case in ("warm_solo", "warm_four", "warm_four_return"))
        actual += (round(arm["min_free_gib"], 3),)
        assert actual == target, (name, actual)
        assert cases["warm_four"]["output_tokens"] == cases["warm_four_return"]["output_tokens"] == 2048
        assert all(value is None for value in cases["warm_four"]["first_text_s"])
        native_shares = [float(re.search(r"VRAM [\d.]+ \(([\d.]+)%\)", line)[1])
                         for line in arm["batch_summaries"]]
        print(name, actual, "native VRAM shares", native_shares)
    print("observed extra free GiB", round(arms["GPU25_DYNAMIC"]["min_free_gib"] - arms["AUTO_DYNAMIC"]["min_free_gib"], 3))

    for filename in ("a6000-continuous-CB3-summary.json", "a6000-continuous-CB4-summary.json"):
        for name, row in load(filename).items():
            assert row["requests"] == 24 and row["long_output_tokens"] == 7188
            assert row["load_summary"]["errors"] == 0
            print(name, "aux queue / first body / total", row["aux_queue_max_s"], row["aux_first_output_max_s"], row["wall_s"])
    for row in load("a6000-accounting-proof.json")["cold_proofs"]:
        legs = row["native_legs"]
        assert row["external_reused"] + row["fresh_input"] == row["input_tokens"]
        assert sum(leg["prompt_read"] for leg in legs) == row["fresh_input"]
        assert abs(sum(leg["prompt_ms"] for leg in legs) - row["api_prompt_ms"]) < 0.01
        assert row["native_prompt_ms"] == row["api_prompt_ms"]
        print(row["case"], "original cache / fresh / prompt ms", row["external_reused"], row["fresh_input"], row["api_prompt_ms"])

    # v0.2 A6000 first-principles & personal-0.1.56-p2 checks
    v2_exam = load("2026-11-22_a6000-v2/v2_rc_grand_exam.json")
    v2_abl = load("2026-11-22_a6000-v2/phase2_rfc_ablations.json")
    d1_sum = recompute_v2_rc_d1_summary(v2_exam)
    assert d1_sum["off_win_ms"] == 41.915 and d1_sum["on_win_ms"] == 34.315
    assert d1_sum["win_reduction_pct"] == 18.1 and d1_sum["slope_reduction_pct"] == 21.0
    assert d1_sum["emit_reduction_pct"] == 33.7
    assert d1_sum["off_wall_s"] == 20.55 and d1_sum["on_wall_s"] == 11.92 and d1_sum["wall_speedup_x"] == 1.72
    for arm_name, row in v2_exam["D1_four_lane_100k_abab"].items():
        ok, reasons = classify_arm_validity(row)
        assert ok, (arm_name, reasons)
    assert v2_abl["rfcs"]["RFC-S1-L1"]["excluded_from_release"] is True
    assert min(v2_exam["D6_vram_floor_and_redlines"]["all_6_on_mem_free_min_mib"]) >= 4700
    print(
        "v0.2 D1 pure decode / end-to-end:",
        f"{d1_sum['off_win_ms']}->{d1_sum['on_win_ms']} ms (-{d1_sum['win_reduction_pct']}%)",
        f"/ {d1_sum['off_wall_s']}->{d1_sum['on_wall_s']} s ({d1_sum['wall_speedup_x']}x)",
    )
    print("All published numerical checks passed.")


if __name__ == "__main__":
    main()
