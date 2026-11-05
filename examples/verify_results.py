"""Recompute the reported figures from the public, allowlisted measurements."""

import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


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
    print("All published numerical checks passed.")


if __name__ == "__main__":
    main()
