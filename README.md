# MoE Residency Lab

Hardware-specific MoE inference experiments on a 16 GiB workstation and an RTX A6000, with routing statistics, residency analysis, historical deployment code and reproducible measurements.

**v0.2 (November 2026) adds a first-principles hardware ceiling ledger, six opt-in engine/host optimizations (plus one honest falsified negative result), and an integrated RTX A6000 full-stack benchmark (`personal-0.1.56-p2`).** **v0.1 documents experiments from October 30 to November 4, 2026** and includes a CPU-only offline replay tool. All reports state exact hardware, software baselines, failures, and verification boundaries.

## Start here

- [中文总览](README.zh-CN.md)
- **[v0.2 A6000 first-principles hardware anatomy & full-stack optimization (EN)](docs/a6000-first-principles-v2.md)** / **[中文深研报告](docs/a6000-first-principles-v2.zh-CN.md)**
- [v0.1 16 GiB workstation case study](docs/gongwei.md)
- [v0.1 48 GB A6000 case study](docs/a6000.md)
- [Expert statistics and residency allocation](docs/expert-residency.md)
- [Measurement method and limitations](docs/methodology.md)
- [Code provenance and upstream contributions](docs/code-and-upstream.md)

## Run an offline example

Python 3.10 or newer, using the standard library. Run from the repository root.

```sh
python examples/make_fixture.py demo-output
python -m moe_residency --ledger demo-output/counts.bin --layout demo-output/layout.json --budgets-gib 0.008 0.016 0.032 --export-profile demo-output/profile.bin
python examples/verify_results.py
python -m unittest discover -s tests -v
```

The fixture contains synthetic aggregate counts. The command compares frequency and byte-density rankings. Its budgets use GiB, where 1 GiB equals 1,073,741,824 bytes. Output creation is exclusive, so use a new destination for each export. The CLI has no network, engine or process controls. Remove `demo-output` after inspecting the example.

For your own ledger, supply a layout containing its SHA-256, model identity, capture coverage and one expert-blob size per layer. STRC v1 carries geometry and counts; the layout binds those bytes to their declared model. A baseline STRP profile can be compared with `--baseline-profile`.

## What the measurements support

### v0.2 RTX A6000 first-principles full-stack study (November 2026)

On the RTX A6000 48 GB (`sm_86`, WDDM, PCIe Gen3 x16, 2× AMD EPYC 7542, `Qwen3.8-Flash-Next` SC117 `IQ3_XXS`, 262K INT8 KV, 31 GiB VRAM expert cache, Strata `0.1.56`), I measured physical ceilings (`713 GB/s` VRAM read, `12.57/13.21 GB/s` PCIe Gen3 x16 pinned, `120 GB/s` NUMA0 DDR4), audited five workload regimes across 44 arms, implemented six opt-in optimizations in [`YusenZhang0601/Strata@personal-0.1.56-p2`](https://github.com/YusenZhang0601/Strata/tree/personal-0.1.56-p2) (commit `a1c2c904`, all default OFF), and verified the integrated build with zero thermal throttling, zero CPU interference, and `4,703–4,723 MiB` minimum free VRAM:

- **Four-lane 100K pure decode window (K1 + S1-L2, matched `918` completion tokens)**: `41.92 -> 34.32 ms/win` (**`-18.1%`**, intercept-adjusted slope `5.90 -> 4.66 ms/row`, **`-21.0%`**); solo 100K decode showed zero regression.
- **Four-lane 100K end-to-end burst wall time (all 6 optimizations stacked: park/restore + hot incremental prefill + delta IPC + 918-tok decode)**: `20.55 -> 11.92 s` (**`-42.0%`**, **`1.72×`**, `44.8 -> 77.1 tok/s`; `-17.8%` in decode plus `-51.8%` in non-decode stages). I report pure decode and end-to-end burst speedups separately.
- **100K incremental prefill (`<= 256` tok)**: `941.0 -> 726.6 ms` (**`-22.8%`**, `0` cache slots lent; `257..768` tok uses dedicated small-chunk workspace with `0` slots lent; clean `CHECK=0` sparse-KV crossover measured at `T ~= 350–380 tok`).
- **100K cold session park & restore**: Cold park `1,056 -> 185 ms` (**`-82.5%`**, **`5.7×`**), `resize` zero-fill faults `701 -> 0.28–0.52 ms`, hot reuse park `126.5 -> 15.7 ms` (**`8.1×`**), hot restore `295 -> 184–238 ms`, with `596`-char post-restore byte-exact parity.
- **167K host-outside-engine latency (`S2A-1 + S2A-2`)**: `135.06 -> 62.92 ms` (**`-53.4%`**), with real Windows pipe 4-state `REF_MISS` recovery verified.
- **Honest falsified list**: I vetoed and excluded standalone `STRATA_BATCH_COMMIT_ASYNC=1` (`RFC-S1-L1`: GDDR6/300W TDP contention + WDDM tail spill gave `<0.5%` net), and also falsified `STRATA_KV_PREFETCH=1`, solo `STRATA_QSA_SPLIT_ATTN=1`, Jinja incremental caching, Rust proxy rewrite, and cross-layer `sm_86` Megakernel. See [the v0.2 report](docs/a6000-first-principles-v2.md) and [`data/2026-11-22_a6000-v2/`](data/2026-11-22_a6000-v2/v2_rc_grand_exam.json).

### v0.1 historical case studies (October–November 4, 2026)

On the A6000, five fixed-order arms used four independent histories of 64K/64K/16K/16K tokens. Each continuation generated 512 tokens. The 25 GiB dynamic expert arm took 17.156 s for its first four-way round, against 17.079 s for auto, while observed minimum free VRAM rose from 1.498 to 11.937 GiB. The experiment supports a useful capacity/latency tradeoff for that workload. See the complete conditions and five arms in the [A6000 report](docs/a6000.md).

On the smaller workstation, five near-full-window samples per arm exposed a different tradeoff. Full-RAM experts with growing KV had median cold-call time 172.895 s and decode 51.8 tok/s. Complement residency with 64K physical KV had 175.664 s and 54.4 tok/s. Decode alone would select a different winner from whole-call latency. See [workstation report](docs/gongwei.md).

The independent benefit of the learned ranking still needs a matched default-profile/learned-profile test on held-out work. The included density rule is a greedy prefix policy. Runtime adaptation, CPU computation and PCIe transfer also affect latency.

## Included code

- `moe_residency/analysis.py` reads STRC, compares static budget coverage and encodes STRP; `moe_residency/ledger_v2.py` provides intercept-adjusted decode slope, CUDA P2 thermal-validity gates, sparse-KV crossover solver, and v0.2 recomputation helpers.
- `reference/routing_counter_20261104.py` preserves the historical model-bound Windows counter byte-for-byte. Its original SHA is recorded in `data/provenance.json`. Review its integration and limitations before any deployment.
- `patches/` contains v0.1 source changes with their exact bases and validation status; v0.2 source optimizations live in [`YusenZhang0601/Strata@personal-0.1.56-p2`](https://github.com/YusenZhang0601/Strata/tree/personal-0.1.56-p2).
- `data/` contains sanitized measurements (`data/*.json` for v0.1 and `data/2026-11-22_a6000-v2/*.json` for v0.2) without private prompts, credentials, hostnames, or network configuration.

The offline encoder uses exact rational comparisons and modulo indexing. The historical counter uses floating-point comparisons and a 512-expert geometry. These adaptations are covered by tests and documented in [code provenance](docs/code-and-upstream.md).

## Licence and attribution

MIT for this project. [Third-party notices](THIRD_PARTY_NOTICES.md) retain Strata's licence and attribution. Strata provides the inference engine and native adaptive cache mechanisms. Model weights and runtime binaries have their own licences and are acquired separately from their publishers.
