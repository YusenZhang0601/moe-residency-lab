# MoE Residency Lab

Hardware-specific MoE inference experiments on a 16 GiB workstation and an RTX A6000, with routing statistics, residency analysis, historical deployment code and reproducible measurements.

**v0.1 documents experiments from October 30 to November 4, 2026.** It includes a CPU-only offline replay tool. The hardware reports describe historical configurations and their measured tradeoffs.

## Start here

- [中文总览](README.zh-CN.md)
- [16 GiB workstation case study](docs/gongwei.md)
- [48 GB A6000 case study](docs/a6000.md)
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

On the A6000, five fixed-order arms used four independent histories of 64K/64K/16K/16K tokens. Each continuation generated 512 tokens. The 25 GiB dynamic expert arm took 17.156 s for its first four-way round, against 17.079 s for auto, while observed minimum free VRAM rose from 1.498 to 11.937 GiB. The experiment supports a useful capacity/latency tradeoff for that workload. See the complete conditions and five arms in the [A6000 report](docs/a6000.md).

On the smaller workstation, five near-full-window samples per arm exposed a different tradeoff. Full-RAM experts with growing KV had median cold-call time 172.895 s and decode 51.8 tok/s. Complement residency with 64K physical KV had 175.664 s and 54.4 tok/s. Decode alone would select a different winner from whole-call latency. See [workstation report](docs/gongwei.md).

The independent benefit of the learned ranking still needs a matched default-profile/learned-profile test on held-out work. The included density rule is a greedy prefix policy. Runtime adaptation, CPU computation and PCIe transfer also affect latency.

## Included code

- `moe_residency/` reads STRC, compares static budget coverage and encodes STRP.
- `reference/routing_counter_20261104.py` preserves the historical model-bound Windows counter byte-for-byte. Its original SHA is recorded in `data/provenance.json`. Review its integration and limitations before any deployment.
- `patches/` contains source changes with their exact bases and validation status.
- `data/` contains selected measurements without private prompts, credentials or network configuration.

The offline encoder uses exact rational comparisons and modulo indexing. The historical counter uses floating-point comparisons and a 512-expert geometry. These adaptations are covered by tests and documented in [code provenance](docs/code-and-upstream.md).

## Licence and attribution

MIT for this project. [Third-party notices](THIRD_PARTY_NOTICES.md) retain Strata's licence and attribution. Strata provides the inference engine and native adaptive cache mechanisms. Model weights and runtime binaries have their own licences and are acquired separately from their publishers.
