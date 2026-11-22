# First-Principles Hardware Anatomy and Full-Stack Optimization on RTX A6000 (`v0.2`)

This report documents my November 2026 second-phase investigation into long-context sparse MoE inference on heterogeneous workstation hardware. Starting from measured physical hardware ceilings rather than parameter tuning, I audited every stage of the inference pipeline across five workload regimes, isolated true critical-path bottlenecks through closed-profile ABAB ablations, implemented six opt-in engine/host optimizations (plus one honest negative result that I vetoed and excluded), and verified the integrated `personal-0.1.56-p2` release in an end-to-end D1–D6 benchmark.

- **Public source release**: [`YusenZhang0601/Strata` tag `personal-0.1.56-p2`](https://github.com/YusenZhang0601/Strata/tree/personal-0.1.56-p2) (compiled commit `a1c2c904139d5c6a028750141d7a75f35253b76a`, `personal/stable` docs HEAD `0d5725c9657d625b85c4457f4723b9461c7f2232`, built on the Strata `0.1.56` baseline; all six optimizations are opt-in and default OFF).
- **Sanitized datasets**: [`data/2026-11-22_a6000-v2/`](../data/2026-11-22_a6000-v2/v2_rc_grand_exam.json) (`hardware_ceilings.json`, `phase2_rfc_ablations.json`, `v2_rc_grand_exam.json`, `falsified_registry.json`).
- **Offline verification**: `python -B -m unittest discover -s tests -v` and `python -B examples/verify_results.py`.

---

## 1. Hardware Conditions and Measured Physical Ceilings

All measurements in this report were collected under exclusive GPU ownership on a single workstation with zero thermal throttling (`0` NVML thermal slowdown hits, `temp_max <= 74 °C` in decode and `<= 83 °C` in cold prefill) and zero CPU interference (`0` non-engine CPU spikes):

| Component | Measured Condition / Physical Ceiling | Evidence |
| --- | --- | --- |
| GPU | NVIDIA RTX A6000 48 GB (`sm_86`, WDDM driver 597.30, 300 W TDP) | Measured |
| VRAM Read / Nominal | **713.0 GB/s** sustained read (`768.1 GB/s` nominal); CUDA P2 active clock **7601 MHz** (idle tail drops to 405 MHz) | Measured |
| GPU FP32 Compute | **38.7 TFLOPS** | Measured |
| Host & Interconnect | 2× AMD EPYC 7542 (64C/128T, 2 NUMA nodes, 256 GiB DDR4-3200), **PCIe Gen3 x16** pinned **12.57 GB/s H2D / 13.21 GB/s D2H**; WDDM bidirectional sum **12.84 GB/s** (no full-duplex overlap) | Measured |
| Host RAM Bandwidth | NUMA0 DDR4-3200: **120 GB/s** (30 threads) / **19.8 GB/s** (1 thread) | Measured |
| Model & Baseline | `Qwen3.8-Flash-Next` SC117 `IQ3_XXS` (176B total / ~6B active, 48 layers = 36 GDN + 12 QSA, 512 experts top-10, 262K context, INT8 KV); Strata `0.1.56` (`personal-0.1.56-p1`, 31 GiB / 19,379 VRAM expert slots, `--kv-resident 32768`, `--batch 4 --batch-mtp`) | Measured |

> **Measurement guardrails**: (1) Because NVIDIA CUDA P2 locks RTX A6000 memory at `7601 MHz` even at 42 °C and idle tails fall to `405 MHz`, I validate thermal throttling strictly via NVML thermal flags, `temp_max >= 87 °C`, or `mem_mhz_p50 < 7500`. (2) Because decode follows a linear law `win_ms ~= 6.64 + slope * avg_rows`, comparing raw `win_ms / avg_rows` across arms with different `avg_rows` introduces intercept bias; I always compare exact matched-row pairs or intercept-adjusted `(win_ms - 6.64) / avg_rows` (implemented in `moe_residency/ledger_v2.py`). (3) Serial per-family profiler views (`STRATA_VERIFY_PROFILE=1`) disable cross-stream forks and serve only as clues; critical-path claims require closed-profile ABAB ablations.

---

## 2. Phase 1: Five-Scenario Utilization & Root-Cause Anatomy

Across 44 exclusive baseline arms (and source-level kernel/host audits), I mapped every bottleneck to one of three classes: **physical wall**, **conditional model wall**, or **engineering waste**.

1. **Solo 100K Decode (`S1`)**: `~19.6 ms/win` at `avg_rows ~= 3.45` (`47.5%` of the `713 GB/s` byte floor). Pure solo QSA attention is only `0.32–0.61 ms/win`. A prior `7.07 ms` QSA attention attribution turned out to be a profiling-stamp bug (`batch_rec_` omitted stamps 11/12, collapsing `scores + topk + kv-resolve + attention` into one bucket).
2. **Four-Lane 100K Decode (`S2`)**: Baseline `41.92 ms/win` (`avg_rows = 5.98`, intercept-adjusted slope `5.90 ms/row`), frequently touching the 300 W TDP cap with `0` thermal throttle. Three distinct engineering wastes dominated:
   - `QSA_sel_res_attn` degraded **2.48×** to **`9.47 ms/win`** (`seg[5..9]` window-weighted) because `--kv-resident 32768` (`kv_mode=1`) bypassed `QSA_SLOTS` / `BATCH_ATTN_ROWS` and `STRATA_ATTN_DAILY=1` forced per-row `pre75_attn()` fallback;
   - `batch_rec_` disabled `STRATA_SIDE_MIXER=2`, exposing **`3.34 ms/win`** (`GDN ab+z 2.25 ms + QSA q+q-idx 1.09 ms` vs `0.09 ms` solo);
   - `bt_emit` (`2.90 ms/win`) was **96% four-way serial GPU `slot_mtp->draft()`** (reading MTP dense weights + 248,320-vocab `lm_head` four separate times per window), whereas pure host `printf/fflush` was only `0.11 ms/win`.
   - Meanwhile, VRAM-resident expert reads in 4-lane decode measured `0.98 ms/row` (already near the `1.05–1.33 ms/row` physical byte floor), and NUMA0 available host memory dropped to `12.9 GiB` at 4×100K.
3. **Small Incremental Prefill (`S3`, 130–600 tok at 100K ctx)**: 27-stage breakdown showed `dequant 190–263 ms`, `gemm 184–384 ms`, `host grouping 95–115 ms`, constant full-prefix `kv stage 98–99 ms`, `wait copy` only `3–5 ms`, plus `391–473` VRAM expert cache slots lent and refilled (`56–68 ms`) on every small turn.
4. **100K Cold Prefill (`S4`)**: `39.7 s` (`2,425 tok/s`), reaching `~75%` FP32 utilization and riding the 300 W TDP cap — largely a physical compute/power wall.
5. **100K Session Park / Restore (`S5`) & Host Layer (`E6`)**:
   - `STRATA_PARK_TRACE=1` revealed **74–81% (`682–703 ms`) of cold 100K park time was single-threaded `std::vector::resize()` zero-fill page faults**, not host copy (`130–240 ms`). Parking a 14th 100K session exhausted NUMA0 (`0.0 GiB` free), spilling onto NUMA1 and doubling `resize` to **`1,683–1,694 ms`** (cold park `2.10–2.15 s`, restore `+120–160 ms`).
   - Host-outside-engine latency grew from `32.28 ms` (1K) to `135.06 ms` (167K), but Jinja rendering was only `1.28 ms` and warm BPE tokenization was `2.90 ms`. **84% (`~86 ms`) of the linear growth came from Python `','.join` ASCII serialization (`45.75 ms`) plus Pipe/C++ `getline_fd` $O(N^2)$ scanning (`~40.4 ms` residual)**.

---

## 3. Phase 2: Seven Opt-In RFCs (6 Merged, 1 Honest Negative Result)

Each RFC was built in an isolated worktree on the `0.1.56` baseline, validated in isolated ABAB benchmarks on RTX A6000, and audited against correctness and VRAM safety gates (`mem_free >= 2,048 MiB`, zero expert skipping, zero precision drop):

| RFC | Mechanism | Isolated ABAB Result (Measured) | Extra VRAM | Verification & Status |
| --- | --- | --- | ---: | --- |
| **RFC-K1** | `kv_mode=1` QSA slot-group batching + `attn_chunk_slots_pre75_kernel` (`ATTN_DAILY=1` compatible) + batch `SIDE_MIXER=2` | Matched 4×100K pair (`win=135, rows=5.44`): `win_ms` **`39.05 -> 33.17 ms` (`-5.88 ms/win`, `-15.1%`)**, `gpu_wait -18.0%`, `(win-6.64)/r` **`5.96 -> 4.88 ms/row` (`-18.1%`)**; `seg[5..9]` `QSA_sel_res_attn` **`9.47 -> 2.96 ms` (`-68.7%`)**; no-MTP 4-lane slope `5.83 -> 4.86 ms/row` | **`0 MiB`** | `qsa_slots_parity` **16/16 `memcmp` bitwise**; `10/10` acceptance; **Merged** |
| **RFC-S1-L1** | Standalone `STRATA_BATCH_COMMIT_ASYNC=1` (async `commit_bm_` + multi-stream `slot_mtp`) | **Honest negative result**: `commit 1.54–1.85 -> 0.09–0.12 ms`, but `emit` jumped `2.41–2.92 -> 4.16–4.25 ms` and `gpu_wait` rose `+0.54 to +0.75 ms`; net slope `-0.03 ms/row` (`<0.5%`, noise floor) | `0 MiB` | **Vetoed & Excluded** (GDDR6 bus/300W TDP contention + WDDM `cs_` tail spill) |
| **RFC-S1-L2** | Multi-Slot Batched MTP (`STRATA_BATCH_MTP_FUSED=1`, 4×→1× MTP dense + 248K `lm_head` reads, synchronous commit) | Matched-row reference (`6.08r` vs `6.00/6.14r`): `emit` **`2.92 -> 1.85 ms` (`-1.07 ms/win`, `-36.6%`)**, `commit` (`1.84 ms`) zero bucket-shift, `run` zero tail spill, `(win-6.64)/r` **`5.95 -> 5.74 ms/row` (`-3.7%`)** | **`+14 to +50 MiB`** (staging `0.53 MiB`) | `10/10` acceptance; `MTP%` zero drop; **Merged** |
| **RFC-S3A** | Idle TTL expiry extended to retained `reuse_` buffer (`STRATA_CONV_CACHE_TTL_S`) | Live `TTL=15 s` test expired `162.6 MB` retained `reuse_` cleanly to **`parked=0 bytes=0`** | **`0 MiB`** | `conversation_cache_test` **4,871 checks**; **Merged** |
| **RFC-S3B** | `VirtualAllocExNuma(0)` + `resize_for_overwrite` + persistent parallel `HostCopyPool` + `EXACT_STAGE_VRAM` (fixing 9× `kv_need` over-allocation) | Sync (`0 MiB`): `resize 685 -> 74.1 ms`, cold park `1,121 -> 402 ms`; + async exact (`+306 MiB`, reclaiming **`~1.3 GiB`** vs Phase 1): `resize` **`0.43 ms` (`1,590×`)**, cold park **`182–186 ms` (`6.0×`)**, hot park **`15.7 ms` (`8.1×`)**, hot restore **`183.7 ms` (`-38%`)** | `0` (sync) / **`+306 MiB`** (async exact) | `on_node=256 off_node=0` (`4,848 checks`); 100K park→restore 596-char SHA-256 **byte-exact**; **Merged** |
| **RFC-S2A-1** | C++ `getline_fd` 64 KiB buffered read + incremental `\n` scan + zero-copy `BGEN` + `std::from_chars` | Host-outside median: `1K 32.28->28.46`, `100K 88.18->72.89`, **`167K 135.06 -> 107.66 ms` (`-27.40 ms`, `-20.3%`)** | **`0 MiB`** | **Merged** (engine-side always-on) |
| **RFC-S2A-2** | Text delta token IPC (`ref/keep/h` CRC32+Adler32 + `REF_MISS` recovery) + `unmark_think` range regex | Isolated (without S2A-1): `167K 138.07 -> 72.93 ms` (`-65.14 ms`, `-47.2%`); stacked with S2A-1: **`62.92 ms` (`-53.4%` vs Phase 1)** | **`0 MiB`** (~20 MB host RAM) | `test_ipc_delta.py` **5/5**; real Windows pipe 4-state `REF_MISS` recovery PASS (zero phantom `save_id`); **Merged** |
| **RFC-K3 v2** | Selection-First sparse prefill KV (default `SPARSE_MAX_T=256`, monotonic `overflow_count`) + dedicated small-chunk workspace (`<=768 tok`, zero cache lending) | Clean `CHECK=0` crossover verified at **`T ~= 350–380 tok`**: `183 tok` `kv+qa 104->70 ms (-32.7%)`, `284 tok` `107->80 ms (-25.2%)`, `424–566 tok` flips positive; default `<=256` gets K3-A+B (`-20 to -22.8%`), `257..768` seamlessly uses K3-B zero-borrow (`-10 to -11.5%`, `0` slots lent) | **`+788.3 MiB`** (`no_kv_stage=0`) | `res4096 CHECK=1` **`ovf=0`**; `10/10` acceptance; WDDM 100K 575/596-char **byte-exact**; **Merged** |

---

## 4. Integrated `personal-0.1.56-p2` Full-Stack Grand Exam (D1–D6)

I merged the six approved RFCs (20 commits, 25 files `+3176/-272`, excluding `RFC-S1-L1`) into `personal-0.1.56-p2` (`a1c2c904`, `strata.exe` SHA-256 `9ca1348a2ea895294b80cf0ee882a50273f31c16c5feaffd903d5c5c502baa17`), verified all three native/Python test suites (`qsa_slots_parity` **16/16 bitwise**, `conversation_cache_test` **`on_node=256 off_node=0, 4,877 checks`**, `test_ipc_delta` **5/5**), and ran a full-stack `ALL OFF` vs `ALL 6 ON` evaluation (`data/2026-11-22_a6000-v2/v2_rc_grand_exam.json`).

### 4.1 D1 — Four-Lane 100K Concurrent Decode ABAB (Strictly Matched `918` Completion Tokens)

All four arms generated identical per-lane lengths (`[150, 256, 256, 256] = 918` tokens, `win = 141–142`, `avg_rows = 5.94–6.00`) with `0` thermal throttle and `0` CPU interference. I report **pure decode window speedup** and **end-to-end 4×100K burst speedup** separately:

| Arm | Mode | End-to-End `wall_ms` | Decode-Only `win×win_ms` | Non-Decode (Park/Restore/Prefill/IPC) | `win_ms` | `gpu_wait` | `commit` | `emit` | `(win-6.64)/r` | `mem_free` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2rc_off1` | ALL OFF | 21,403.3 | 5,926.2 | 15,477.0 | 42.03 | 33.27 | 1.82 | 2.90 | 5.898 | 5,769 MiB |
| `v2rc_off2` | ALL OFF | 19,703.7 | 5,935.6 | 13,768.1 | 41.80 | 32.64 | 1.81 | 2.89 | 5.899 | 5,745 MiB |
| **`v2rc_on1`** | **ALL 6 ON** | **12,219.4** | **4,914.6** | **7,304.8** | **34.61** | **27.32** | 1.84 | **1.96** | **4.709** | **4,703 MiB** |
| **`v2rc_on2`** | **ALL 6 ON** | **11,622.5** | **4,830.8** | **6,791.6** | **34.02** | **27.08** | 1.82 | **1.88** | **4.609** | **4,723 MiB** |
| **OFF → ON Mean** | — | **`-42.0%` (`1.72×`)** | **`-17.8%`** | **`-51.8%`** | **`-18.1%` (`-7.60 ms`)** | **`-17.5%`** | `+0.01` | **`-33.7%`** | **`-21.0%`** | **`4,713 MiB` free** |

- **Pure decode gain (K1 + S1-L2)**: `win_ms` fell from **`41.92 -> 34.32 ms` (`-7.60 ms/win`, `-18.1%`)**, raising effective VRAM utilization from `~47.5%` to `~55–58%` of the `713 GB/s` byte floor.
- **Full-stack compound gain**: End-to-end 4×100K burst wall time dropped from **`20.55 -> 11.92 s` (`-42.0%`, `1.72×`, `44.8 -> 77.1 tok/s`)**, driven by `-1.06 s` in decode (`5.93 -> 4.87 s`) plus `-7.58 s` (`-51.8%`) across concurrent park/restore, zero-borrow sparse incremental prefill, and delta IPC.

### 4.2 D2–D6 Highlights & Operational Boundaries

- **D2 (Solo 100K decode)**: Zero regression (`3.85` vs `3.90 ms/row` intercept-adjusted; `draft 2.46` vs `2.46 ms`).
- **D3 (100K hot incremental prefill)**: Pure K3 gain at `155 tok <= 256` measured **`941.0 -> 726.6 ms` (`-214.4 ms`, `-22.8%`, `kv+qa 104->48 ms`, `0` slots lent)**. At `527 tok > 256`, execution seamlessly routed through K3-B full staging with `0` slots lent (note: `t240/t505` baseline arms in D3 included a cold park, so pure K3 prefill accounting uses `t130` and the clean Wave 2 ABAB table).
- **D4 (100K park/restore & NUMA0 capacity note)**: Cold park dropped **`1,056 -> 185 ms` (`-82.5%`, `5.7×`)**, `resize` dropped **`701 -> 0.28–0.52 ms`**, restore `273.3 -> 237.7 ms`. Saturating 8–10 parked 100K sessions reduced `node0_avail` to `5.87 GiB (OFF) / 8.58 GiB (ON)`, confirming that `STRATA_CONV_CACHE_TTL_S=1800` and bounded `--conversation-cache-slots` (`8–12`) are essential to preserve NUMA0 headroom.
- **D5 (167K host layer)**: Stacked `S2A-1 + S2A-2` reduced 167K host-outside latency from **`135.06 -> 62.92 ms` (`-72.14 ms`, `-53.4%`)**, leaving only `~10–15 ms` inside Python (below the `25 ms` threshold for rewriting the HTTP proxy in Rust).
- **D6 (Stacked VRAM, red lines & acceptance)**: With all six optimizations active, steady-state `mem_free` stayed at **`4,703–4,723 MiB`** (`+2,655 MiB` above the `2,048 MiB` safety floor), and end-to-end 3-protocol + vision + 4-way acceptance passed **`10 / 10`**.
- **Three mandatory operational red lines**: (1) leave `STRATA_BATCH_MTP_ATTN_SLOTS` unset so MTP enforces its `<= 32 MiB` auto-guard (forcing `=1` at `window==0` / 262K adds `387–580 MiB` VRAM); (2) keep `STRATA_PREFILL_SPARSE_CHECK=0` in steady state; (3) never enable `STRATA_BATCH_COMMIT_ASYNC=1` (`RFC-S1-L1`, excluded).

---

## 5. Honest Falsified List & Verification Boundaries

Preserving disproved hypotheses prevents wasted engineering cycles (`data/2026-11-22_a6000-v2/falsified_registry.json`):

1. **Standalone `STRATA_BATCH_COMMIT_ASYNC=1` (`RFC-S1-L1`)**: Shifted `commit` into `emit` and spilled `+0.54–0.75 ms` into `gpu_wait` under GDDR6/300W TDP contention (`<0.5%` net). Replaced by `RFC-S1-L2` Multi-Slot Batched MTP.
2. **`STRATA_KV_PREFETCH=1`**: Zero benefit on WDDM PCIe Gen3 x16 (no full-duplex overlap: `12.84 GB/s` bidir sum). Replaced by Selection-First sparse KV (`RFC-K3-A`).
3. **Solo `STRATA_QSA_SPLIT_ATTN=1`**: Pure solo QSA attention is only `0.32–0.61 ms` (`-0.07 ms`, below noise).
4. **Jinja incremental cache & Rust HTTP proxy rewrite**: 167K Jinja is `1.28 ms`; after `S2A-1+2`, Python accounts for only `~10–15 ms`.
5. **Cross-layer persistent Megakernel on RTX A6000 (`sm_86`)**: 6 MB L2 and 8.3 MB total shared memory across 84 SMs cannot hold multi-gigabyte HBM MoE weights on-chip.
6. **Prior v0.1 falsifications retained**: Multi-lane time-slicing myth, `STRATA_HC_PERSIST=1` / deeper HC prefetch (`STRATA_HC_DEEP`).

**Explicit scope disclosures**:
- All benchmarks in this report were measured on the Strata **`0.1.56`** personal fork baseline (`personal-0.1.56-p2 @ a1c2c904`); upstream `Niko1221/Strata` `main` ports and compile checks are tracked separately.
- Unlike `p1`, I did **not** re-run the 24,574-line `kvq` teacher-forcing harness for `p2`; numerical integrity was verified via kernel/continuation bitwise parity (`qsa_slots_parity` 16/16 `memcmp`, 100K park→restore 596-char SHA-256, 100K WDDM 575/596-char byte-exact, `res4096 ovf=0`, CRC32+Adler32 pipe recovery) plus `10/10` acceptance.
- On my live reference A6000 server, I preserved the existing A+ Python `serve/` layer intact to maintain `10/10` tool/Responses parity; accordingly, five C++/CUDA optimizations plus always-on engine-side `S2A-1` are active live, while `STRATA_IPC_DELTA` (`S2A-2`) remains compiled and ready in the engine pending clean A+ `serve/` integration.
