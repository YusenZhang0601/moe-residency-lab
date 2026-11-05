r"""serve/routing_counter.py - Bounded, fixed-size cumulative expert routing counter & Self-Learning Profile Engine for Strata.

SC117 migration version (D2-compatible on the wire, instance-bound internally).

What changed against the D2 build:
1. Every path, byte table and budget lives in a frozen ``CounterSpec``.  Nothing is
   read from module globals and no function carries a default argument bound to an
   old profile path, so a second model cannot silently reuse the first one's ledger.
2. ``configure(spec)`` installs the spec before ``manager()`` is first used.  A
   second call with a different identity raises instead of mixing ledgers.
3. A ``identity.json`` sidecar binds the binary ledger to the model artifact
   (repo/revision/quant/shard SHA/pack SHA).  ``bind_mode="strict"`` refuses to
   load a ledger whose identity does not match; ``legacy`` keeps D2 behaviour.
4. Saving is serialised on its own lock (the count lock is held only briefly), the
   three outputs are written atomically with retries, and ``dirty`` survives a
   failed write instead of dropping counts.
5. ``get_stats()`` keeps every original field and adds model identity, counter
   scope, static-estimate provenance and an explicit basis note.

On-disk formats are unchanged: STRC v1 (52-byte header + 2 x uint64[L*E]) and
STRP v1 (24-byte header + pairs + table).  A D2 ledger still loads.
"""
from __future__ import annotations

import atexit
import ctypes
import gzip
import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Binary formats (unchanged - STRC v1 / STRP v1)
# --------------------------------------------------------------------------- #
MAGIC = b"STRC"
VERSION = 1

DEFAULT_N_LAYERS = 48
DEFAULT_N_EXPERTS = 512
DEFAULT_TOP_K = 10

# Header: magic(4s), version(I), n_layers(I), n_experts(I), top_k(I),
#         hist_records(Q), live_records(Q), live_flushes(Q), updated_unix_ms(Q) = 52 bytes
HEADER_STRUCT = struct.Struct("<4sIIIIQQQQ")
# STRP v1: "STRP"(4s) + 5x uint32 + L*E x (uint16, uint16) + L*E x int32
STRP_HEADER_STRUCT = struct.Struct("<4s5I")
# Fast record unpacker for k=10: int32 layer, int32 k, 10x int32 expert_ids, 40 pad bytes
RECORD_K10_STRUCT = struct.Struct("<ii10i40x")
RECORD_K10_BYTES = RECORD_K10_STRUCT.size  # 88 bytes

# The D2 deployment's exact byte table, kept only for ``legacy`` compatibility.
LEGACY_LAYER_BLOB_BYTES = (
    1868800, 1305600, 1868800, 1305600, 1868800, 1408000, 1305600, 1305600,
    1868800, 1408000, 1408000, 1305600, 1510400, 1305600, 1408000, 1510400,
    1305600, 1510400, 1868800, 1408000, 1868800, 1510400, 1868800, 1868800,
    1868800, 1408000, 1510400, 1305600, 2329600, 1971200, 2329600, 1766400,
    2329600, 1868800, 2329600, 2176000, 1715200, 2329600, 1971200, 1868800,
    1510400, 1715200, 1971200, 2329600, 2176000, 1715200, 1971200, 2176000,
)

# D2 defaults: only used when nothing configured (legacy mode).
_LEGACY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA_DIR = _LEGACY_ROOT / "data"
LEGACY_COUNTS_BIN = LEGACY_DATA_DIR / "expert-routing-counts.bin"
LEGACY_STATS_JSON = LEGACY_DATA_DIR / "expert-routing-stats.json"
LEGACY_LIVE_PROFILE = LEGACY_DATA_DIR / "expert-profile-a-start.bin"
LEGACY_FROZEN_PROFILE = (_LEGACY_ROOT / "maintenance" / "20261029-three-tier-49-15"
                         / "learned-profile-user-selected-current.bin")
DEFAULT_PIPE_NAME = r"\\.\pipe\strata_routing_counts"
LEGACY_GPU_SLOTS = 11522
LEGACY_RAM_BUDGET = None  # None => all experts resident in RAM


class CounterIdentityError(RuntimeError):
    """The ledger on disk belongs to a different model than the configured spec."""


def _align256(n: int) -> int:
    return (n + 255) // 256 * 256


# --------------------------------------------------------------------------- #
# CounterSpec - everything that used to be a module global
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CounterSpec:
    n_layers: int
    n_experts: int
    top_k: int
    layer_blob_bytes: tuple[int, ...]
    gpu_slots: int
    counts_bin: Path
    stats_json: Path
    profile_path: Path
    ram_budget_bytes: int | None = None      # None => all experts resident
    frozen_profile_path: Path | None = None
    identity: dict = field(default_factory=dict)
    seed_prior_profile: Path | None = None
    bind_mode: str = "legacy"                # "strict" | "legacy"
    coverage_note: str = ""
    state_dir: Path | None = None

    # ---- derived geometry ----
    @property
    def n_pairs(self) -> int:
        return self.n_layers * self.n_experts

    @property
    def max_slot_bytes(self) -> int:
        return max(self.layer_blob_bytes)

    @property
    def gpu_budget_bytes(self) -> int:
        return self.gpu_slots * _align256(self.max_slot_bytes)

    @property
    def total_expert_bytes(self) -> int:
        return sum(self.layer_blob_bytes) * self.n_experts

    @property
    def ram_tier_bytes(self) -> int:
        return self.total_expert_bytes if self.ram_budget_bytes is None else self.ram_budget_bytes

    @property
    def expected_bin_bytes(self) -> int:
        return HEADER_STRUCT.size + 2 * (self.n_pairs * 8)

    @property
    def expected_strp_bytes(self) -> int:
        return STRP_HEADER_STRUCT.size + self.n_pairs * 4 + self.n_pairs * 4

    def counts_struct(self) -> struct.Struct:
        return struct.Struct(f"<{self.n_pairs}Q")

    # ---- constructors ----
    @classmethod
    def legacy(cls) -> "CounterSpec":
        return cls(
            n_layers=DEFAULT_N_LAYERS,
            n_experts=DEFAULT_N_EXPERTS,
            top_k=DEFAULT_TOP_K,
            layer_blob_bytes=LEGACY_LAYER_BLOB_BYTES,
            gpu_slots=LEGACY_GPU_SLOTS,
            counts_bin=LEGACY_COUNTS_BIN,
            stats_json=LEGACY_STATS_JSON,
            profile_path=LEGACY_LIVE_PROFILE,
            ram_budget_bytes=LEGACY_RAM_BUDGET,
            frozen_profile_path=LEGACY_FROZEN_PROFILE,
            identity={},
            seed_prior_profile=None,
            bind_mode="legacy",
            coverage_note="legacy D2 counter (pre-SC117)",
        )

    @classmethod
    def from_pack(cls, pack_dir: Path, **kw) -> "CounterSpec":
        """Read per-layer blob bytes from a pack's native_experts.txt."""
        path = Path(pack_dir) / "native_experts.txt"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: an unfinished pack has no native_experts.txt")
        rows: list[tuple[int, int]] = []
        declared_experts = None
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                import re as _re
                m = _re.search(r"n_expert\s+(\d+)", line)
                if m:
                    declared_experts = int(m.group(1))
                continue
            parts = line.split()
            rows.append((int(parts[0]), int(parts[4])))
        rows.sort()
        layers = [r[0] for r in rows]
        n_layers = kw.pop("n_layers", None) or len(rows)
        if layers != list(range(n_layers)):
            raise ValueError(f"{path}: expected contiguous layers 0..{n_layers - 1}, got {layers[:5]}...")
        blob = tuple(r[1] for r in rows)
        n_experts = kw.pop("n_experts", None) or declared_experts or DEFAULT_N_EXPERTS
        sd = kw.get("state_dir")
        if sd is not None:
            if not kw.get("counts_bin"):
                kw["counts_bin"] = Path(sd) / "expert-routing-counts.bin"
            if not kw.get("stats_json"):
                kw["stats_json"] = Path(sd) / "expert-routing-stats.json"
            if not kw.get("profile_path"):
                kw["profile_path"] = Path(sd) / "expert-profile.bin"
        kw.setdefault("top_k", DEFAULT_TOP_K)
        return cls(n_layers=n_layers, n_experts=n_experts,
                   layer_blob_bytes=blob, **kw)

    @classmethod
    def from_config(cls, section: dict, *, config_dir: Path | None = None) -> "CounterSpec":
        """Build a spec from the config's optional ``routing_counter`` section."""

        def _p(v) -> Path:
            p = Path(v)
            if not p.is_absolute() and config_dir is not None:
                p = Path(config_dir) / p
            return p

        state_dir = _p(section["state_dir"]) if section.get("state_dir") else None
        pack_dir = section.get("pack_dir")
        base = dict(
            gpu_slots=int(section.get("gpu_slots", LEGACY_GPU_SLOTS)),
            ram_budget_bytes=section.get("ram_budget_bytes"),
            top_k=int(section.get("top_k", DEFAULT_TOP_K)),
            identity=dict(section.get("identity") or {}),
            bind_mode=str(section.get("bind_mode", "strict")),
            coverage_note=str(section.get("coverage_note", "")),
            state_dir=state_dir,
            counts_bin=_p(section["counts_bin"]) if section.get("counts_bin") else None,
            stats_json=_p(section["stats_json"]) if section.get("stats_json") else None,
            profile_path=_p(section["profile"]) if section.get("profile") else None,
            frozen_profile_path=_p(section["frozen_profile"]) if section.get("frozen_profile") else None,
            seed_prior_profile=_p(section["seed_prior_profile"]) if section.get("seed_prior_profile") else None,
            n_layers=section.get("n_layers"),
            n_experts=section.get("n_experts"),
        )
        if state_dir is not None:
            if base["counts_bin"] is None:
                base["counts_bin"] = state_dir / "expert-routing-counts.bin"
            if base["stats_json"] is None:
                base["stats_json"] = state_dir / "expert-routing-stats.json"
            if base["profile_path"] is None:
                base["profile_path"] = state_dir / "expert-profile.bin"
        if pack_dir:
            spec = cls.from_pack(_p(pack_dir), **base)
        else:
            blob = tuple(int(x) for x in section["layer_blob_bytes"])
            n_layers = int(base.pop("n_layers") or len(blob))
            n_experts = int(base.pop("n_experts") or DEFAULT_N_EXPERTS)
            spec = cls(n_layers=n_layers, n_experts=n_experts, layer_blob_bytes=blob, **base)
        if spec.counts_bin is None or spec.stats_json is None or spec.profile_path is None:
            raise ValueError("routing_counter section needs counts_bin, stats_json and profile")
        return spec


# --------------------------------------------------------------------------- #
# Profile engine
# --------------------------------------------------------------------------- #
def build_self_learning_profile(counts: list[int], spec: CounterSpec) -> dict:
    """Rank every (layer, expert) by cumulative heat density ``(calls + 1) / blob_bytes``.

    GPU tier keeps strict heat-density descending order; the RAM complement keeps a
    16-expert head guard, sorts its core by ``(layer, expert)`` for physically
    contiguous DDR5 layout, and a 32-expert tail guard at the RAM/SSD cutoff.
    """
    n_pairs = spec.n_pairs
    blob_of = spec.layer_blob_bytes
    gpu_budget = spec.gpu_budget_bytes
    ram_budget = spec.ram_tier_bytes

    heat_order = list(range(n_pairs))
    heat_order.sort(
        key=lambda idx: (
            (counts[idx] + 1.0) / blob_of[idx // spec.n_experts],
            counts[idx],
            -idx,
        ),
        reverse=True,
    )
    gpu_tier: list[int] = []
    ram_tier: list[int] = []
    ssd_tier: list[int] = []
    gpu_bytes = 0
    ram_bytes = 0
    gpu_full = False
    ram_full = False
    for flat_idx in heat_order:
        b = blob_of[flat_idx // spec.n_experts]
        if not gpu_full and gpu_bytes + b <= gpu_budget:
            gpu_tier.append(flat_idx)
            gpu_bytes += b
        else:
            gpu_full = True
            if not ram_full and ram_bytes + b <= ram_budget:
                ram_tier.append(flat_idx)
                ram_bytes += b
            else:
                ram_full = True
                ssd_tier.append(flat_idx)

    if len(ram_tier) > 64:
        tier_2_layout = ram_tier[:16] + sorted(ram_tier[16:-32]) + ram_tier[-32:]
    else:
        tier_2_layout = sorted(ram_tier)
    order = gpu_tier + tier_2_layout + ssd_tier
    assert len(order) == n_pairs, f"Expected {n_pairs} pairs, got {len(order)}"

    pairs_u16 = [0] * (n_pairs * 2)
    table_i32 = [-1] * n_pairs
    for rank, flat_idx in enumerate(order):
        pairs_u16[rank * 2] = flat_idx // spec.n_experts
        pairs_u16[rank * 2 + 1] = flat_idx & (spec.n_experts - 1)
        table_i32[flat_idx] = rank

    hdr = STRP_HEADER_STRUCT.pack(b"STRP", 1, spec.n_layers, spec.n_experts, n_pairs, n_pairs)
    b_pairs = struct.pack(f"<{n_pairs * 2}H", *pairs_u16)
    b_table = struct.pack(f"<{n_pairs}i", *table_i32)
    blob = hdr + b_pairs + b_table
    assert len(blob) == spec.expected_strp_bytes, f"Unexpected STRP size: {len(blob)}"

    out_path = spec.profile_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".bin.tmp")
    tmp_path.write_bytes(blob)
    tmp_path.replace(out_path)
    return {
        "path": str(out_path),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "gpu_experts": len(gpu_tier),
        "ram_experts": len(ram_tier),
        "ssd_experts": len(ssd_tier),
        "gpu_gib": round(gpu_bytes / (1 << 30), 6),
        "ram_gib": round(ram_bytes / (1 << 30), 6),
    }


def _load_profile_tiers(profile_path: Path | None, n_layers: int, n_experts: int,
                        blob_of: tuple[int, ...], gpu_budget: int, ram_budget: int,
                        cache: dict) -> dict | None:
    if profile_path is None:
        return None
    try:
        st = profile_path.stat()
    except OSError:
        return None
    key = (str(profile_path), st.st_mtime_ns, st.st_size, gpu_budget, ram_budget)
    if cache.get("key") == key:
        return cache["val"]
    try:
        blob = profile_path.read_bytes()
        if len(blob) < STRP_HEADER_STRUCT.size or blob[:4] != b"STRP":
            return None
        _magic, _ver, nl, ne, _slots, n = STRP_HEADER_STRUCT.unpack_from(blob, 0)
        if (nl, ne) != (n_layers, n_experts) or len(blob) < STRP_HEADER_STRUCT.size + 4 * n:
            return None
        ranked = [struct.unpack_from("<HH", blob, STRP_HEADER_STRUCT.size + 4 * i) for i in range(n)]
    except Exception:  # noqa: BLE001 - a broken profile just disables the estimate
        return None

    gpu_indices: list[int] = []
    ram_indices: list[int] = []
    ssd_indices: list[int] = []
    gpu_bytes = ram_bytes = ssd_bytes = 0
    gpu_full = ram_full = False
    for layer, expert in ranked:
        if layer >= n_layers or expert >= n_experts:
            continue
        flat_idx = layer * n_experts + expert
        b = blob_of[layer]
        if not gpu_full and gpu_bytes + b <= gpu_budget:
            gpu_indices.append(flat_idx)
            gpu_bytes += b
        else:
            gpu_full = True
            if not ram_full and ram_bytes + b <= ram_budget:
                ram_indices.append(flat_idx)
                ram_bytes += b
            else:
                ram_full = True
                ssd_indices.append(flat_idx)
                ssd_bytes += b

    val = {
        "path": str(profile_path),
        "mtime_unix": round(st.st_mtime, 3),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "gpu_indices": gpu_indices,
        "ram_indices": ram_indices,
        "ssd_indices": ssd_indices,
        "gpu_experts": len(gpu_indices),
        "ram_experts": len(ram_indices),
        "ssd_experts": len(ssd_indices),
        "gpu_gib": round(gpu_bytes / (1 << 30), 6),
        "ram_gib": round(ram_bytes / (1 << 30), 6),
        "ssd_gib": round(ssd_bytes / (1 << 30), 6),
    }
    cache["key"] = key
    cache["val"] = val
    return val


def _tier_breakdown(counts: list[int], tier_info: dict | None) -> dict | None:
    if tier_info is None:
        return None
    total = sum(counts)
    gpu_sel = sum(counts[i] for i in tier_info["gpu_indices"])
    ram_sel = sum(counts[i] for i in tier_info["ram_indices"])
    ssd_sel = sum(counts[i] for i in tier_info["ssd_indices"])
    denom = total if total > 0 else 1

    def pct(x: int) -> float:
        return round(100.0 * x / denom, 4) if total > 0 else 0.0

    return {
        "total_route_selections": total,
        "gpu_plus_ram_hit_percent": pct(gpu_sel + ram_sel),
        "gpu_tier": {"experts": tier_info["gpu_experts"], "weight_gib": tier_info["gpu_gib"],
                     "route_selections": gpu_sel, "call_share_percent": pct(gpu_sel)},
        "ram_tier": {"experts": tier_info["ram_experts"], "weight_gib": tier_info["ram_gib"],
                     "route_selections": ram_sel, "call_share_percent": pct(ram_sel)},
        "cold_ssd_tier": {"experts": tier_info["ssd_experts"], "weight_gib": tier_info["ssd_gib"],
                          "route_selections": ssd_sel, "call_share_percent": pct(ssd_sel)},
    }


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class RoutingCounterManager:
    def __init__(self, spec: CounterSpec | None = None,
                 counts_bin: Path | None = None, stats_json: Path | None = None):
        if spec is None:
            base = CounterSpec.legacy()
            spec = CounterSpec(
                **{**base.__dict__,
                   "counts_bin": Path(counts_bin) if counts_bin else base.counts_bin,
                   "stats_json": Path(stats_json) if stats_json else base.stats_json}
            )
        self.spec = spec
        self.counts_bin = spec.counts_bin
        self.stats_json = spec.stats_json
        self.lock = threading.Lock()
        self._save_lock = threading.Lock()
        self.historical_counts = [0] * spec.n_pairs
        self.live_counts = [0] * spec.n_pairs
        self.session_counts = [0] * spec.n_pairs
        self.historical_records = 0
        self.live_records = 0
        self.session_records = 0
        self.live_flushes = 0
        self.updated_unix_ms = int(time.time() * 1000)
        self.pipe_name: str | None = None
        self.active_pipes = 0
        self.last_save_mono = time.monotonic()
        self.last_profile_build_mono = 0.0
        self.profile_builds = 0
        self._dirty = False
        self._frozen_cache: dict = {}
        self._live_prof_cache: dict = {}
        self._ram15_cache: dict = {}
        self.seed_prior_info: dict | None = None
        self._load_existing()
        try:
            self._write_identity()
        except Exception as ex:  # noqa: BLE001
            print(f"[strata-routing-counter] identity write failed: {ex}", flush=True)
        atexit.register(self.save_if_dirty)

    # ---- identity ----
    def _identity_path(self) -> Path:
        return self.counts_bin.with_name("identity.json")

    def _check_identity(self) -> None:
        if self.spec.bind_mode != "strict":
            return
        ident = self._identity_path()
        expected = self.spec.identity
        if not expected:
            raise CounterIdentityError(
                "bind_mode=strict but routing_counter.identity is empty; refusing to guess")
        if self.counts_bin.is_file() and not ident.is_file():
            raise CounterIdentityError(
                f"{self.counts_bin} exists without {ident.name}: refusing to adopt an unbound ledger")
        if ident.is_file():
            try:
                on_disk = json.loads(ident.read_text(encoding="utf-8"))
            except Exception as ex:  # noqa: BLE001
                raise CounterIdentityError(f"{ident}: unreadable ({ex})") from ex
            a = on_disk.get("identity") or {}
            if a != expected:
                diff = sorted(k for k in set(a) | set(expected) if a.get(k) != expected.get(k))
                raise CounterIdentityError(
                    f"{ident}: model identity mismatch on {diff}; "
                    f"refusing to mix another model's ledger")

    def _write_identity(self) -> None:
        if self.spec.bind_mode != "strict" or not self.spec.identity:
            return
        ident = self._identity_path()
        if ident.is_file():
            return
        payload = {
            "identity": self.spec.identity,
            "created_unix": int(time.time()),
            "counts_bin": str(self.counts_bin),
            "binary_format": {"magic": MAGIC.decode(), "version": VERSION,
                              "n_layers": self.spec.n_layers, "n_experts": self.spec.n_experts,
                              "top_k": self.spec.top_k},
            "geometry": {"layer_blob_bytes": list(self.spec.layer_blob_bytes),
                         "total_expert_bytes": self.spec.total_expert_bytes,
                         "gpu_slots": self.spec.gpu_slots,
                         "gpu_budget_bytes": self.spec.gpu_budget_bytes,
                         "max_slot_bytes": self.spec.max_slot_bytes},
        }
        tmp = ident.with_name(ident.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(ident)

    # ---- load / seed ----
    def _load_existing(self) -> bool:
        self._check_identity()
        if not self.counts_bin.is_file():
            self._seed_prior()
            return False
        try:
            blob = self.counts_bin.read_bytes()
            if len(blob) != self.spec.expected_bin_bytes:
                raise CounterIdentityError(
                    f"{self.counts_bin}: {len(blob)} bytes, expected {self.spec.expected_bin_bytes}")
            magic, ver, nl, ne, top_k, h_rec, l_rec, flushes, upd_ms = HEADER_STRUCT.unpack_from(blob, 0)
            if magic != MAGIC or ver != VERSION or (nl, ne) != (self.spec.n_layers, self.spec.n_experts):
                raise CounterIdentityError(
                    f"{self.counts_bin}: header {magic!r} v{ver} {nl}x{ne} does not match "
                    f"{MAGIC!r} v{VERSION} {self.spec.n_layers}x{self.spec.n_experts}")
            cs = self.spec.counts_struct()
            off1 = HEADER_STRUCT.size
            off2 = off1 + cs.size
            self.historical_counts = list(cs.unpack_from(blob, off1))
            self.live_counts = list(cs.unpack_from(blob, off2))
            self.historical_records = h_rec
            self.live_records = l_rec
            self.live_flushes = flushes
            self.updated_unix_ms = upd_ms
            self._write_identity()
            return True
        except CounterIdentityError:
            raise
        except Exception as ex:  # noqa: BLE001
            raise CounterIdentityError(f"{self.counts_bin}: cannot be loaded ({ex})") from ex

    def _seed_prior(self) -> None:
        """First start: counts stay at zero, the ranking order is copied as a prior."""
        prior = self.spec.seed_prior_profile
        if prior is None:
            return
        try:
            if not prior.is_file():
                print(f"[strata-routing-counter] seed prior missing: {prior}", flush=True)
                return
            blob = prior.read_bytes()
            if blob[:4] != b"STRP":
                print(f"[strata-routing-counter] seed prior is not STRP: {prior}", flush=True)
                return
            _magic, _ver, nl, ne, _slots, _n = STRP_HEADER_STRUCT.unpack_from(blob, 0)
            if (nl, ne) != (self.spec.n_layers, self.spec.n_experts):
                print(f"[strata-routing-counter] seed prior geometry {nl}x{ne} != "
                      f"{self.spec.n_layers}x{self.spec.n_experts}; not copied", flush=True)
                return
            self.spec.profile_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.spec.profile_path.with_suffix(".bin.tmp")
            tmp.write_bytes(blob)
            tmp.replace(self.spec.profile_path)
            self.seed_prior_info = {
                "source": str(prior),
                "source_sha256": hashlib.sha256(blob).hexdigest(),
                "copied_at_unix": int(time.time()),
                "copied_to": str(self.spec.profile_path),
                "note": "rank order only; SC117 counts start at zero",
            }
            side = self.counts_bin.with_name("seed-prior.json")
            side.write_text(json.dumps(self.seed_prior_info, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
            print(f"[strata-routing-counter] seeded rank prior from {prior} "
                  f"(counts start at zero)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[strata-routing-counter] seed prior error: {ex}", flush=True)

    # ---- ingest ----
    def ingest_raw_bytes(self, buf: bytes | memoryview, is_historical: bool = False) -> int:
        n_bytes = len(buf)
        if n_bytes < RECORD_K10_BYTES:
            return 0
        n_pairs = self.spec.n_pairs
        n_experts = self.spec.n_experts
        _layer0, k0 = struct.unpack_from("<ii", buf, 0)
        if k0 == self.spec.top_k and n_bytes % RECORD_K10_BYTES == 0:
            local_add = [0] * n_pairs
            valid_recs = 0
            for rec in RECORD_K10_STRUCT.iter_unpack(buf):
                layer = rec[0]
                if 0 <= layer < self.spec.n_layers:
                    base = layer * n_experts
                    valid_recs += 1
                    for e in rec[2:]:
                        if 0 <= e < n_experts:
                            local_add[base + e] += 1
            with self.lock:
                if is_historical:
                    hc = self.historical_counts
                    for i in range(n_pairs):
                        if local_add[i]:
                            hc[i] += local_add[i]
                    self.historical_records += valid_recs
                else:
                    lc = self.live_counts
                    sc = self.session_counts
                    for i in range(n_pairs):
                        v = local_add[i]
                        if v:
                            lc[i] += v
                            sc[i] += v
                    self.live_records += valid_recs
                    self.session_records += valid_recs
                self.updated_unix_ms = int(time.time() * 1000)
                self._dirty = True
            return n_bytes

        off = 0
        local_add = [0] * n_pairs
        valid_recs = 0
        while off + 8 <= n_bytes:
            layer, k = struct.unpack_from("<ii", buf, off)
            if k <= 0 or k > 64:
                break
            rec_len = 8 + 8 * k
            if off + rec_len > n_bytes:
                break
            ids = struct.unpack_from(f"<{k}i", buf, off + 8)
            if 0 <= layer < self.spec.n_layers:
                base = layer * n_experts
                valid_recs += 1
                for e in ids:
                    if 0 <= e < n_experts:
                        local_add[base + e] += 1
            off += rec_len
        if valid_recs > 0:
            with self.lock:
                target = self.historical_counts if is_historical else self.live_counts
                for i in range(n_pairs):
                    if local_add[i]:
                        target[i] += local_add[i]
                        if not is_historical:
                            self.session_counts[i] += local_add[i]
                if is_historical:
                    self.historical_records += valid_recs
                else:
                    self.live_records += valid_recs
                    self.session_records += valid_recs
                self.updated_unix_ms = int(time.time() * 1000)
                self._dirty = True
        return off

    # ---- persistence ----
    def rebuild_profile(self) -> dict:
        with self.lock:
            cum = [self.historical_counts[i] + self.live_counts[i] for i in range(self.spec.n_pairs)]
            self.profile_builds += 1
            self.last_profile_build_mono = time.monotonic()
        return build_self_learning_profile(cum, self.spec)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> bool:
        last: Exception | None = None
        for attempt in range(3):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
                return True
            except Exception as ex:  # noqa: BLE001
                last = ex
                time.sleep(0.2 * (attempt + 1))
        print(f"[strata-routing-counter] write {path} failed after 3 tries: {last}", flush=True)
        return False

    def save(self, force_rebuild_profile: bool = False) -> dict:
        with self._save_lock:
            with self.lock:
                self.live_flushes += 1
                self.updated_unix_ms = int(time.time() * 1000)
                hdr = HEADER_STRUCT.pack(
                    MAGIC, VERSION, self.spec.n_layers, self.spec.n_experts, self.spec.top_k,
                    self.historical_records, self.live_records,
                    self.live_flushes, self.updated_unix_ms)
                cs = self.spec.counts_struct()
                b_hist = cs.pack(*self.historical_counts)
                b_live = cs.pack(*self.live_counts)
                cum = [self.historical_counts[i] + self.live_counts[i] for i in range(self.spec.n_pairs)]
                now_mono = time.monotonic()
                should_rebuild = (
                    force_rebuild_profile
                    or (now_mono - self.last_profile_build_mono >= 60.0
                        and now_mono - self.last_save_mono >= 0.0)
                )
                if should_rebuild:
                    self.profile_builds += 1
                    self.last_profile_build_mono = now_mono

            ok_bin = self._atomic_write(self.counts_bin, hdr + b_hist + b_live)
            self._write_identity()
            if ok_bin:
                with self.lock:
                    self._dirty = False
                    self.last_save_mono = time.monotonic()
            profile_ok = True
            if should_rebuild and sum(cum) > 0:
                try:
                    build_self_learning_profile(cum, self.spec)
                except Exception as ex:  # noqa: BLE001
                    profile_ok = False
                    print(f"[strata-routing-counter] profile rebuild error: {ex}", flush=True)
            stats = self.get_stats(flush_first=False)
            ok_json = self._atomic_write(
                self.stats_json,
                (json.dumps(stats, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            if not (ok_bin and ok_json and profile_ok):
                with self.lock:
                    self._dirty = True
            return stats

    def save_if_dirty(self) -> None:
        with self.lock:
            dirty = self._dirty
        if dirty:
            try:
                self.save()
            except Exception as ex:  # noqa: BLE001
                print(f"[strata-routing-counter] save_if_dirty error: {ex}", flush=True)

    # ---- stats ----
    def get_stats(self, flush_first: bool = True) -> dict:
        if flush_first:
            self.save_if_dirty()
        with self.lock:
            hist = list(self.historical_counts)
            live = list(self.live_counts)
            sess = list(self.session_counts)
            h_rec, l_rec, s_rec = self.historical_records, self.live_records, self.session_records
            flushes, prof_builds, upd_ms = self.live_flushes, self.profile_builds, self.updated_unix_ms
            pipe_name, active_pipes = self.pipe_name, self.active_pipes
        spec = self.spec
        cum = [hist[i] + live[i] for i in range(spec.n_pairs)]
        live_prof = _load_profile_tiers(spec.profile_path, spec.n_layers, spec.n_experts,
                                        spec.layer_blob_bytes, spec.gpu_budget_bytes,
                                        spec.ram_tier_bytes, self._live_prof_cache)
        frozen_info = _load_profile_tiers(spec.frozen_profile_path, spec.n_layers, spec.n_experts,
                                          spec.layer_blob_bytes, spec.gpu_budget_bytes,
                                          spec.ram_tier_bytes, self._frozen_cache)
        prof15 = _load_profile_tiers(spec.profile_path, spec.n_layers, spec.n_experts,
                                     spec.layer_blob_bytes, spec.gpu_budget_bytes,
                                     15 * (1 << 30), self._ram15_cache)
        return {
            "counter_version": VERSION,
            "mode": "self_learning_cumulative_heat_density_counter",
            "ranking_algorithm": "self_learning_cumulative_heat_density ((cumulative_calls + 1) / expert_blob_bytes)",
            "bind_mode": spec.bind_mode,
            "model_identity": spec.identity or None,
            "pipe_name": pipe_name or DEFAULT_PIPE_NAME,
            "active_pipe_readers": active_pipes,
            "raw_trace_files_on_disk": False,
            "self_learning_profile_builds": prof_builds,
            "binary_counter_file": {
                "path": str(self.counts_bin),
                "fixed_size_bytes": spec.expected_bin_bytes,
                "fixed_size_kib": round(spec.expected_bin_bytes / 1024.0, 2),
                "flushes": flushes,
                "updated_unix_ms": upd_ms,
                "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(upd_ms / 1000.0)),
            },
            "dimensions": {
                "n_layers": spec.n_layers,
                "n_experts_per_layer": spec.n_experts,
                "total_experts": spec.n_pairs,
                "top_k_per_token_layer": spec.top_k,
                "max_slot_bytes": spec.max_slot_bytes,
                "gpu_budget_bytes": spec.gpu_budget_bytes,
                "gpu_budget_gib": round(spec.gpu_budget_bytes / (1 << 30), 6),
                "ram_budget_bytes": spec.ram_tier_bytes,
                "ram_budget_gib": round(spec.ram_tier_bytes / (1 << 30), 6),
                "total_expert_bytes": spec.total_expert_bytes,
            },
            "counter_scope": {
                "sources": ["prefill", "decode"],
                "record_bytes": RECORD_K10_BYTES,
                "records_per_token_layer": 1,
                "declared_coverage": spec.coverage_note or
                    "engine --dump-routing records (prefill+decode); batch-mtp uses the same dump path",
                "note": "token_positions = records / n_layers; engine route emissions, not API usage tokens",
            },
            "count_basis": {
                "historical_seed_tokens": "records folded in from archived traces (this ledger only)",
                "live_since_counter_enabled": "counts accumulated since this ledger was created (SC117 identity)",
                "current_session_tokens": "counts since this server process started (all conversations)",
                "single_conversation": "not tracked separately; use /v1/usage or engine timings for one chat",
            },
            "seed_prior": self.seed_prior_info,
            "token_positions_counted": {
                "historical_seed_tokens": round(h_rec / spec.n_layers, 2),
                "live_tokens_since_counter_enabled": round(l_rec / spec.n_layers, 2),
                "current_session_tokens": round(s_rec / spec.n_layers, 2),
                "cumulative_tokens": round((h_rec + l_rec) / spec.n_layers, 2),
            },
            "observed_distinct_experts": {
                "historical_seed": sum(1 for c in hist if c > 0),
                "live_since_counter_enabled": sum(1 for c in live if c > 0),
                "cumulative": sum(1 for c in cum if c > 0),
            },
            "static_estimate_source": {
                "profile_path": live_prof["path"] if live_prof else str(spec.profile_path),
                "profile_sha256": live_prof["sha256"] if live_prof else None,
                "profile_mtime_unix": live_prof["mtime_unix"] if live_prof else None,
                "gpu_budget_bytes": spec.gpu_budget_bytes,
                "gpu_budget_gib": round(spec.gpu_budget_bytes / (1 << 30), 6),
                "gpu_slots": spec.gpu_slots,
                "max_slot_bytes": spec.max_slot_bytes,
                "alignment": "gpu_slots * ceil256(max_slot_bytes)",
                "note": "static coverage is the share of counted selections inside the ranking "
                        "prefix that fits the GPU budget; it is not the engine's runtime GPU hit rate",
            },
            "by_latest_learned_ranking": {
                "profile_path": live_prof["path"] if live_prof else None,
                "profile_sha256": live_prof["sha256"] if live_prof else None,
                "profile_mtime_unix": live_prof["mtime_unix"] if live_prof else None,
                "live_since_counter_enabled": _tier_breakdown(live, live_prof),
                "current_session": _tier_breakdown(sess, live_prof),
                "cumulative_all_time": _tier_breakdown(cum, live_prof),
                "cumulative_at_15gib_ram_equiv": _tier_breakdown(cum, prof15),
            },
            "by_frozen_baseline_ranking": {
                "profile_path": frozen_info["path"] if frozen_info else None,
                "profile_sha256": frozen_info["sha256"] if frozen_info else None,
                "live_since_counter_enabled": _tier_breakdown(live, frozen_info),
                "current_session": _tier_breakdown(sess, frozen_info),
                "historical_seed": _tier_breakdown(hist, frozen_info),
                "cumulative_all_time": _tier_breakdown(cum, frozen_info),
            },
        }

    # ---- named pipe ----
    def prepare_engine_pipe(self, args: list[str]) -> None:
        if os.name != "nt" or "--dump-routing" not in args:
            return
        idx = args.index("--dump-routing")
        if idx + 1 >= len(args):
            return
        target = args[idx + 1]
        if not target.lower().startswith("\\\\.\\pipe\\"):
            return
        self.pipe_name = target
        k32 = ctypes.windll.kernel32
        PIPE_ACCESS_INBOUND = 0x00000001
        PIPE_TYPE_BYTE = 0x00000000
        PIPE_WAIT = 0x00000000
        PIPE_UNLIMITED_INSTANCES = 255
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
        h_pipe = k32.CreateNamedPipeW(
            target, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES, 4 * 1024 * 1024, 4 * 1024 * 1024, 0, None)
        if h_pipe == INVALID_HANDLE_VALUE or h_pipe == 0:
            print(f"[strata-routing-counter] CreateNamedPipeW({target}) failed "
                  f"with error {k32.GetLastError()}", flush=True)
            return
        threading.Thread(target=self._pipe_reader_loop, args=(h_pipe, target),
                         daemon=True, name="strata-routing-pipe").start()

    def _pipe_reader_loop(self, h_pipe: int, pipe_name: str) -> None:
        k32 = ctypes.windll.kernel32
        ERROR_PIPE_CONNECTED = 535
        with self.lock:
            self.active_pipes += 1
        try:
            connected = k32.ConnectNamedPipe(h_pipe, None)
            if not connected and k32.GetLastError() != ERROR_PIPE_CONNECTED:
                return
            buf = ctypes.create_string_buffer(262144)
            br = ctypes.c_ulong()
            avail = ctypes.c_ulong()
            rem = bytearray()
            while True:
                ok = k32.ReadFile(h_pipe, buf, len(buf), ctypes.byref(br), None)
                if not ok or br.value == 0:
                    break
                rem.extend(buf.raw[:br.value])
                usable = (len(rem) // RECORD_K10_BYTES) * RECORD_K10_BYTES
                if usable > 0:
                    consumed = self.ingest_raw_bytes(memoryview(rem)[:usable], is_historical=False)
                    if consumed > 0:
                        del rem[:consumed]
                if k32.PeekNamedPipe(h_pipe, None, 0, None, ctypes.byref(avail), None) and avail.value == 0:
                    if time.monotonic() - self.last_save_mono >= 2.0:
                        self.save_if_dirty()
        except Exception as ex:  # noqa: BLE001
            print(f"[strata-routing-counter] pipe reader exception: {ex}", flush=True)
        finally:
            try:
                if rem:
                    usable = (len(rem) // RECORD_K10_BYTES) * RECORD_K10_BYTES
                    if usable > 0:
                        self.ingest_raw_bytes(memoryview(rem)[:usable], is_historical=False)
                self.save_if_dirty()
            finally:
                k32.DisconnectNamedPipe(h_pipe)
                k32.CloseHandle(h_pipe)
                with self.lock:
                    self.active_pipes = max(0, self.active_pipes - 1)


# --------------------------------------------------------------------------- #
# Process-wide binding
# --------------------------------------------------------------------------- #
_GLOBAL_MANAGER: RoutingCounterManager | None = None
_PENDING_SPEC: CounterSpec | None = None
_GLOBAL_LOCK = threading.Lock()


def configure(spec: CounterSpec) -> RoutingCounterManager:
    """Install the model-bound spec before the manager is first used."""
    global _GLOBAL_MANAGER, _PENDING_SPEC
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is not None:
            if _GLOBAL_MANAGER.spec.identity != spec.identity:
                raise CounterIdentityError(
                    "routing_counter is already bound to a different model identity")
            return _GLOBAL_MANAGER
        _PENDING_SPEC = spec
        return None  # type: ignore[return-value]


def configure_from_config(cfg: dict, *, config_dir: Path | None = None) -> RoutingCounterManager | None:
    section = (cfg or {}).get("routing_counter")
    if not section:
        return None
    return configure(CounterSpec.from_config(section, config_dir=config_dir))


def manager() -> RoutingCounterManager:
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = RoutingCounterManager(_PENDING_SPEC or CounterSpec.legacy())
        return _GLOBAL_MANAGER


def reset_for_tests() -> None:
    """Tests only: drop the process-wide manager so a new spec can be installed."""
    global _GLOBAL_MANAGER, _PENDING_SPEC
    with _GLOBAL_LOCK:
        _GLOBAL_MANAGER = None
        _PENDING_SPEC = None


def seed_from_historical_traces(trace_gz_paths: list[Path], reset_live: bool = False) -> dict:
    mgr = manager()
    with mgr.lock:
        mgr.historical_counts = [0] * mgr.spec.n_pairs
        mgr.historical_records = 0
        if reset_live:
            mgr.live_counts = [0] * mgr.spec.n_pairs
            mgr.live_records = 0
            mgr.session_counts = [0] * mgr.spec.n_pairs
            mgr.session_records = 0
    for p in trace_gz_paths:
        if not p.is_file():
            continue
        with gzip.open(p, "rb") as f:
            raw = f.read()
        mgr.ingest_raw_bytes(raw, is_historical=True)
    return mgr.save(force_rebuild_profile=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-profile", action="store_true")
    ap.add_argument("--self-test-pipe", action="store_true")
    ap.add_argument("--spec-json", help="routing_counter section JSON (strict binding)")
    args = ap.parse_args()

    if args.spec_json:
        configure(CounterSpec.from_config(json.loads(args.spec_json)))

    if args.rebuild_profile:
        stats = manager().save(force_rebuild_profile=True)
        print(json.dumps(stats["by_latest_learned_ranking"]["cumulative_all_time"], indent=2))

    if args.self_test_pipe:
        spec_legacy = CounterSpec.legacy()
        test_bin = spec_legacy.counts_bin.with_name("expert-routing-counts.selftest.bin")
        test_json = spec_legacy.stats_json.with_name("expert-routing-stats.selftest.json")
        try:
            base = CounterSpec.legacy()
            tm = RoutingCounterManager(CounterSpec(**{**base.__dict__,
                "counts_bin": test_bin, "stats_json": test_json,
                "profile_path": test_bin.with_name("selftest-profile.bin"),
                "frozen_profile_path": None, "seed_prior_profile": None,
                "bind_mode": "legacy"}))
            test_pipe = r"\\.\pipe\strata_routing_selftest"
            tm.prepare_engine_pipe(["--dump-routing", test_pipe])
            ucrt = ctypes.cdll.ucrtbase
            ucrt.fopen.restype = ctypes.c_void_p
            ucrt.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            ucrt.fwrite.restype = ctypes.c_size_t
            ucrt.fwrite.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
            ucrt.fflush.argtypes = [ctypes.c_void_p]
            ucrt.fclose.argtypes = [ctypes.c_void_p]
            fp = ucrt.fopen(test_pipe.encode(), b"wb")
            assert fp, "fopen failed on test pipe"
            for layer in range(tm.spec.n_layers):
                rec = struct.pack("<ii10i10f", layer, 10, *range(10), *[1.0] * 10)
                ucrt.fwrite(rec, len(rec), 1, fp)
            ucrt.fflush(fp)
            ucrt.fclose(fp)
            time.sleep(0.2)
            st = tm.get_stats()
            assert st["token_positions_counted"]["live_tokens_since_counter_enabled"] == 1.0, st
            assert st["by_latest_learned_ranking"]["live_since_counter_enabled"]["total_route_selections"] == 480, st
            assert test_bin.stat().st_size == tm.spec.expected_bin_bytes
            print("PIPE SELF-TEST PASSED: 48 layers x 10 experts = 480 selections, file size =",
                  test_bin.stat().st_size)
        finally:
            test_bin.unlink(missing_ok=True)
            test_json.unlink(missing_ok=True)
            test_bin.with_name("selftest-profile.bin").unlink(missing_ok=True)
