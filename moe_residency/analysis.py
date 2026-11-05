"""Extracted STRC/STRP formats and heat-density ranking, with explicit inputs.

The deployed counter is retained in reference/. This module only reads inputs
and builds offline projections. It has no engine, network, or process controls.
"""

from dataclasses import dataclass
from fractions import Fraction
import struct


LEDGER_HEADER = struct.Struct("<4sIIIIQQQQ")
PROFILE_HEADER = struct.Struct("<4s5I")


@dataclass(frozen=True)
class Ledger:
    layers: int
    experts: int
    top_k: int
    historical_records: int
    live_records: int
    counts: tuple[int, ...]


def read_ledger(blob: bytes) -> Ledger:
    """Read STRC v1 without constructing the write-capable deployment manager."""
    if len(blob) < LEDGER_HEADER.size:
        raise ValueError("truncated STRC header")
    magic, version, layers, experts, top_k, historical, live, _, _ = LEDGER_HEADER.unpack_from(blob)
    if magic != b"STRC" or version != 1:
        raise ValueError("expected STRC v1")
    if not (1 <= layers <= 65535 and 1 <= experts <= 65535 and 1 <= top_k <= experts):
        raise ValueError("invalid ledger geometry")
    pairs = layers * experts
    if len(blob) != LEDGER_HEADER.size + pairs * 16:
        raise ValueError("STRC size does not match its geometry")
    values = struct.unpack_from(f"<{pairs * 2}Q", blob, LEDGER_HEADER.size)
    counts = tuple(values[i] + values[pairs + i] for i in range(pairs))
    return Ledger(layers, experts, top_k, historical, live, counts)


def validate_layout(ledger: Ledger, layer_bytes: list[int]) -> None:
    if len(layer_bytes) != ledger.layers or any(type(size) is not int or size <= 0 for size in layer_bytes):
        raise ValueError("one positive integer blob size is required per layer")


def rank_experts(ledger: Ledger, layer_bytes: list[int], policy: str = "density") -> list[int]:
    """Preserve the deployed density rule, using exact ratios for uint64 counts."""
    validate_layout(ledger, layer_bytes)
    if policy not in ("density", "frequency"):
        raise ValueError("policy must be density or frequency")
    values = ledger.counts
    if policy == "frequency":
        return sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)
    return sorted(range(len(values)),
                  key=lambda i: (Fraction(values[i] + 1, layer_bytes[i // ledger.experts]), values[i], -i),
                  reverse=True)


def validate_order(order: list[int], pairs: int) -> None:
    if len(order) != pairs or set(order) != set(range(pairs)):
        raise ValueError("profile must contain every expert exactly once")


def coverage(ledger: Ledger, layer_bytes: list[int], order: list[int], budget_bytes: int) -> dict:
    """Project the ranked prefix that fits, stopping at its first budget barrier.

    Later smaller experts are not backfilled. This follows the published counter's
    prefix semantics and does not solve an optimal knapsack problem.
    """
    validate_layout(ledger, layer_bytes)
    validate_order(order, len(ledger.counts))
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget must be nonnegative integer bytes")
    used = calls = members = 0
    for index in order:
        size = layer_bytes[index // ledger.experts]
        if used + size > budget_bytes:
            break
        used += size
        calls += ledger.counts[index]
        members += 1
    total = sum(ledger.counts)
    return {"budget_bytes": budget_bytes, "resident_blob_bytes": used, "experts": members,
            "counted_selections": calls, "total_counted_selections": total,
            "selection_share": calls / total if total else None,
            "basis": "static ranked-prefix coverage on the supplied ledger"}


def encode_profile(ledger: Ledger, order: list[int]) -> bytes:
    """Encode STRP v1 full ordering and inverse table."""
    pairs = len(ledger.counts)
    validate_order(order, pairs)
    inverse = [0] * pairs
    rows = bytearray()
    for rank, index in enumerate(order):
        rows.extend(struct.pack("<HH", index // ledger.experts, index % ledger.experts))
        inverse[index] = rank
    return (PROFILE_HEADER.pack(b"STRP", 1, ledger.layers, ledger.experts, pairs, pairs)
            + rows + struct.pack(f"<{pairs}i", *inverse))


def decode_profile(blob: bytes, ledger: Ledger) -> list[int]:
    pairs = len(ledger.counts)
    if len(blob) != PROFILE_HEADER.size + pairs * 8:
        raise ValueError("STRP size does not match ledger geometry")
    if PROFILE_HEADER.unpack_from(blob) != (b"STRP", 1, ledger.layers, ledger.experts, pairs, pairs):
        raise ValueError("STRP header does not match ledger geometry")
    order = []
    for layer, expert in struct.iter_unpack("<HH", blob[24:24 + pairs * 4]):
        if layer >= ledger.layers or expert >= ledger.experts:
            raise ValueError("invalid expert in profile")
        order.append(layer * ledger.experts + expert)
    validate_order(order, pairs)
    inverse = struct.unpack_from(f"<{pairs}i", blob, 24 + pairs * 4)
    if any(inverse[index] != rank for rank, index in enumerate(order)):
        raise ValueError("invalid profile inverse table")
    return order
