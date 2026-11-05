"""Explicit offline analysis. Outputs are created exclusively, never overwritten."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

from .analysis import coverage, decode_profile, encode_profile, rank_experts, read_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True, help="JSON metadata and per-layer blob sizes")
    parser.add_argument("--budgets-gib", nargs="+", required=True)
    parser.add_argument("--baseline-profile", type=Path)
    parser.add_argument("--export-profile", type=Path)
    args = parser.parse_args()
    try:
        raw = args.ledger.read_bytes()
        ledger = read_ledger(raw)
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        sizes = layout["layer_blob_bytes"]
        if not layout.get("model_identity") or not layout.get("capture_coverage"):
            raise ValueError("layout requires model_identity and capture_coverage")
        expected_hash = layout.get("ledger_sha256")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_hash is None or digest != expected_hash:
            raise ValueError("ledger SHA-256 does not match the layout binding")
        decimals = [Decimal(value) for value in args.budgets_gib]
        if any(not value.is_finite() or value < 0 for value in decimals):
            raise ValueError("budgets must be finite and nonnegative")
        budgets = [int(value * (1 << 30)) for value in decimals]
        orders = {policy: rank_experts(ledger, sizes, policy) for policy in ("frequency", "density")}
        if args.baseline_profile:
            orders["baseline"] = decode_profile(args.baseline_profile.read_bytes(), ledger)
        report = {"schema_version": 1, "model_identity": layout["model_identity"],
                  "capture_coverage": layout["capture_coverage"], "ledger_sha256": digest,
                  "geometry": {"layers": ledger.layers, "experts_per_layer": ledger.experts,
                               "top_k": ledger.top_k},
                  "count_consistency": {"observed_selections": sum(ledger.counts),
                                        "records_times_top_k": (ledger.historical_records + ledger.live_records) * ledger.top_k},
                  "projections": {policy: [coverage(ledger, sizes, order, budget) for budget in budgets]
                                  for policy, order in orders.items()},
                  "interpretation": "Offline coverage on these counted selections. Validate future workload latency separately."}
        if args.export_profile:
            # Encoding is validated before opening the destination.
            blob = encode_profile(ledger, orders["density"])
            with args.export_profile.open("xb") as handle:
                handle.write(blob)
            report["exported_profile_sha256"] = hashlib.sha256(blob).hexdigest()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
