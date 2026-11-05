"""Create a bounded synthetic STRC ledger in a new directory."""

import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_residency.analysis import LEDGER_HEADER


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/make_fixture.py NEW_DIRECTORY")
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=False)
    counts = [10 * ((i * 7) % 11 + 1) for i in range(32)]
    blob = (LEDGER_HEADER.pack(b"STRC", 1, 2, 16, 10, 0, sum(counts) // 10, 1, 0)
            + struct.pack("<64Q", *([0] * 32 + counts)))
    (destination / "counts.bin").write_bytes(blob)
    layout = {"model_identity": "synthetic-demo-v1",
              "capture_coverage": "Synthetic aggregate counts. No model requests.",
              "ledger_sha256": hashlib.sha256(blob).hexdigest(),
              "layer_blob_bytes": [2 * (1 << 20), 1 << 20]}
    (destination / "layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
