"""Record present legacy artifact bytes without upgrading their provenance."""
from __future__ import annotations

import argparse
from pathlib import Path

from eeg_ma.provenance import ProvenanceError, legacy_inventory, protect_output, _atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit read-only inventory of unverified historical artifacts")
    parser.add_argument("--root", default="results")
    parser.add_argument("--out", default="outputs/inventory/discovery-results.json")
    args = parser.parse_args()
    output = protect_output(args.out)
    source = Path(args.root).resolve()
    if source == output.resolve() or source in output.resolve().parents:
        raise ProvenanceError("Inventory output must be outside the source tree")
    if output.exists():
        raise ProvenanceError("Inventory already exists; choose a new snapshot path")
    inventory = legacy_inventory(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, inventory)
    print(f"legacy_unverified_inventory={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
