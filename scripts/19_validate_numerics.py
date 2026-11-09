"""Bounded public synthetic regression and controlled dependency comparison."""
from __future__ import annotations

import argparse
import json

from eeg_ma.numerical_validation import DEFAULT_REFERENCE, run_numerical_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="New immutable evidence directory")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="Reviewed numerical reference JSON")
    parser.add_argument("--require-reference-environment", action="store_true")
    parser.add_argument("--environment-spec", help="Explicit reference environment JSON, including a candidate lock profile")
    parser.add_argument("--compare-to", help="Validated baseline numerical_report.json; reuses its hash-verified raw bytes")
    args = parser.parse_args()
    try:
        result = run_numerical_validation(args.out_dir, reference_path=args.reference,
            require_reference_environment=args.require_reference_environment,
            environment_spec=args.environment_spec, compare_to=args.compare_to)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({key: result[key] for key in ("status", "purpose", "report_sha256", "comparisons")}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
