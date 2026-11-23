"""Run and verify explicitly scoped five-classifier exploratory LOPO comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_ma.exploratory import (
    ExploratoryError, _read_json, _verify_plan, build_exploratory_report,
    validate_exploratory_report,
)
from eeg_ma.schema import load_feature_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Create a new complete immutable S/F report")
    run.add_argument("--features", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--allow-synthetic", action="store_true")
    check = commands.add_parser("check", help="Validate input and plan without training")
    check.add_argument("--features", required=True)
    check.add_argument("--config", required=True)
    check.add_argument("--plan", required=True)
    check.add_argument("--allow-synthetic", action="store_true")
    verify = commands.add_parser("verify", help="Verify an immutable complete report")
    verify.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            result = validate_exploratory_report(args.report)
        elif args.command == "check":
            features = load_feature_table(args.features)
            _verify_plan(_read_json(Path(args.plan)), _read_json(Path(args.config)),
                         features, allow_synthetic=args.allow_synthetic)
            result = {"status": "validated_input_contract",
                      "scope": "descriptive_exploratory",
                      "n_participants": features.subject_id.nunique(),
                      "n_epochs": len(features), "no_training": True}
        else:
            result = build_exploratory_report(
                args.features, args.config, args.plan, args.out_dir,
                allow_synthetic=args.allow_synthetic)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError, ExploratoryError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
