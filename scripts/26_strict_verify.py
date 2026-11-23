"""Reference-only full inner-training replay plus independent tiny math oracles.

Strict verification NEVER changes an existing run and never treats a resource
limit or an unverified environment as a successful proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_ma.artifacts import new_output_directory
from eeg_ma.environment import EnvironmentError
from eeg_ma.provenance import ProvenanceError, _write_json
from eeg_ma.strict_oracles import OracleMismatch, OracleNotApplicable, run_tiny_oracle_canary
from eeg_ma.strict_replay import (
    ReplayBudget, ReplayResourceLimit, StrictReplayError, verify_strict_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Strictly refit all archived inner candidates and outer models")
    verify.add_argument("--run", required=True)
    verify.add_argument("--out-dir")
    verify.add_argument("--preflight-only", action="store_true")
    verify.add_argument("--max-outer-folds", type=int, default=8)
    verify.add_argument("--max-candidates", type=int, default=8)
    verify.add_argument("--max-model-fits", type=int, default=128)
    verify.add_argument("--max-projected-nodes", type=int, default=250_000)
    verify.add_argument("--max-seconds", type=float, default=600.0)
    oracle = commands.add_parser("oracles", help="Independently exhaust tiny bounded CART/Newton fixtures")
    oracle.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            budget = ReplayBudget(
                max_outer_folds=args.max_outer_folds,
                max_candidates=args.max_candidates,
                max_model_fits=args.max_model_fits,
                max_projected_nodes=args.max_projected_nodes,
                max_seconds=args.max_seconds,
            )
            result = verify_strict_run(
                args.run, budget=budget,
                report_dir=None if args.preflight_only else args.out_dir,
                preflight_only=args.preflight_only,
            )
        else:
            result = run_tiny_oracle_canary()
            with new_output_directory(args.out_dir) as output:
                _write_json(Path(output) / "oracle_report.json", result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except ReplayResourceLimit as exc:
        print(json.dumps({"status": "not_run_resource_limit",
                          "error": str(exc),
                          "complete_verification": False}, ensure_ascii=False))
        return 3
    except (ProvenanceError, StrictReplayError, EnvironmentError, OracleMismatch,
            OracleNotApplicable, AssertionError, ValueError, TypeError, OSError, KeyError) as exc:
        print(json.dumps({"status": "failed_integrity_or_environment",
                          "error": f"{type(exc).__name__}: {exc}",
                          "complete_verification": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
