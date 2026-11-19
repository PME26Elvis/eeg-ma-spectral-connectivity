"""Inspect/freeze the prospective B plan and validate bounded phase artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_ma.artifacts import new_output_directory
from eeg_ma.control_plan import (authorize_phase_inputs, check_control_plan,
    freeze_control_plan, read_plan, validate_control_plan_lock)
from eeg_ma.provenance import code_identity, environment_identity, fingerprint, sha256_file


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="List decisions/evidence still needed; a draft exits 2")
    check.add_argument("--plan", type=Path, default=Path("configs/phase_control_plan.json"))
    check.add_argument("--out-dir", type=Path)
    freeze = commands.add_parser("freeze-plan", help="Freeze a complete reviewed B plan before new acquisition")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    verify = commands.add_parser("verify-plan")
    verify.add_argument("--lock-dir", type=Path, required=True)
    authorize = commands.add_parser("authorize-inputs", help="Validate plan/session evidence and bind prospective phase inputs")
    authorize.add_argument("--lock-dir", type=Path, required=True)
    authorize.add_argument("--epochs", type=Path, required=True)
    authorize.add_argument("--splits", type=Path, required=True)
    authorize.add_argument("--session-history", type=Path, required=True)
    authorize.add_argument("--arm", choices=("recorded_reference", "seven_frontal_average", "annotation_clean"), default="recorded_reference")
    authorize.add_argument("--annotations", type=Path)
    authorize.add_argument("--out-dir", type=Path, required=True)
    extract = commands.add_parser("extract", help="Build exactly three matched representations from supplied alpha phases")
    extract.add_argument("--mode", choices=("synthetic-fixture", "prospective"), required=True)
    extract.add_argument("--phases", type=Path, required=True)
    extract.add_argument("--epochs", type=Path, required=True)
    extract.add_argument("--splits", type=Path, required=True)
    extract.add_argument("--lock-dir", type=Path)
    extract.add_argument("--session-history", type=Path)
    extract.add_argument("--arm", choices=("recorded_reference", "seven_frontal_average", "annotation_clean"), default="recorded_reference")
    extract.add_argument("--annotations", type=Path)
    extract.add_argument("--out-dir", type=Path, required=True)
    features = commands.add_parser("verify-features", help="Verify matched feature numerics and shared provenance; no acquisition claim")
    features.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        output = check_control_plan(read_plan(args.plan), base_dir=args.plan.parent)
        output.update(plan_sha256=sha256_file(args.plan), validator_sha256=sha256_file(Path(__file__).resolve().parents[1] / "src/eeg_ma/control_plan.py"))
        if args.out_dir:
            with new_output_directory(args.out_dir) as stage:
                (stage / "readiness.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if output["ready_to_lock"] else 2
    if args.command == "freeze-plan":
        output = freeze_control_plan(args.plan, args.out_dir)
        print(json.dumps({k: output[k] for k in ("control_plan_lock_id", "study_id", "created_at", "scope")}, indent=2))
    elif args.command == "verify-plan":
        output = validate_control_plan_lock(args.lock_dir)
        print(json.dumps({"valid": True, "control_plan_lock_id": output["manifest"]["control_plan_lock_id"]}, indent=2))
    elif args.command == "authorize-inputs":
        output = authorize_phase_inputs(args.lock_dir, read_plan(args.epochs), read_plan(args.splits), read_plan(args.session_history),
                                        arm=args.arm, annotation_manifest=read_plan(args.annotations) if args.annotations else None)
        with new_output_directory(args.out_dir) as stage:
            (stage / "authorization.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"authorized": True, "scope": "prospective_B_phase_inputs_only", "out_dir": str(args.out_dir)}, indent=2))
    elif args.command == "extract":
        from eeg_ma.phase_controls import canonical_sha256, extract_matched_alpha_features, export_matched_phase_bundle
        epochs, splits = read_plan(args.epochs), read_plan(args.splits)
        if args.mode == "prospective":
            if args.lock_dir is None or args.session_history is None:
                parser.error("prospective extraction requires --lock-dir and --session-history")
            from eeg_ma.control_extraction import build_control_extraction
            archive = build_control_extraction(args.phases, epochs, splits, read_plan(args.session_history), args.out_dir,
                lock_dir=args.lock_dir, arm=args.arm, annotation_manifest=read_plan(args.annotations) if args.annotations else None)
            print(json.dumps({key: archive[key] for key in ("archive_id", "bundle_sha256", "evaluable", "n_source_epochs",
                "n_accepted_epochs", "n_recruited", "contains_model_performance_results", "scope")}, indent=2))
            return 0
        else:
            if args.lock_dir is not None or args.session_history is not None or args.annotations is not None or args.arm != "recorded_reference":
                parser.error("synthetic fixtures cannot attach a live control-plan lock or participant history")
            from eeg_ma.control_plan import CHANNELS
            spec = read_plan(Path(__file__).resolve().parents[1] / "configs/phase_control_plan.json")
            policy = {"schema_version": "eeg-matched-phase-policy-1", "analysis_id": spec["analysis_id"],
                      "evidence_kind": "synthetic_fixture", "protocol_lock_id": None,
                      "code_sha256": code_identity()["working_source_sha256"], "environment_sha256": fingerprint(environment_identity()),
                      "preprocessing_sha256": fingerprint(spec["preprocessing"]), "selection_sha256": fingerprint(spec["selection"]),
                      "epoch_manifest_sha256": canonical_sha256(epochs), "split_manifest_sha256": canonical_sha256(splits),
                      "channels": CHANNELS, "band_hz": [8.0, 13.0], "ciplv_min_denominator": 1e-6,
                      "undefined_ciplv_policy": "exclude_epoch_from_all_representations", "min_epochs_per_class": 1, "max_epochs_per_class": 30}
        phases = np.load(args.phases, allow_pickle=False)
        bundle = extract_matched_alpha_features(phases, epochs, splits, input_policy=policy)
        export_matched_phase_bundle(bundle, args.out_dir)
        print(json.dumps({"bundle_sha256": bundle.manifest["bundle_sha256"], "evidence_kind": policy["evidence_kind"],
                          "n_accepted_epochs": len(bundle.epoch_ids), "evaluable": bundle.evaluable,
                          "representations": ["plv", "iplv", "ciplv"], "features_each": 21}, indent=2))
    else:
        if (args.bundle / "archive_manifest.json").exists():
            from eeg_ma.control_extraction import verify_control_extraction
            archive = verify_control_extraction(args.bundle)
            print(json.dumps({"valid": True, **{key: archive[key] for key in ("archive_id", "bundle_sha256", "evaluable",
                "n_recruited", "scope", "acquisition_evidence_scope")}}, indent=2))
        else:
            from eeg_ma.phase_controls import verify_matched_phase_bundle
            bundle = verify_matched_phase_bundle(args.bundle)
            print(json.dumps({"valid": True, "bundle_sha256": bundle.manifest["bundle_sha256"], "evaluable": bundle.evaluable,
                              "scope": "internal_numerical_and_provenance_consistency_not_acquisition_authentication"}, indent=2))
    return 0


def main():
    try:
        return _main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
