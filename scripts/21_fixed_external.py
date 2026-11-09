"""Fit fixed development models, prepare label-free inputs, predict, then score."""
from __future__ import annotations

import argparse
import json

from eeg_ma.fixed_model import fit_development
from eeg_ma.external import (prepare_external_inputs, predict_external, score_external,
                             validate_predictions, validate_external_score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-development", help="Fit/export only frozen development sources in the verified reference environment")
    fit.add_argument("--protocol", required=True)
    fit.add_argument("--out-dir", required=True)
    prepare = commands.add_parser("prepare-inputs", help="Custodian: separate QA-validated features and labels")
    prepare.add_argument("--features", required=True)
    for name in ("predict-external", "score-external", "verify-predictions", "verify-score"):
        commands.add_parser(name)
    for name in ("prepare-inputs", "predict-external", "score-external", "verify-predictions", "verify-score"):
        command = commands.choices[name]
        command.add_argument("--lock-dir", required=True)
        if name not in {"verify-predictions", "verify-score"}:
            command.add_argument("--cohort", required=True)
            command.add_argument("--out-dir", required=True)
    commands.choices["predict-external"].add_argument("--inputs", required=True, help="Only the label-free inputs/ directory")
    score = commands.choices["score-external"]
    score.add_argument("--predictions", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--enrollment", required=True)
    commands.choices["verify-predictions"].add_argument("--predictions", required=True)
    commands.choices["verify-score"].add_argument("--scores", required=True)
    commands.choices["verify-score"].add_argument("--predictions", required=True)
    args = parser.parse_args()
    try:
        if args.command == "fit-development":
            result = fit_development(args.protocol, args.out_dir)
        elif args.command == "prepare-inputs":
            result = prepare_external_inputs(args.features, args.out_dir, lock_dir=args.lock_dir, cohort_manifest_path=args.cohort)
        elif args.command == "predict-external":
            result = predict_external(args.inputs, args.out_dir, lock_dir=args.lock_dir, cohort_manifest_path=args.cohort)
        elif args.command == "score-external":
            result = score_external(args.predictions, args.labels, args.out_dir, lock_dir=args.lock_dir,
                cohort_manifest_path=args.cohort, enrollment_manifest_path=args.enrollment)
        elif args.command == "verify-predictions":
            result = validate_predictions(args.predictions, lock_dir=args.lock_dir)["manifest"]
        else:
            result = validate_external_score(args.scores, prediction_dir=args.predictions, lock_dir=args.lock_dir)
        keys = ("status", "analysis_id", "bundle_id", "input_id", "labels_id", "prediction_id", "score_id", "n_epochs", "n_predictions", "summary", "paired_deltas")
        print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
