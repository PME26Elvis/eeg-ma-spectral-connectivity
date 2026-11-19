"""Build and verify complete participant reports from immutable result archives."""
from __future__ import annotations

import argparse
import json

from eeg_ma.external_reporting import build_external_report, validate_external_report
from eeg_ma.historical_reporting import (rebuild_historical_results, write_historical_report,
                                         verify_historical_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    external = commands.add_parser("external", help="Report validated scored fixed development models; no model fitting")
    external.add_argument("--scores", required=True)
    external.add_argument("--predictions", required=True)
    external.add_argument("--lock-dir", required=True)
    external.add_argument("--out-dir", required=True)
    verify = commands.add_parser("verify-external", help="Recompute an external report from its self-contained evidence")
    verify.add_argument("--report", required=True)
    historical = commands.add_parser("historical", help="Rebuild the frozen descriptive LOPO matrix from saved predictions; no fitting")
    historical.add_argument("--results-root", default="results")
    historical.add_argument("--baseline-features", default="data/features/features_all.csv")
    historical.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    historical.add_argument("--iplv-features", default="data/features/final_validation/features_iplv.csv")
    historical.add_argument("--allow-legacy-results", action="store_true")
    historical.add_argument("--legacy-manifest")
    historical.add_argument("--experiments", nargs="+")
    historical.add_argument("--partial-report", action="store_true")
    historical.add_argument("--out-dir", required=True)
    verify_old = commands.add_parser("verify-historical", help="Recompute archived tables and compare their original source bytes")
    verify_old.add_argument("--report", required=True)
    verify_old.add_argument("--detached", action="store_true", help="Only internal archive consistency when original source files are unavailable")
    args = parser.parse_args()
    try:
        if args.command == "external":
            result = build_external_report(args.scores, args.out_dir, prediction_dir=args.predictions, lock_dir=args.lock_dir)
        elif args.command == "verify-external":
            result = validate_external_report(args.report)
        elif args.command == "historical":
            report = rebuild_historical_results(args.results_root, baseline_features=args.baseline_features,
                extension_features=args.extension_features, iplv_features=args.iplv_features,
                legacy_manifest=args.legacy_manifest, allow_legacy=args.allow_legacy_results,
                experiments=args.experiments, partial=args.partial_report)
            write_historical_report(report, args.out_dir)
            result = verify_historical_report(args.out_dir)
        else:
            result = verify_historical_report(args.report, verify_sources=not args.detached)
        keys = ("status", "report_id", "report_title", "analysis_id", "score_id", "prediction_id", "cohort_id",
                "protocol_lock_id", "protocol_sha256", "model_ids", "inference_scope", "scope", "n_subjects", "historical_provenance", "source_verification")
        print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
