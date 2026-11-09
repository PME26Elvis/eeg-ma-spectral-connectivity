from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.experiment_runs import add_identity_arguments, load_features, prepare_matrix, source_inputs
from eeg_ma.provenance import read_csv_literal


def _summary(root, feature_set):
    rows = []
    for scope in ("intra", "inter"):
        path = root / scope / ("intra_overall_summary.csv" if scope == "intra" else "inter_summary.csv")
        table = read_csv_literal(path).rename(columns={
            "participant_accuracy_mean": "accuracy_mean", "participant_accuracy_std": "accuracy_std", "n_subjects": "n_units",
            "lopo_accuracy_mean": "accuracy_mean", "lopo_accuracy_std": "accuracy_std", "n_held_out_subjects": "n_units"})
        table["scope"] = scope
        table["feature_set"] = feature_set
        rows.append(table[["scope", "feature_set", "comparison", "classifier", "accuracy_mean", "accuracy_std", "n_units"]])
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run BP/COH comparisons with immutable runs and complete-inventory reports")
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--results-root", default="results", help="Read-only baseline reference root; never an output destination")
    p.add_argument("--out-root", default="outputs/results/feature_sets")
    p.add_argument("--feature-sets", nargs="+", choices=["bp", "coh"], default=["bp", "coh"])
    add_identity_arguments(p)
    args = p.parse_args()
    features = load_features(args.features, args.legacy_manifest)
    cfg = load_config(args.config)
    experiments = [{"name": name, "feature_set": name} for name in ("bp", "coh")]
    matrix, evaluator = prepare_matrix(features, cfg, experiments=experiments, selected=args.feature_sets,
        scopes=("intra", "inter"), out_root=args.out_root,
        input_paths=source_inputs({"features": args.features, "config": args.config}, args.legacy_manifest),
        reference_paths={"poster_baseline": args.results_root}, matrix_config={"experiments": experiments}, resume=args.resume)
    baseline = _summary(Path(args.results_root), "all")
    def report(inventory):
        table = pd.concat([baseline] + [_summary(root, name) for name, root in inventory.items()], ignore_index=True)
        return {"feature_set_comparison.csv": table.sort_values(["scope", "comparison", "classifier", "feature_set"])}
    _, report_root = matrix.run(evaluator, report)
    print(f"report={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
