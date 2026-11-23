from __future__ import annotations

import argparse
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.experiment_runs import add_identity_arguments, load_features, run_single, source_inputs, add_classifier_arguments, apply_classifier_arguments, add_cost_arguments, check_cost_preflight
from eeg_ma.provenance import read_csv_literal


def main() -> int:
    p = argparse.ArgumentParser(description="Intra-subject: SFS + LDA / RBF-SVM / KFDA / handcrafted RandomForest / LightGBM with outer 5-fold CV")
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-dir", default="outputs/results/baseline/intra")
    add_identity_arguments(p)
    add_classifier_arguments(p)
    add_cost_arguments(p)
    args = p.parse_args()
    cfg = apply_classifier_arguments(load_config(args.config), args.classifiers)

    df = load_features(args.features, args.legacy_manifest)
    execute, code = check_cost_preflight(args, df, cfg, "intra")
    if not execute:
        return code
    root = run_single(df, cfg, "intra", args.out_dir,
        input_paths=source_inputs({"features": args.features, "config": args.config}, args.legacy_manifest),
        resume=args.resume)
    folds = read_csv_literal(root / "intra_folds.csv")
    print(f"complete_run={root}")
    print(folds.groupby(["subject_id", "comparison", "classifier"])["accuracy"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
