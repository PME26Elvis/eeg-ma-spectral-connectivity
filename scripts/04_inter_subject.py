from __future__ import annotations

import argparse
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.experiment_runs import add_identity_arguments, load_features, run_single, source_inputs
from eeg_ma.provenance import read_csv_literal


def main() -> int:
    p = argparse.ArgumentParser(description="Inter-subject: SFS + LDA / RBF-SVM / KFDA with outer LOPO-CV")
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-dir", default="outputs/results/baseline/inter")
    add_identity_arguments(p)
    args = p.parse_args()

    cfg = load_config(args.config)
    df = load_features(args.features, args.legacy_manifest)
    root = run_single(df, cfg, "inter", args.out_dir,
        input_paths=source_inputs({"features": args.features, "config": args.config}, args.legacy_manifest),
        resume=args.resume)
    folds = read_csv_literal(root / "inter_folds.csv")
    print(f"complete_run={root}")
    print(folds.groupby(["comparison", "classifier"])["accuracy"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
