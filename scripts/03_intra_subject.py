from __future__ import annotations

import argparse
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_intra_subject


def main() -> int:
    p = argparse.ArgumentParser(description="Intra-subject: SFS + LDA / RBF-SVM / KFDA with outer 5-fold CV")
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-dir", default="results/intra")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.features, encoding="utf-8-sig")
    folds, _ = run_intra_subject(df, cfg, args.out_dir)
    print(folds.groupby(["subject_id", "comparison", "classifier"])["accuracy"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
