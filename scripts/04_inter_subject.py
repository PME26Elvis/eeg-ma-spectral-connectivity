from __future__ import annotations

import argparse
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_inter_subject


def main() -> int:
    p = argparse.ArgumentParser(description="Inter-subject: SFS + LDA / RBF-SVM / KFDA with outer LOPO-CV")
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-dir", default="results/inter")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.features, encoding="utf-8-sig")
    folds, _ = run_inter_subject(df, cfg, args.out_dir)
    print(folds.groupby(["comparison", "classifier"])["accuracy"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
