from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_inter_subject, run_intra_subject
from eeg_ma.visualization import collect_feature_set_summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run BP-only / COH-only SFS + LDA / RBF-SVM / KFDA comparisons"
    )
    p.add_argument("--features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--results-root", default="results")
    p.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["bp", "coh"],
        default=["bp", "coh"],
        help="Feature sets to rerun. Canonical BP+COH results are read from results/intra and results/inter.",
    )
    args = p.parse_args()

    features = pd.read_csv(args.features, encoding="utf-8-sig")
    base_cfg = load_config(args.config)
    results_root = Path(args.results_root)

    for feature_set in args.feature_sets:
        cfg = copy.deepcopy(base_cfg)
        cfg["feature_set"] = feature_set
        root = results_root / "feature_sets" / feature_set
        root.mkdir(parents=True, exist_ok=True)
        (root / "config_snapshot.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"\n=== {feature_set.upper()} ONLY: INTRA ===")
        intra_folds, _ = run_intra_subject(features, cfg, root / "intra")
        print(
            intra_folds.groupby(["comparison", "classifier"])["accuracy"]
            .mean()
            .to_string()
        )

        print(f"\n=== {feature_set.upper()} ONLY: INTER ===")
        inter_folds, _ = run_inter_subject(features, cfg, root / "inter")
        print(
            inter_folds.groupby(["comparison", "classifier"])["accuracy"]
            .mean()
            .to_string()
        )

    summary = collect_feature_set_summary(results_root, feature_sets=("all", "bp", "coh"))
    summary_path = results_root / "feature_set_comparison.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== COMBINED FEATURE-SET SUMMARY ===")
    if summary.empty:
        print("No summary rows found.")
    else:
        print(
            summary.sort_values(["scope", "comparison", "classifier", "feature_set"])
            .to_string(index=False)
        )
    print(f"\nsummary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
