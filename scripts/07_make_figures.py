from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eeg_ma.visualization import (
    collect_feature_set_summary,
    plot_all_feature_classification,
    plot_behavioral,
    plot_connectivity_networks,
    plot_feature_set_comparison,
    plot_sfs_top_features,
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate report/poster-ready EEG analysis figures")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--top-n", type=int, default=12, help="Top SFS / connectivity features to show")
    args = p.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    intra_overall = _read(root / "intra" / "intra_overall_summary.csv")
    inter_summary = _read(root / "inter" / "inter_summary.csv")
    intra_freq = _read(root / "intra" / "intra_feature_frequency.csv")
    inter_freq = _read(root / "inter" / "inter_feature_frequency.csv")
    behavior = _read(root / "behavioral_summary.csv")

    summary_path = root / "feature_set_comparison.csv"
    if summary_path.exists():
        feature_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    else:
        feature_summary = collect_feature_set_summary(root, feature_sets=("all", "bp", "coh"))
        if set(feature_summary.get("feature_set", [])) >= {"all", "bp", "coh"}:
            feature_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            print(
                "[WARN] BP-only / COH-only result summaries are incomplete. "
                "Run scripts/06_compare_feature_sets.py first for the full comparison figures."
            )

    made = []
    made += plot_all_feature_classification(intra_overall, inter_summary, out)
    made += plot_sfs_top_features(intra_freq, inter_freq, out, top_n=args.top_n)
    made += plot_connectivity_networks(inter_freq, out, top_n=args.top_n)
    made += plot_behavioral(behavior, out)
    if not feature_summary.empty:
        made += plot_feature_set_comparison(feature_summary, out)

    print(f"Generated {len(made)} figure files:")
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
