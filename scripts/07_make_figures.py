from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eeg_ma.artifacts import new_output_directory
from eeg_ma.provenance import input_inventory, read_csv_literal, resolve_report_root

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
    return read_csv_literal(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate report/poster-ready EEG analysis figures")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="outputs/figures/baseline")
    p.add_argument("--feature-results-root", help="Feature-set family with a validated CURRENT.json report")
    p.add_argument("--allow-legacy-results", action="store_true", help="Explicitly read historical results with unverified execution provenance")
    p.add_argument("--top-n", type=int, default=12, help="Top SFS / connectivity features to show")
    args = p.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)
    intra = resolve_report_root(root / "intra", allow_legacy=args.allow_legacy_results)
    inter = resolve_report_root(root / "inter", allow_legacy=args.allow_legacy_results)
    sources = {
        "intra_overall": intra / "intra_overall_summary.csv",
        "inter_summary": inter / "inter_summary.csv",
        "intra_frequency": intra / "intra_feature_frequency.csv",
        "inter_frequency": inter / "inter_feature_frequency.csv",
        "behavioral": root / "behavioral_summary.csv",
    }
    intra_overall, inter_summary, intra_freq, inter_freq, behavior = [_read(p) for p in sources.values()]

    if args.feature_results_root:
        report = resolve_report_root(args.feature_results_root, allow_legacy=args.allow_legacy_results)
        sources["feature_comparison"] = report / "feature_set_comparison.csv"
        feature_summary = _read(sources["feature_comparison"])
    elif (root / "feature_set_comparison.csv").exists() and args.allow_legacy_results:
        sources["feature_comparison"] = root / "feature_set_comparison.csv"
        feature_summary = _read(sources["feature_comparison"])
    elif args.allow_legacy_results:
        feature_summary = collect_feature_set_summary(root, feature_sets=("all", "bp", "coh"))
        for feature_set in ("bp", "coh"):
            for scope in ("intra", "inter"):
                filename = "intra_overall_summary.csv" if scope == "intra" else "inter_summary.csv"
                path = root / "feature_sets" / feature_set / scope / filename
                if path.exists():
                    sources[f"{feature_set}_{scope}_summary"] = path
    else:
        feature_summary = pd.DataFrame()

    available_sets = set(feature_summary["feature_set"].astype(str)) if not feature_summary.empty else set()
    has_full_feature_sets = {"all", "bp", "coh"}.issubset(available_sets)
    if not has_full_feature_sets:
        print(
            "[WARN] BP-only / COH-only result summaries are incomplete. "
            "Run scripts/06_compare_feature_sets.py first for the full comparison figures."
        )

    with new_output_directory(out) as stage:
        made = []
        made += plot_all_feature_classification(intra_overall, inter_summary, stage)
        made += plot_sfs_top_features(intra_freq, inter_freq, stage, top_n=args.top_n)
        made += plot_connectivity_networks(inter_freq, stage, top_n=args.top_n)
        made += plot_behavioral(behavior, stage)
        if has_full_feature_sets:
            made += plot_feature_set_comparison(feature_summary, stage)
        (stage / "figure_sources.json").write_text(json.dumps({
            "inputs": input_inventory(sources),
            "allow_legacy_results": args.allow_legacy_results,
        }, indent=2) + "\n", encoding="utf-8")
        made = [out / p.relative_to(stage) for p in made]

    print(f"Generated {len(made)} figure files:")
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
