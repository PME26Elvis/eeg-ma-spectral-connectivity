from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_inter_subject, run_intra_subject
from eeg_ma.extensions import merge_baseline_and_extensions


def _summary_from_folds(scope: str, experiment: str, feature_set: str, folds: pd.DataFrame) -> pd.DataFrame:
    if scope == "intra":
        per_subject = (
            folds.groupby(["subject_id", "comparison", "classifier"], dropna=False)["accuracy"]
            .mean()
            .reset_index(name="unit_accuracy")
        )
        summary = (
            per_subject.groupby(["comparison", "classifier"], dropna=False)["unit_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_units"})
        )
    elif scope == "inter":
        summary = (
            folds.groupby(["comparison", "classifier"], dropna=False)["accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_units"})
        )
    else:
        raise ValueError(scope)
    summary.insert(0, "feature_set", feature_set)
    summary.insert(0, "experiment", experiment)
    summary.insert(0, "scope", scope)
    return summary


def _baseline_summary(baseline_results_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    intra_path = baseline_results_root / "intra" / "intra_overall_summary.csv"
    if intra_path.exists():
        x = pd.read_csv(intra_path, encoding="utf-8-sig")
        x = x.rename(columns={
            "participant_accuracy_mean": "accuracy_mean",
            "participant_accuracy_std": "accuracy_std",
            "n_subjects": "n_units",
        })
        x.insert(0, "feature_set", "baseline")
        x.insert(0, "experiment", "poster_baseline")
        x.insert(0, "scope", "intra")
        rows.append(x[["scope", "experiment", "feature_set", "comparison", "classifier", "accuracy_mean", "accuracy_std", "n_units"]])

    inter_path = baseline_results_root / "inter" / "inter_summary.csv"
    if inter_path.exists():
        x = pd.read_csv(inter_path, encoding="utf-8-sig")
        x = x.rename(columns={
            "lopo_accuracy_mean": "accuracy_mean",
            "lopo_accuracy_std": "accuracy_std",
            "n_held_out_subjects": "n_units",
        })
        x.insert(0, "feature_set", "baseline")
        x.insert(0, "experiment", "poster_baseline")
        x.insert(0, "scope", "inter")
        rows.append(x[["scope", "experiment", "feature_set", "comparison", "classifier", "accuracy_mean", "accuracy_std", "n_units"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _delta_tables(experiment: str, experiment_root: Path, baseline_results_root: Path) -> List[pd.DataFrame]:
    rows: List[pd.DataFrame] = []

    base_intra = baseline_results_root / "intra" / "intra_summary.csv"
    ext_intra = experiment_root / "intra" / "intra_summary.csv"
    if base_intra.exists() and ext_intra.exists():
        b = pd.read_csv(base_intra, encoding="utf-8-sig")
        e = pd.read_csv(ext_intra, encoding="utf-8-sig")
        keys = ["subject_id", "comparison", "classifier"]
        m = b[keys + ["accuracy_mean"]].merge(
            e[keys + ["accuracy_mean"]], on=keys, suffixes=("_baseline", "_extension"), validate="one_to_one"
        )
        m["delta_accuracy"] = m["accuracy_mean_extension"] - m["accuracy_mean_baseline"]
        m.insert(0, "experiment", experiment)
        m.insert(0, "scope", "intra")
        m = m.rename(columns={"subject_id": "unit_id"})
        rows.append(m[["scope", "experiment", "unit_id", "comparison", "classifier", "accuracy_mean_baseline", "accuracy_mean_extension", "delta_accuracy"]])

    base_inter = baseline_results_root / "inter" / "inter_folds.csv"
    ext_inter = experiment_root / "inter" / "inter_folds.csv"
    if base_inter.exists() and ext_inter.exists():
        b = pd.read_csv(base_inter, encoding="utf-8-sig")
        e = pd.read_csv(ext_inter, encoding="utf-8-sig")
        keys = ["held_out_subject", "comparison", "classifier"]
        m = b[keys + ["accuracy"]].merge(
            e[keys + ["accuracy"]], on=keys, suffixes=("_baseline", "_extension"), validate="one_to_one"
        )
        m["delta_accuracy"] = m["accuracy_extension"] - m["accuracy_baseline"]
        m.insert(0, "experiment", experiment)
        m.insert(0, "scope", "inter")
        m = m.rename(
            columns={
                "held_out_subject": "unit_id",
                "accuracy_baseline": "accuracy_mean_baseline",
                "accuracy_extension": "accuracy_mean_extension",
            }
        )
        rows.append(m[["scope", "experiment", "unit_id", "comparison", "classifier", "accuracy_mean_baseline", "accuracy_mean_extension", "delta_accuracy"]])

    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run leakage-safe E1 asymmetry and E2 PLV experiments against the frozen poster baseline"
    )
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--extension-config", default="configs/extensions_e1_e2.json")
    p.add_argument("--baseline-results-root", default="results")
    p.add_argument("--out-root", default="results/extensions")
    p.add_argument(
        "--include-combined",
        action="store_true",
        help="Also run exploratory baseline+asymmetry+PLV. E1 and E2 remain the default primary experiments.",
    )
    args = p.parse_args()

    baseline = pd.read_csv(args.baseline_features, encoding="utf-8-sig")
    extensions = pd.read_csv(args.extension_features, encoding="utf-8-sig")
    merged = merge_baseline_and_extensions(baseline, extensions)

    base_cfg = load_config(args.config)
    ext_cfg = json.loads(Path(args.extension_config).read_text(encoding="utf-8"))
    experiments = [x for x in ext_cfg.get("experiments", []) if x.get("enabled_by_default", False)]
    if args.include_combined:
        for x in ext_cfg.get("experiments", []):
            if x.get("name") == "e1e2_combined" and x not in experiments:
                experiments.append(x)

    if not experiments:
        raise ValueError("extension config 沒有任何 enabled experiment")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    baseline_results_root = Path(args.baseline_results_root)

    summary_frames: List[pd.DataFrame] = []
    base_summary = _baseline_summary(baseline_results_root)
    if not base_summary.empty:
        summary_frames.append(base_summary)
    delta_frames: List[pd.DataFrame] = []

    print("=== E1/E2 EXTENSION EXPERIMENTS ===")
    print(f"rows={len(merged)} subjects={merged['subject_id'].nunique()}")

    for experiment in experiments:
        name = str(experiment["name"])
        feature_set = str(experiment["feature_set"])
        cfg: Dict = copy.deepcopy(base_cfg)
        cfg["feature_set"] = feature_set

        root = out_root / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "baseline_config": cfg,
                    "extension_config": ext_cfg,
                    "experiment": experiment,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"\n--- {name}: feature_set={feature_set} ---")
        intra_folds, _ = run_intra_subject(merged, cfg, root / "intra")
        inter_folds, _ = run_inter_subject(merged, cfg, root / "inter")
        summary_frames.append(_summary_from_folds("intra", name, feature_set, intra_folds))
        summary_frames.append(_summary_from_folds("inter", name, feature_set, inter_folds))
        delta_frames.extend(_delta_tables(name, root, baseline_results_root))

    summary = pd.concat(summary_frames, ignore_index=True)
    scope_order = pd.Categorical(summary["scope"], categories=["inter", "intra"], ordered=True)
    summary = summary.assign(_scope_order=scope_order).sort_values(
        ["_scope_order", "comparison", "classifier", "experiment"]
    ).drop(columns="_scope_order")
    summary.to_csv(out_root / "extension_summary.csv", index=False, encoding="utf-8-sig")

    if delta_frames:
        deltas = pd.concat(delta_frames, ignore_index=True)
        deltas.to_csv(out_root / "extension_delta_by_unit.csv", index=False, encoding="utf-8-sig")
        delta_summary = (
            deltas.groupby(["scope", "experiment", "comparison", "classifier"], dropna=False)["delta_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "n_units"})
        )
        delta_summary.to_csv(out_root / "extension_delta_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== EXTENSION SUMMARY ===")
    print(summary.to_string(index=False))
    if delta_frames:
        print("\n=== DELTA VS POSTER BASELINE ===")
        print(delta_summary.to_string(index=False))
    print(f"\nsummary={out_root / 'extension_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
