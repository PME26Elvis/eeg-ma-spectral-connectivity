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
from eeg_ma.experiment_runs import add_identity_arguments, load_features, prepare_matrix, source_inputs
from eeg_ma.provenance import read_csv_literal


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
        x = read_csv_literal(intra_path, encoding="utf-8-sig")
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
        x = read_csv_literal(inter_path, encoding="utf-8-sig")
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
        b = read_csv_literal(base_intra, encoding="utf-8-sig")
        e = read_csv_literal(ext_intra, encoding="utf-8-sig")
        keys = ["subject_id", "comparison", "classifier"]
        m = b[keys + ["accuracy_mean"]].merge(
            e[keys + ["accuracy_mean"]], on=keys, suffixes=("_baseline", "_extension"), validate="one_to_one"
        )
        if len(m) != len(b) or len(m) != len(e):
            raise ValueError("Baseline/extension unit coverage differs")
        m["delta_accuracy"] = m["accuracy_mean_extension"] - m["accuracy_mean_baseline"]
        m.insert(0, "experiment", experiment)
        m.insert(0, "scope", "intra")
        m = m.rename(columns={"subject_id": "unit_id"})
        rows.append(m[["scope", "experiment", "unit_id", "comparison", "classifier", "accuracy_mean_baseline", "accuracy_mean_extension", "delta_accuracy"]])

    base_inter = baseline_results_root / "inter" / "inter_folds.csv"
    ext_inter = experiment_root / "inter" / "inter_folds.csv"
    if base_inter.exists() and ext_inter.exists():
        b = read_csv_literal(base_inter, encoding="utf-8-sig")
        e = read_csv_literal(ext_inter, encoding="utf-8-sig")
        keys = ["held_out_subject", "comparison", "classifier"]
        m = b[keys + ["accuracy"]].merge(
            e[keys + ["accuracy"]], on=keys, suffixes=("_baseline", "_extension"), validate="one_to_one"
        )
        if len(m) != len(b) or len(m) != len(e):
            raise ValueError("Baseline/extension unit coverage differs")
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
    p = argparse.ArgumentParser(description="Run E1/E2 using immutable runs against read-only historical references")
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--extension-config", default="configs/extensions_e1_e2.json")
    p.add_argument("--baseline-results-root", default="results")
    p.add_argument("--out-root", default="outputs/results/extensions")
    p.add_argument("--include-combined", action="store_true")
    add_identity_arguments(p)
    args = p.parse_args()
    baseline = load_features(args.baseline_features, args.legacy_manifest)
    extensions = load_features(args.extension_features, args.legacy_manifest)
    merged = merge_baseline_and_extensions(baseline, extensions)
    cfg = load_config(args.config)
    spec = json.loads(Path(args.extension_config).read_text(encoding="utf-8"))
    experiments = spec.get("experiments", [])
    required = [x["name"] for x in experiments if x.get("enabled_by_default", False)]
    selected = required + [x["name"] for x in experiments if args.include_combined and x["name"] == "e1e2_combined" and x["name"] not in required]
    matrix, evaluator = prepare_matrix(merged, cfg, experiments=experiments, selected=selected,
        scopes=("intra", "inter"), out_root=args.out_root,
        input_paths=source_inputs({"baseline_features": args.baseline_features, "extension_features": args.extension_features,
            "config": args.config, "extension_config": args.extension_config}, args.legacy_manifest),
        reference_paths={"poster_baseline": args.baseline_results_root}, matrix_config=spec, resume=args.resume)
    baseline_root = Path(args.baseline_results_root)
    base_summary = _baseline_summary(baseline_root)
    by_name = {x["name"]: x for x in experiments}
    def report(inventory):
        summaries = [base_summary]
        deltas = []
        for name, root in inventory.items():
            for scope in ("intra", "inter"):
                folds = read_csv_literal(root / scope / f"{scope}_folds.csv")
                summaries.append(_summary_from_folds(scope, name, by_name[name]["feature_set"], folds))
            deltas.extend(_delta_tables(name, root, baseline_root))
        table = pd.concat(summaries, ignore_index=True).sort_values(["scope", "comparison", "classifier", "experiment"])
        unit = pd.concat(deltas, ignore_index=True)
        summary = unit.groupby(["scope", "experiment", "comparison", "classifier"], dropna=False).delta_accuracy.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "n_units"})
        return {"extension_summary.csv": table, "extension_delta_by_unit.csv": unit, "extension_delta_summary.csv": summary}
    _, report_root = matrix.run(evaluator, report, required_experiments=required)
    print(f"report={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
