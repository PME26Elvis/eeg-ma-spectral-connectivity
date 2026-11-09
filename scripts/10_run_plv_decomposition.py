from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_inter_subject, run_intra_subject
from eeg_ma.extensions import merge_baseline_and_extensions
from eeg_ma.extract import feature_columns
from eeg_ma.experiment_runs import add_identity_arguments, load_features, prepare_matrix, source_inputs
from eeg_ma.provenance import read_csv_literal


UNIT_KEYS = ["scope", "unit_id", "comparison", "classifier"]


def _feature_family(name: str) -> str:
    name = str(name)
    if name.startswith("BP__"):
        return "BP"
    if name.startswith("COH__"):
        return "COH"
    if name.startswith("PLV__"):
        return "PLV"
    if name.startswith("ASYM_"):
        return "ASYM"
    return "OTHER"


def _accepted_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _units_from_folds(intra_folds: pd.DataFrame, inter_folds: pd.DataFrame) -> pd.DataFrame:
    intra = (
        intra_folds.groupby(["subject_id", "comparison", "classifier"], dropna=False)["accuracy"]
        .mean()
        .reset_index(name="accuracy")
        .rename(columns={"subject_id": "unit_id"})
    )
    intra.insert(0, "scope", "intra")

    inter = inter_folds[
        ["held_out_subject", "comparison", "classifier", "accuracy"]
    ].rename(columns={"held_out_subject": "unit_id"}).copy()
    inter.insert(0, "scope", "inter")

    units = pd.concat([intra, inter], ignore_index=True)
    units["unit_id"] = units["unit_id"].astype(str)
    if units.duplicated(UNIT_KEYS).any():
        raise ValueError("Unit-level accuracy table contains duplicate keys")
    return units


def _units_from_result_root(root: Path) -> pd.DataFrame:
    intra_path = root / "intra" / "intra_folds.csv"
    inter_path = root / "inter" / "inter_folds.csv"
    if not intra_path.exists() or not inter_path.exists():
        raise FileNotFoundError(
            f"Reference result root {root} must contain intra/intra_folds.csv and inter/inter_folds.csv"
        )
    intra = read_csv_literal(intra_path, encoding="utf-8-sig")
    inter = read_csv_literal(inter_path, encoding="utf-8-sig")
    return _units_from_folds(intra, inter)


def _summary_from_units(
    units: pd.DataFrame,
    experiment: str,
    feature_set: str,
    n_candidate_features: int,
    role: str,
) -> pd.DataFrame:
    summary = (
        units.groupby(["scope", "comparison", "classifier"], dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_units"})
    )
    summary.insert(0, "n_candidate_features", int(n_candidate_features))
    summary.insert(0, "feature_set", feature_set)
    summary.insert(0, "experiment", experiment)
    summary.insert(0, "role", role)
    return summary


def _paired_delta(
    target_units: pd.DataFrame,
    reference_units: pd.DataFrame,
    experiment: str,
    reference: str,
) -> pd.DataFrame:
    merged = reference_units.merge(
        target_units,
        on=UNIT_KEYS,
        how="inner",
        suffixes=("_reference", "_experiment"),
        validate="one_to_one",
    )
    if len(merged) != len(target_units) or len(merged) != len(reference_units):
        raise ValueError(
            f"Cannot pair {experiment} against {reference}: "
            f"target={len(target_units)}, reference={len(reference_units)}, paired={len(merged)}"
        )
    merged["delta_accuracy"] = merged["accuracy_experiment"] - merged["accuracy_reference"]
    merged.insert(0, "reference", reference)
    merged.insert(0, "experiment", experiment)
    return merged[
        [
            "experiment", "reference", "scope", "unit_id", "comparison", "classifier",
            "accuracy_reference", "accuracy_experiment", "delta_accuracy",
        ]
    ]


def _delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    return (
        deltas.groupby(
            ["experiment", "reference", "scope", "comparison", "classifier"],
            dropna=False,
        )["delta_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "n_units"})
    )


def _sfs_family_usage(experiment: str, root: Path, scope: str) -> pd.DataFrame:
    if scope == "intra":
        path = root / "intra" / "intra_sfs_path.csv"
        unit_cols = ["subject_id", "outer_fold"]
    elif scope == "inter":
        path = root / "inter" / "inter_sfs_path.csv"
        unit_cols = ["held_out_subject", "outer_fold"]
    else:
        raise ValueError(scope)

    if not path.exists():
        return pd.DataFrame()
    sfs = read_csv_literal(path, encoding="utf-8-sig")
    if sfs.empty:
        return pd.DataFrame()

    feature_col = "candidate_name" if "candidate_name" in sfs.columns else "feature"
    if feature_col not in sfs.columns or "accepted" not in sfs.columns:
        raise ValueError(f"Unexpected SFS path schema: {path}")

    totals = (
        sfs[["comparison"] + unit_cols]
        .drop_duplicates()
        .groupby("comparison")
        .size()
        .to_dict()
    )
    accepted = sfs.loc[_accepted_mask(sfs["accepted"])].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["family"] = accepted[feature_col].map(_feature_family)

    rows: List[Dict[str, object]] = []
    for (comparison, family), g in accepted.groupby(["comparison", "family"], dropna=False):
        fold_presence = len(g[["comparison"] + unit_cols].drop_duplicates())
        total = int(totals[str(comparison)])
        rows.append(
            {
                "experiment": experiment,
                "scope": scope,
                "comparison": comparison,
                "family": family,
                "selection_occurrences": int(len(g)),
                "fold_presence_count": int(fold_presence),
                "n_outer_units": total,
                "fold_presence_fraction": float(fold_presence / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _read_folds(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_csv_literal(root / "intra" / "intra_folds.csv", encoding="utf-8-sig"),
        read_csv_literal(root / "inter" / "inter_folds.csv", encoding="utf-8-sig"),
    )


def _select_experiments(matrix: Dict, include_secondary: bool, requested: Sequence[str] | None) -> List[Dict]:
    primary = list(matrix.get("primary_experiments", []))
    secondary = list(matrix.get("secondary_experiments", []))
    all_experiments = primary + secondary
    by_name = {str(x["name"]): x for x in all_experiments}

    if requested:
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}; available={sorted(by_name)}")
        return [by_name[name] for name in requested]

    return primary + (secondary if include_secondary else [])


def main() -> int:
    p = argparse.ArgumentParser(description="Run PLV decomposition with immutable runs and inventory aggregation")
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--experiment-config", default="configs/plv_decomposition.json")
    p.add_argument("--out-root", default="outputs/results/plv_decomposition")
    p.add_argument("--poster-results-root", default=None)
    p.add_argument("--e2-full-results-root", default=None)
    p.add_argument("--include-secondary", action="store_true")
    p.add_argument("--experiments", nargs="*", default=None)
    add_identity_arguments(p)
    args = p.parse_args()
    baseline = load_features(args.baseline_features, args.legacy_manifest)
    extensions = load_features(args.extension_features, args.legacy_manifest)
    merged = merge_baseline_and_extensions(baseline, extensions)
    cfg = load_config(args.config)
    spec = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    selected = _select_experiments(spec, args.include_secondary, args.experiments)
    all_experiments = spec.get("primary_experiments", []) + spec.get("secondary_experiments", [])
    refs = spec.get("references", {})
    reference_defs = [
        ("poster_baseline", "baseline", Path(args.poster_results_root or refs.get("poster_baseline_results_root", "results"))),
        ("e2_full_baseline_plv", "baseline_plv", Path(args.e2_full_results_root or refs.get("e2_full_results_root", "results/extensions/e2_plv"))),
    ]
    matrix, evaluator = prepare_matrix(merged, cfg, experiments=all_experiments, selected=[x["name"] for x in selected],
        scopes=("intra", "inter"), out_root=args.out_root,
        input_paths=source_inputs({"baseline_features": args.baseline_features, "extension_features": args.extension_features,
            "config": args.config, "experiment_config": args.experiment_config}, args.legacy_manifest),
        reference_paths={name: root for name, _, root in reference_defs}, matrix_config=spec, resume=args.resume)
    reference_units = {label: _units_from_result_root(root) for label, _, root in reference_defs}
    by_name = {x["name"]: x for x in all_experiments}
    def report(inventory):
        summaries, families, poster_delta, e2_delta = [], [], [], []
        for label, feature_set, root in reference_defs:
            summaries.append(_summary_from_units(reference_units[label], label, feature_set, len(feature_columns(merged, feature_set)), "reference"))
            for scope in ("inter", "intra"):
                usage = _sfs_family_usage(label, root, scope)
                if not usage.empty:
                    families.append(usage)
        for name, root in inventory.items():
            feature_set = by_name[name]["feature_set"]
            units = _units_from_folds(*_read_folds(root))
            summaries.append(_summary_from_units(units, name, feature_set, len(feature_columns(merged, feature_set)), "experiment"))
            poster_delta.append(_paired_delta(units, reference_units["poster_baseline"], name, "poster_baseline"))
            e2_delta.append(_paired_delta(units, reference_units["e2_full_baseline_plv"], name, "e2_full_baseline_plv"))
            for scope in ("inter", "intra"):
                usage = _sfs_family_usage(name, root, scope)
                if not usage.empty:
                    families.append(usage)
        poster = pd.concat(poster_delta, ignore_index=True)
        e2 = pd.concat(e2_delta, ignore_index=True)
        tables = {"plv_decomposition_summary.csv": pd.concat(summaries, ignore_index=True).sort_values(["scope", "comparison", "classifier", "role", "experiment"]),
            "delta_vs_poster_by_unit.csv": poster, "delta_vs_e2_full_by_unit.csv": e2,
            "delta_vs_poster_summary.csv": _delta_summary(poster), "delta_vs_e2_full_summary.csv": _delta_summary(e2)}
        if families:
            tables["sfs_feature_family_usage.csv"] = pd.concat(families, ignore_index=True).sort_values(["scope", "comparison", "experiment", "family"])
        return tables
    _, report_root = matrix.run(evaluator, report, required_experiments=[x["name"] for x in spec.get("primary_experiments", [])])
    print(f"report={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
