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
    intra = pd.read_csv(intra_path, encoding="utf-8-sig")
    inter = pd.read_csv(inter_path, encoding="utf-8-sig")
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
    sfs = pd.read_csv(path, encoding="utf-8-sig")
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
        pd.read_csv(root / "intra" / "intra_folds.csv", encoding="utf-8-sig"),
        pd.read_csv(root / "inter" / "inter_folds.csv", encoding="utf-8-sig"),
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
    p = argparse.ArgumentParser(
        description=(
            "Run E2a/E2b/E2c PLV decomposition plus E3, using the same leakage-safe "
            "SFS/CV/classifier pipeline as the frozen Poster/Lab baseline"
        )
    )
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--experiment-config", default="configs/plv_decomposition.json")
    p.add_argument("--out-root", default="results/plv_decomposition")
    p.add_argument("--poster-results-root", default=None)
    p.add_argument("--e2-full-results-root", default=None)
    p.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also run E2d asymmetry+PLV, which is secondary/exploratory rather than a primary decomposition.",
    )
    p.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional exact experiment names; overrides the default primary/secondary selection.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an experiment if both intra_folds.csv and inter_folds.csv already exist under its output root.",
    )
    args = p.parse_args()

    baseline = pd.read_csv(args.baseline_features, encoding="utf-8-sig")
    extensions = pd.read_csv(args.extension_features, encoding="utf-8-sig")
    merged = merge_baseline_and_extensions(baseline, extensions)

    cfg_base = load_config(args.config)
    matrix_path = Path(args.experiment_config)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    experiments = _select_experiments(matrix, args.include_secondary, args.experiments)
    if not experiments:
        raise ValueError("No decomposition experiments selected")

    refs = matrix.get("references", {})
    poster_root = Path(args.poster_results_root or refs.get("poster_baseline_results_root", "results"))
    e2_full_root = Path(args.e2_full_results_root or refs.get("e2_full_results_root", "results/extensions/e2_plv"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Reference results are never recomputed here; they remain frozen anchors.
    reference_defs = [
        ("poster_baseline", "baseline", poster_root),
        ("e2_full_baseline_plv", "baseline_plv", e2_full_root),
    ]
    reference_units: Dict[str, pd.DataFrame] = {}
    summary_frames: List[pd.DataFrame] = []
    family_frames: List[pd.DataFrame] = []

    for label, feature_set, root in reference_defs:
        units = _units_from_result_root(root)
        reference_units[label] = units
        n_features = len(feature_columns(merged, feature_set))
        summary_frames.append(_summary_from_units(units, label, feature_set, n_features, role="reference"))
        for scope in ("inter", "intra"):
            usage = _sfs_family_usage(label, root, scope)
            if not usage.empty:
                family_frames.append(usage)

    snapshot = {
        "baseline_config_path": str(args.config),
        "experiment_matrix_path": str(matrix_path),
        "poster_results_root": str(poster_root),
        "e2_full_results_root": str(e2_full_root),
        "selected_experiments": experiments,
    }
    (out_root / "experiment_matrix_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    delta_vs_poster: List[pd.DataFrame] = []
    delta_vs_e2: List[pd.DataFrame] = []

    print("=== PLV DECOMPOSITION EXPERIMENTS ===")
    print(f"rows={len(merged)} subjects={merged['subject_id'].nunique()}")
    print("Frozen references: poster_baseline, e2_full_baseline_plv")

    for experiment in experiments:
        name = str(experiment["name"])
        feature_set = str(experiment["feature_set"])
        n_features = len(feature_columns(merged, feature_set))
        root = out_root / name
        root.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(cfg_base)
        cfg["feature_set"] = feature_set
        (root / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "baseline_config": cfg,
                    "experiment": experiment,
                    "n_candidate_features": n_features,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        can_resume = (
            args.resume
            and (root / "intra" / "intra_folds.csv").exists()
            and (root / "inter" / "inter_folds.csv").exists()
        )
        print(f"\n--- {name}: feature_set={feature_set}, candidates={n_features} ---")
        if can_resume:
            print("[resume] Reusing existing folds")
            intra_folds, inter_folds = _read_folds(root)
        else:
            intra_folds, _ = run_intra_subject(merged, cfg, root / "intra")
            inter_folds, _ = run_inter_subject(merged, cfg, root / "inter")

        units = _units_from_folds(intra_folds, inter_folds)
        summary_frames.append(_summary_from_units(units, name, feature_set, n_features, role="experiment"))
        delta_vs_poster.append(
            _paired_delta(units, reference_units["poster_baseline"], name, "poster_baseline")
        )
        delta_vs_e2.append(
            _paired_delta(units, reference_units["e2_full_baseline_plv"], name, "e2_full_baseline_plv")
        )
        for scope in ("inter", "intra"):
            usage = _sfs_family_usage(name, root, scope)
            if not usage.empty:
                family_frames.append(usage)

    summary = pd.concat(summary_frames, ignore_index=True)
    scope_order = pd.Categorical(summary["scope"], categories=["inter", "intra"], ordered=True)
    summary = (
        summary.assign(_scope_order=scope_order)
        .sort_values(["_scope_order", "comparison", "classifier", "role", "experiment"])
        .drop(columns="_scope_order")
    )
    summary.to_csv(out_root / "plv_decomposition_summary.csv", index=False, encoding="utf-8-sig")

    poster_delta = pd.concat(delta_vs_poster, ignore_index=True)
    e2_delta = pd.concat(delta_vs_e2, ignore_index=True)
    poster_delta.to_csv(out_root / "delta_vs_poster_by_unit.csv", index=False, encoding="utf-8-sig")
    e2_delta.to_csv(out_root / "delta_vs_e2_full_by_unit.csv", index=False, encoding="utf-8-sig")
    poster_delta_summary = _delta_summary(poster_delta)
    e2_delta_summary = _delta_summary(e2_delta)
    poster_delta_summary.to_csv(out_root / "delta_vs_poster_summary.csv", index=False, encoding="utf-8-sig")
    e2_delta_summary.to_csv(out_root / "delta_vs_e2_full_summary.csv", index=False, encoding="utf-8-sig")

    if family_frames:
        family = pd.concat(family_frames, ignore_index=True).sort_values(
            ["scope", "comparison", "experiment", "family"]
        )
        family.to_csv(out_root / "sfs_feature_family_usage.csv", index=False, encoding="utf-8-sig")
    else:
        family = pd.DataFrame()

    print("\n=== PLV DECOMPOSITION SUMMARY ===")
    print(summary.to_string(index=False))

    primary_comp = str(matrix.get("reporting", {}).get("primary_comparison", "rest1_vs_type1"))
    print(f"\n=== INTER {primary_comp}: DELTA VS POSTER BASELINE ===")
    x = poster_delta_summary.loc[
        (poster_delta_summary["scope"] == "inter")
        & (poster_delta_summary["comparison"] == primary_comp)
    ]
    print(x.to_string(index=False))

    print(f"\n=== INTER {primary_comp}: DELTA VS E2 FULL (BP+COH+PLV) ===")
    x = e2_delta_summary.loc[
        (e2_delta_summary["scope"] == "inter")
        & (e2_delta_summary["comparison"] == primary_comp)
    ]
    print(x.to_string(index=False))

    if not family.empty:
        print(f"\n=== INTER {primary_comp}: SFS FEATURE-FAMILY USAGE ===")
        x = family.loc[(family["scope"] == "inter") & (family["comparison"] == primary_comp)]
        print(x.to_string(index=False))

    print(f"\nsummary={out_root / 'plv_decomposition_summary.csv'}")
    print(f"per-unit deltas={out_root / 'delta_vs_poster_by_unit.csv'}")
    print(f"family usage={out_root / 'sfs_feature_family_usage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
