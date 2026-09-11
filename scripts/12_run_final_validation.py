from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.evaluation import run_inter_subject
from eeg_ma.extensions import merge_baseline_and_extensions
from eeg_ma.extract import feature_columns


UNIT_KEYS = ["unit_id", "comparison", "classifier"]


def _units_from_inter_folds(folds: pd.DataFrame) -> pd.DataFrame:
    units = folds[
        ["held_out_subject", "comparison", "classifier", "accuracy"]
    ].rename(columns={"held_out_subject": "unit_id"}).copy()
    units["unit_id"] = units["unit_id"].astype(str)
    if units.duplicated(UNIT_KEYS).any():
        raise ValueError("Inter-subject unit table contains duplicate keys")
    return units


def _units_from_root(root: Path) -> pd.DataFrame:
    path = root / "inter" / "inter_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference inter folds: {path}")
    return _units_from_inter_folds(pd.read_csv(path, encoding="utf-8-sig"))


def _summary(
    units: pd.DataFrame,
    experiment: str,
    feature_set: str,
    n_candidate_features: int,
    role: str,
) -> pd.DataFrame:
    out = (
        units.groupby(["comparison", "classifier"], dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_units"})
    )
    out.insert(0, "n_candidate_features", int(n_candidate_features))
    out.insert(0, "feature_set", feature_set)
    out.insert(0, "experiment", experiment)
    out.insert(0, "role", role)
    return out


def _paired_delta(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    experiment: str,
    reference_name: str,
) -> pd.DataFrame:
    merged = reference.merge(
        target,
        on=UNIT_KEYS,
        how="inner",
        suffixes=("_reference", "_experiment"),
        validate="one_to_one",
    )
    if len(merged) != len(target) or len(merged) != len(reference):
        raise ValueError(
            f"Cannot pair {experiment} against {reference_name}: "
            f"target={len(target)}, reference={len(reference)}, paired={len(merged)}"
        )
    merged["delta_accuracy"] = merged["accuracy_experiment"] - merged["accuracy_reference"]
    merged.insert(0, "reference", reference_name)
    merged.insert(0, "experiment", experiment)
    return merged[
        [
            "experiment", "reference", "unit_id", "comparison", "classifier",
            "accuracy_reference", "accuracy_experiment", "delta_accuracy",
        ]
    ]


def _delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    return (
        deltas.groupby(
            ["experiment", "reference", "comparison", "classifier"],
            dropna=False,
        )["delta_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "n_units"})
    )


def _select_experiments(matrix: Dict, requested: Sequence[str] | None) -> List[Dict]:
    experiments = list(matrix.get("experiments", []))
    by_name = {str(x["name"]): x for x in experiments}
    if requested:
        unknown = [x for x in requested if x not in by_name]
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}; available={sorted(by_name)}")
        return [by_name[x] for x in requested]
    return experiments


def _band_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    bands = summary.loc[
        (summary["role"] == "experiment")
        & summary["experiment"].str.startswith("plv_")
        & summary["experiment"].str.endswith("_only")
    ].copy()
    if bands.empty:
        return pd.DataFrame()

    # Descriptive only: average the three classifier means so the report can
    # compare bands without selecting one classifier post hoc.
    ranked = (
        bands.groupby(["experiment", "comparison", "n_candidate_features"], dropna=False)["accuracy_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_across_classifier_means",
                "std": "std_across_classifier_means",
                "count": "n_classifiers",
            }
        )
    )
    ranked["band"] = (
        ranked["experiment"]
        .str.removeprefix("plv_")
        .str.removesuffix("_only")
    )
    return ranked.sort_values(
        ["comparison", "mean_across_classifier_means"],
        ascending=[True, False],
    ).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Final bounded validation: six PLV band-only ablations plus iPLV-only, "
            "using the frozen inter-subject LOPO pipeline"
        )
    )
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--iplv-features", default="data/features/final_validation/features_iplv.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--validation-config", default="configs/final_validation.json")
    p.add_argument("--out-root", default="results/final_validation")
    p.add_argument("--poster-results-root", default=None)
    p.add_argument("--plv-only-results-root", default=None)
    p.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional exact experiment names; default runs the full bounded matrix.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an experiment when its inter/inter_folds.csv already exists.",
    )
    args = p.parse_args()

    baseline = pd.read_csv(args.baseline_features, encoding="utf-8-sig")
    extensions = pd.read_csv(args.extension_features, encoding="utf-8-sig")
    iplv = pd.read_csv(args.iplv_features, encoding="utf-8-sig")
    merged = merge_baseline_and_extensions(baseline, extensions)
    merged = merge_baseline_and_extensions(merged, iplv)

    cfg_base = load_config(args.config)
    matrix_path = Path(args.validation_config)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    experiments = _select_experiments(matrix, args.experiments)
    if not experiments:
        raise ValueError("No final-validation experiments selected")

    refs = matrix.get("references", {})
    poster_root = Path(args.poster_results_root or refs.get("poster_baseline_results_root", "results"))
    plv_root = Path(args.plv_only_results_root or refs.get("plv_only_results_root", "results/plv_decomposition/e2a_plv_only"))

    reference_defs = [
        ("poster_baseline", "baseline", poster_root),
        ("plv_all_bands", "plv", plv_root),
    ]
    reference_units: Dict[str, pd.DataFrame] = {}
    summary_frames: List[pd.DataFrame] = []

    for label, feature_set, root in reference_defs:
        units = _units_from_root(root)
        reference_units[label] = units
        summary_frames.append(
            _summary(
                units,
                label,
                feature_set,
                len(feature_columns(merged, feature_set)),
                role="reference",
            )
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "baseline_config_path": str(args.config),
        "validation_config_path": str(matrix_path),
        "poster_results_root": str(poster_root),
        "plv_only_results_root": str(plv_root),
        "scope": "inter_only",
        "selected_experiments": experiments,
        "stop_rule": matrix.get("reporting", {}).get("stop_rule"),
    }
    (out_root / "experiment_matrix_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    delta_poster_frames: List[pd.DataFrame] = []
    delta_plv_frames: List[pd.DataFrame] = []

    print("=== FINAL BOUNDED PHASE VALIDATION ===")
    print(f"rows={len(merged)} subjects={merged['subject_id'].nunique()}")
    print("scope=inter-subject LOPO only")
    print("references=poster_baseline, plv_all_bands")

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

        folds_path = root / "inter" / "inter_folds.csv"
        print(f"\n--- {name}: feature_set={feature_set}, candidates={n_features} ---")
        if args.resume and folds_path.exists():
            print("[resume] Reusing existing inter folds")
            folds = pd.read_csv(folds_path, encoding="utf-8-sig")
        else:
            folds, _ = run_inter_subject(merged, cfg, root / "inter")

        units = _units_from_inter_folds(folds)
        summary_frames.append(
            _summary(units, name, feature_set, n_features, role="experiment")
        )
        delta_poster_frames.append(
            _paired_delta(units, reference_units["poster_baseline"], name, "poster_baseline")
        )
        delta_plv_frames.append(
            _paired_delta(units, reference_units["plv_all_bands"], name, "plv_all_bands")
        )

    summary = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["comparison", "classifier", "role", "experiment"]
    )
    summary.to_csv(out_root / "final_validation_summary.csv", index=False, encoding="utf-8-sig")

    delta_poster = pd.concat(delta_poster_frames, ignore_index=True)
    delta_plv = pd.concat(delta_plv_frames, ignore_index=True)
    delta_poster.to_csv(out_root / "delta_vs_poster_by_unit.csv", index=False, encoding="utf-8-sig")
    delta_plv.to_csv(out_root / "delta_vs_plv_all_by_unit.csv", index=False, encoding="utf-8-sig")

    delta_poster_summary = _delta_summary(delta_poster)
    delta_plv_summary = _delta_summary(delta_plv)
    delta_poster_summary.to_csv(out_root / "delta_vs_poster_summary.csv", index=False, encoding="utf-8-sig")
    delta_plv_summary.to_csv(out_root / "delta_vs_plv_all_summary.csv", index=False, encoding="utf-8-sig")

    band_ranking = _band_ranking(summary)
    band_ranking.to_csv(out_root / "plv_band_descriptive_ranking.csv", index=False, encoding="utf-8-sig")

    primary = str(matrix.get("reporting", {}).get("primary_comparison", "rest1_vs_type1"))
    print("\n=== FINAL VALIDATION SUMMARY ===")
    print(summary.to_string(index=False))

    print(f"\n=== INTER {primary}: DELTA VS POSTER BASELINE ===")
    x = delta_poster_summary.loc[delta_poster_summary["comparison"] == primary]
    print(x.to_string(index=False))

    print(f"\n=== INTER {primary}: DELTA VS ALL-BAND PLV ===")
    x = delta_plv_summary.loc[delta_plv_summary["comparison"] == primary]
    print(x.to_string(index=False))

    print("\n=== PLV BAND DESCRIPTIVE RANKING ===")
    print(band_ranking.to_string(index=False))

    print("\nSTOP RULE: after interpreting this matrix, stop feature-engineering experiments on these same five subjects and move to synthesis/figures/reporting.")
    print(f"\nsummary={out_root / 'final_validation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
