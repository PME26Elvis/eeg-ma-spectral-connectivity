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
from eeg_ma.experiment_runs import add_identity_arguments, load_features, prepare_matrix, source_inputs
from eeg_ma.provenance import read_csv_literal


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
    return _units_from_inter_folds(read_csv_literal(path, encoding="utf-8-sig"))


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
    p = argparse.ArgumentParser(description="Final bounded phase validation with verified resume and atomic reports")
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    p.add_argument("--iplv-features", default="data/features/final_validation/features_iplv.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--validation-config", default="configs/final_validation.json")
    p.add_argument("--out-root", default="outputs/results/final_validation")
    p.add_argument("--poster-results-root", default=None)
    p.add_argument("--plv-only-results-root", default=None)
    p.add_argument("--experiments", nargs="*", default=None)
    add_identity_arguments(p)
    args = p.parse_args()
    baseline = load_features(args.baseline_features, args.legacy_manifest)
    extensions = load_features(args.extension_features, args.legacy_manifest)
    iplv = load_features(args.iplv_features, args.legacy_manifest)
    merged = merge_baseline_and_extensions(merge_baseline_and_extensions(baseline, extensions), iplv)
    cfg = load_config(args.config)
    spec = json.loads(Path(args.validation_config).read_text(encoding="utf-8"))
    selected = _select_experiments(spec, args.experiments)
    all_experiments = spec.get("experiments", [])
    refs = spec.get("references", {})
    reference_defs = [
        ("poster_baseline", "baseline", Path(args.poster_results_root or refs.get("poster_baseline_results_root", "results"))),
        ("plv_all_bands", "plv", Path(args.plv_only_results_root or refs.get("plv_only_results_root", "results/plv_decomposition/e2a_plv_only"))),
    ]
    matrix, evaluator = prepare_matrix(merged, cfg, experiments=all_experiments, selected=[x["name"] for x in selected],
        scopes=("inter",), out_root=args.out_root,
        input_paths=source_inputs({"baseline_features": args.baseline_features, "extension_features": args.extension_features,
            "iplv_features": args.iplv_features, "config": args.config, "validation_config": args.validation_config}, args.legacy_manifest),
        reference_paths={name: root for name, _, root in reference_defs}, matrix_config=spec, resume=args.resume)
    reference_units = {label: _units_from_root(root) for label, _, root in reference_defs}
    by_name = {x["name"]: x for x in all_experiments}
    def report(inventory):
        summaries, poster_delta, plv_delta = [], [], []
        for label, feature_set, _ in reference_defs:
            summaries.append(_summary(reference_units[label], label, feature_set, len(feature_columns(merged, feature_set)), "reference"))
        for name, root in inventory.items():
            feature_set = by_name[name]["feature_set"]
            units = _units_from_root(root)
            summaries.append(_summary(units, name, feature_set, len(feature_columns(merged, feature_set)), "experiment"))
            poster_delta.append(_paired_delta(units, reference_units["poster_baseline"], name, "poster_baseline"))
            plv_delta.append(_paired_delta(units, reference_units["plv_all_bands"], name, "plv_all_bands"))
        summary = pd.concat(summaries, ignore_index=True).sort_values(["comparison", "classifier", "role", "experiment"])
        poster = pd.concat(poster_delta, ignore_index=True)
        plv = pd.concat(plv_delta, ignore_index=True)
        tables = {"final_validation_summary.csv": summary,
            "delta_vs_poster_by_unit.csv": poster, "delta_vs_plv_all_by_unit.csv": plv,
            "delta_vs_poster_summary.csv": _delta_summary(poster), "delta_vs_plv_all_summary.csv": _delta_summary(plv)}
        ranking = _band_ranking(summary)
        if not ranking.empty:
            tables["plv_band_descriptive_ranking.csv"] = ranking
        return tables
    _, report_root = matrix.run(evaluator, report)
    print(f"report={report_root}")
    print("STOP RULE: no new feature-engineering searches on these same five subjects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
