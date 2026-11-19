"""Render only complete, prediction-derived participant result matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from eeg_ma.artifacts import new_output_directory
from eeg_ma.constants import DEFAULT_CHANNELS, channel_pairs
from eeg_ma.historical_reporting import (BAND_EXPERIMENTS, CLASSIFIERS, STORY_EXPERIMENTS,
    rebuild_historical_results, read_historical_report, validate_historical_report, write_report_contents)
from eeg_ma.result_contract import ResultContractError, validate_result_matrix
from eeg_ma.provenance import protect_output, sha256_file


CHANNEL_POSITIONS = {"FP1": (-0.42, 0.78), "FP2": (0.42, 0.78), "F7": (-0.88, 0.28),
    "F3": (-0.42, 0.34), "Fz": (0.0, 0.38), "F4": (0.42, 0.34), "F8": (0.88, 0.28)}
LABELS = {"poster_baseline": "Poster baseline\nBP+COH", "e1_asymmetry": "+ Asymmetry",
    "e2_plv": "+ PLV", "plv_all_bands": "PLV all bands", "plv_alpha_only": "Alpha PLV only",
    "iplv_only": "iPLV all bands"}
METRIC_LABELS = {"accuracy": "Participant accuracy", "balanced_accuracy": "Participant balanced accuracy"}


def _context(table: pd.DataFrame) -> dict:
    required = {"analysis_id", "run_id", "cohort_id", "protocol_id", "experiment", "classifier", "comparison"}
    if table.empty or required - set(table) or table.columns.duplicated().any():
        raise ResultContractError("Figure inputs require identified prediction-derived results")
    for column in required:
        if any(not isinstance(value, str) or not value or value != value.strip() for value in table[column]):
            raise ResultContractError(f"Figure inputs require literal nonempty {column} identities")
    for column in ("analysis_id", "cohort_id", "protocol_id"):
        if table[column].nunique(dropna=False) != 1 or not isinstance(table[column].iloc[0], str) or not table[column].iloc[0].strip():
            raise ResultContractError(f"Figure inputs mix or omit {column}")
    if table[list(required)].isna().any().any():
        raise ResultContractError("Figure input result identities are missing")
    if (table.groupby("experiment").run_id.nunique() != 1).any():
        raise ResultContractError("Figure inputs mix different parent runs for one experiment")
    return {column: table[column].iloc[0] for column in ("analysis_id", "cohort_id", "protocol_id")}


def _summary_cells(table, experiments, *, metric):
    if metric not in METRIC_LABELS:
        raise ResultContractError("Figure metric must be accuracy or balanced_accuracy")
    _context(table)
    keys = ["experiment", "classifier", "comparison"]
    if table.duplicated(keys).any():
        raise ResultContractError("Figure summary contains duplicate matrix keys")
    needed = [f"{metric}_mean", f"{metric}_std", "n_subjects", "n_ba_eligible", "aggregation", "uncertainty"]
    if set(needed) - set(table):
        raise ResultContractError("Figure summary lacks participant metric/N/SD metadata")
    work = table.loc[table.comparison.eq("rest1_vs_type1") & table.experiment.isin(experiments)].copy()
    expected = pd.DataFrame([{"experiment": experiment, "classifier": classifier, "comparison": "rest1_vs_type1"}
                             for experiment in experiments for classifier in CLASSIFIERS])
    validate_result_matrix(work, expected, keys=keys)
    numeric = work[[f"{metric}_mean", f"{metric}_std", "n_subjects", "n_ba_eligible"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not work[f"{metric}_mean"].between(0, 1).all() or (work[f"{metric}_std"] < 0).any():
        raise ResultContractError("Figure metric or participant SD is undefined/nonfinite or outside its range")
    if work.n_subjects.nunique() != 1 or (work.n_subjects < 2).any() or (work.n_subjects % 1 != 0).any():
        raise ResultContractError("Figure cells require the same participant N and an estimable sample SD")
    if metric == "balanced_accuracy" and not work.n_ba_eligible.eq(work.n_subjects).all():
        raise ResultContractError("Balanced-accuracy figure requires both classes for every paired participant")
    if not work.aggregation.eq("equal_participant").all() or not work.uncertainty.eq("sample_sd_descriptive_no_ci").all():
        raise ResultContractError("Figure aggregation or uncertainty convention is incompatible")
    return work


def _save(fig, base, *, source, metric, metadata=None, uncertainty="between-participant sample SD; descriptive, no CI"):
    try:
        base = protect_output(base)
        if any(base.with_suffix(suffix).exists() for suffix in (".png", ".svg", ".csv", ".json")):
            raise FileExistsError("Figure artifacts already exist; choose a new immutable output directory")
    except (ValueError, FileExistsError):
        plt.close(fig)
        raise
    meta = metadata or {}
    required = {"cohort_id", "protocol_id", "analysis_id", "source_sha256", "subject_ids_by_comparison", "n_subjects_by_comparison"}
    if required - set(meta) or "comparison" not in source or source.comparison.nunique() != 1:
        plt.close(fig)
        raise ResultContractError("Figure publication requires source-hash metadata and a declared participant roster by comparison")
    if not isinstance(meta["source_sha256"], str) or len(meta["source_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in meta["source_sha256"]):
        plt.close(fig)
        raise ResultContractError("Figure source SHA-256 metadata is invalid")
    comparison = source.comparison.iloc[0]
    subjects = meta["subject_ids_by_comparison"].get(comparison)
    if not isinstance(subjects, list) or not subjects or len(subjects) != len(set(subjects)):
        plt.close(fig)
        raise ResultContractError("Figure participant roster is missing or duplicate")
    n = len(subjects)
    cohort_subjects = set().union(*(set(ids) for ids in meta["subject_ids_by_comparison"].values()))
    if meta.get("n_subjects") != len(cohort_subjects):
        plt.close(fig)
        raise ResultContractError("Figure metadata total N contradicts its participant roster")
    if meta["n_subjects_by_comparison"].get(comparison) != n:
        plt.close(fig)
        raise ResultContractError("Figure metadata N contradicts its participant roster")
    if "n_subjects" in source and not source.n_subjects.eq(n).all():
        plt.close(fig)
        raise ResultContractError("Figure plotted N differs from its declared participant roster")
    if "subject_id" in source and set(source.subject_id) != set(subjects):
        plt.close(fig)
        raise ResultContractError("Figure plotted subjects differ from its declared participant roster")
    for column in ("cohort_id", "protocol_id", "analysis_id"):
        if column in source and set(source[column]) != {meta[column]}:
            plt.close(fig)
            raise ResultContractError("Figure source cohort/protocol/analysis identity differs from its metadata")
    cohort, protocol = meta["cohort_id"], meta["protocol_id"]
    caption = f"Metric: {metric}; N={n} participants; {uncertainty}. Cohort: {cohort}; protocol: {protocol}."
    if meta.get("source_sha256"):
        caption += f" Source SHA-256: {meta['source_sha256']}."
    if meta.get("scope") == "partial_historical_matrix":
        caption = "PARTIAL HISTORICAL REPORT. " + caption
    fig.text(0.02, -0.01, textwrap.fill(caption, 120), ha="left", va="top", fontsize=7)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(suffix) for suffix in (".png", ".svg")]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    source.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    sidecar = {"metric": metric, "n_subjects": int(n) if isinstance(n, (int, np.integer)) else n,
        "uncertainty": uncertainty, "cohort_id": cohort, "protocol_id": protocol,
        "analysis_id": meta.get("analysis_id"), "scope": meta.get("scope"),
        "comparison": comparison, "subject_ids": subjects,
        "source_sha256": meta.get("source_sha256"), "numerical_source": base.with_suffix(".csv").name,
        "numerical_source_sha256": sha256_file(base.with_suffix(".csv"))}
    base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def _grouped_accuracy(table, experiments, labels, title, out_base, *, metric="accuracy", metadata=None):
    work = _summary_cells(table, experiments, metric=metric)
    values = work.set_index(["experiment", "classifier"])
    x = np.arange(len(experiments), dtype=float)
    width = 0.8 / len(CLASSIFIERS)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(experiments)), 5.2))
    for i, classifier in enumerate(CLASSIFIERS):
        rows = values.loc[[(experiment, classifier) for experiment in experiments]]
        ax.bar(x + (i - 1) * width, rows[f"{metric}_mean"].to_numpy(), width=width,
               yerr=rows[f"{metric}_std"].to_numpy(), capsize=3, label=classifier)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x, [labels.get(e, e) for e in experiments], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_base, source=work, metric=metric, metadata=metadata)


def plot_analysis_story(extension_summary, decomposition_summary, final_summary, out_dir, *, metric="accuracy", metadata=None):
    # Every role has one source; duplicate cells must never be resolved by order.
    for table in (extension_summary, decomposition_summary, final_summary):
        _context(table)
        if table.duplicated(["experiment", "classifier", "comparison"]).any():
            raise ResultContractError("Analysis-story source has duplicate result cells")
    story = pd.concat([
        extension_summary.loc[extension_summary.experiment.isin(["poster_baseline", "e1_asymmetry", "e2_plv"])],
        decomposition_summary.loc[decomposition_summary.experiment.eq("plv_all_bands")],
        final_summary.loc[final_summary.experiment.isin(["plv_alpha_only", "iplv_only"])],
    ], ignore_index=True)
    return _grouped_accuracy(story, list(STORY_EXPERIMENTS), LABELS,
        "Rest 1 vs Type 1 — analysis progression", out_dir / "final_story_type1_inter", metric=metric, metadata=metadata)


def plot_band_ranking(final_summary, out_dir, *, metric="accuracy", metadata=None):
    labels = dict(zip(BAND_EXPERIMENTS, ["delta", "theta", "alpha", "beta-low", "beta-high", "gamma"]))
    return _grouped_accuracy(final_summary, list(BAND_EXPERIMENTS), labels,
        "Rest 1 vs Type 1 — single-band PLV comparison", out_dir / "final_plv_band_comparison_type1", metric=metric, metadata=metadata)


def plot_phase_robustness(final_summary, out_dir, *, metric="accuracy", metadata=None):
    return _grouped_accuracy(final_summary, ["poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"], LABELS,
        "Rest 1 vs Type 1 — phase-connectivity validation", out_dir / "final_phase_connectivity_validation", metric=metric, metadata=metadata)


def plot_alpha_subjects(delta_by_unit, out_dir, *, metric="accuracy", metadata=None):
    _context(delta_by_unit)
    keys = ["experiment", "comparison", "classifier", "subject_id"]
    if delta_by_unit.duplicated(keys).any():
        raise ResultContractError("Paired participant figure has duplicate result keys")
    needed = ["reference_experiment", "reference_run_id", "eligible_epochs_sha256", "eligible_epoch_manifest_id", f"{metric}_reference", f"{metric}_experiment"]
    if set(needed) - set(delta_by_unit):
        raise ResultContractError("Paired participant figure lacks exact reference and eligible-epoch identity")
    work = delta_by_unit.loc[delta_by_unit.experiment.eq("plv_alpha_only") & delta_by_unit.comparison.eq("rest1_vs_type1")].copy()
    if work.empty or not work.reference_experiment.eq("poster_baseline").all() or work.reference_run_id.nunique() != 1:
        raise ResultContractError("Paired participant figure has missing or incompatible reference results")
    if not metadata or "rest1_vs_type1" not in metadata.get("subject_ids_by_comparison", {}):
        raise ResultContractError("Paired participant figure requires its independent expected subject roster")
    expected = pd.DataFrame([{"subject_id": subject, "classifier": classifier}
                            for subject in metadata["subject_ids_by_comparison"]["rest1_vs_type1"] for classifier in CLASSIFIERS])
    validate_result_matrix(work, expected, keys=["subject_id", "classifier"])
    if (work.groupby("subject_id")[["eligible_epochs_sha256", "eligible_epoch_manifest_id"]].nunique() != 1).any().any():
        raise ResultContractError("Paired participant classifier cells do not share identical eligible epochs")
    columns = [f"{metric}_reference", f"{metric}_experiment"]
    if not np.isfinite(work[columns].to_numpy(dtype=float)).all() or not work[columns].apply(lambda c: c.between(0, 1).all()).all():
        raise ResultContractError("Paired participant figure contains undefined or invalid metrics")
    subject = work.groupby("subject_id", as_index=False)[columns].mean().sort_values("subject_id")
    subject["comparison"] = "rest1_vs_type1"
    x = np.arange(len(subject), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.bar(x - 0.18, subject[columns[0]], width=0.36, label="Poster baseline")
    ax.bar(x + 0.18, subject[columns[1]], width=0.36, label="Alpha PLV only")
    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x, subject.subject_id)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (mean across 3 classifiers)")
    ax.set_title("Rest 1 vs Type 1 — held-out subject comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_dir / "final_alpha_vs_baseline_by_subject", source=subject,
        metric=f"{metric}; mean across the three prespecified classifiers", metadata=metadata,
        uncertainty="participant values; no uncertainty bars or CI")


def plot_alpha_network(freq, out_dir, top_n=12, *, metadata=None):
    keys = ["comparison", "feature"]
    if set(keys + ["selection_count"]) - set(freq) or freq.duplicated(keys).any():
        raise ResultContractError("Alpha network contains missing or duplicate feature-frequency keys")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ResultContractError("Network top_n must be a positive integer")
    work = freq.loc[freq.comparison.eq("rest1_vs_type1")].copy()
    if work.empty or not np.isfinite(work.selection_count.to_numpy(dtype=float)).all() or (work.selection_count <= 0).any() or (work.selection_count % 1 != 0).any():
        raise ResultContractError("Alpha network has missing or invalid selection counts")
    if metadata and (work.selection_count > metadata["n_subjects_by_comparison"]["rest1_vs_type1"]).any():
        raise ResultContractError("Alpha selection frequency exceeds held-out participant N")
    work = work.sort_values(["selection_count", "feature"], ascending=[False, True]).head(top_n)
    parsed = []
    for row in work.to_dict("records"):
        parts = row["feature"].split("__")
        if len(parts) != 4 or parts[0] != "PLV" or parts[3] != "alpha" or tuple(parts[1:3]) not in channel_pairs(DEFAULT_CHANNELS):
            raise ResultContractError("Alpha network contains an invalid channel pair or band")
        parsed.append((parts[1], parts[2], row["selection_count"]))
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.add_patch(Circle((0.0, 0.15), 1.05, fill=False, linewidth=1.5))
    ax.plot([-0.12, 0.0, 0.12], [1.15, 1.28, 1.15], linewidth=1.2)
    for channel, (x, y) in CHANNEL_POSITIONS.items():
        ax.scatter([x], [y], s=90, zorder=3)
        ax.text(x, y + 0.08, channel, ha="center", va="bottom", fontsize=10)
    maximum = max(count for _, _, count in parsed)
    for first, second, count in parsed:
        x1, y1 = CHANNEL_POSITIONS[first]
        x2, y2 = CHANNEL_POSITIONS[second]
        ax.plot([x1, x2], [y1, y2], linewidth=1 + 3 * count / maximum, alpha=0.55, zorder=1)
        ax.text((x1+x2)/2, (y1+y2)/2, f"alpha×{int(count)}", fontsize=7, ha="center", va="center")
    ax.set_title("Alpha-PLV SFS network — Rest 1 vs Type 1")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.95, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    return _save(fig, out_dir / "final_alpha_plv_sfs_network", source=work,
        metric="training SFS selection count across held-out participant folds", metadata=metadata,
        uncertainty="descriptive feature-selection frequency; no CI")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild participant results from saved predictions and create strict descriptive figures; no retraining")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="outputs/figures/final")
    parser.add_argument("--allow-legacy-results", action="store_true")
    parser.add_argument("--legacy-manifest", default=None, help="Explicit legacy feature/session mapping; required for historical feature tables")
    parser.add_argument("--baseline-features", default="data/features/features_all.csv")
    parser.add_argument("--extension-features", default="data/features/extensions/features_extensions.csv")
    parser.add_argument("--iplv-features", default="data/features/final_validation/features_iplv.csv")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--partial-report", action="store_true", help="Explicit incomplete scope; always write a new immutable directory")
    parser.add_argument("--metric", choices=list(METRIC_LABELS), default="accuracy")
    parser.add_argument("--network-top-n", type=int, default=12)
    args = parser.parse_args()
    report = rebuild_historical_results(args.results_root, baseline_features=args.baseline_features,
        extension_features=args.extension_features, iplv_features=args.iplv_features,
        legacy_manifest=args.legacy_manifest, allow_legacy=args.allow_legacy_results,
        experiments=args.experiments, partial=args.partial_report)
    validate_historical_report(report)
    out = Path(args.out_dir)
    available = set(report.metadata["selected_experiments"])
    options = {"metric": args.metric, "metadata": report.metadata}
    with new_output_directory(out) as stage:
        made = []
        if set(STORY_EXPERIMENTS) <= available:
            made += plot_analysis_story(report.summary, report.summary, report.summary, stage, **options)
        if set(BAND_EXPERIMENTS) <= available:
            made += plot_band_ranking(report.summary, stage, **options)
        if {"poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"} <= available:
            made += plot_phase_robustness(report.summary, stage, **options)
        if "plv_alpha_only" in available:
            made += plot_alpha_subjects(report.paired_deltas, stage, **options)
            made += plot_alpha_network(report.alpha_frequency, stage, top_n=args.network_top_n, metadata=report.metadata)
        if not report.metadata["study_matrix_complete"]:
            made += _grouped_accuracy(report.summary, report.metadata["selected_experiments"], LABELS,
                "Rest 1 vs Type 1 — PARTIAL historical report", stage / "partial_historical_comparison", **options)
        (stage / "figure_sources.json").write_text(json.dumps({**report.metadata, "figure_metric": args.metric,
            "figures": [p.stem for p in made if p.suffix == ".png"],
            "performance_source": "validated epoch predictions → participant results → equal-participant summary",
            "network_source": "training selections from folds matched to validated prediction coverage; recorded frequencies checked against fold selections"},
            indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        write_report_contents(report, stage)
        read_historical_report(stage, verify_sources=False)
        made = [out / path.relative_to(stage) for path in made]
    print(f"report_scope={report.metadata['scope']}")
    for path in made:
        print(path)
    print(f"count={len(made)} files ({len(made)//2} figures, PNG+SVG); metric={args.metric}; error bars=participant SD, no LOPO CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
