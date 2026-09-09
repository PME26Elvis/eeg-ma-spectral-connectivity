from __future__ import annotations

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


def test_plotting_helpers_smoke(tmp_path: Path) -> None:
    out = tmp_path / "figures"

    intra = pd.DataFrame(
        {
            "comparison": ["rest1_vs_type1", "rest1_vs_type1", "rest1_vs_type1"],
            "classifier": ["LDA", "RBF-SVM", "KFDA"],
            "participant_accuracy_mean": [0.7, 0.72, 0.71],
            "participant_accuracy_std": [0.1, 0.08, 0.09],
            "n_subjects": [5, 5, 5],
        }
    )
    inter = pd.DataFrame(
        {
            "comparison": ["rest1_vs_type1", "rest1_vs_type1", "rest1_vs_type1"],
            "classifier": ["LDA", "RBF-SVM", "KFDA"],
            "lopo_accuracy_mean": [0.52, 0.53, 0.51],
            "lopo_accuracy_std": [0.05, 0.04, 0.06],
            "n_held_out_subjects": [5, 5, 5],
            "pooled_epoch_accuracy": [0.52, 0.53, 0.51],
        }
    )
    made = plot_all_feature_classification(intra, inter, out)
    assert made and all(p.exists() for p in made)

    summary = pd.DataFrame(
        {
            "scope": ["intra"] * 3,
            "feature_set": ["all", "bp", "coh"],
            "comparison": ["rest1_vs_type1"] * 3,
            "classifier": ["LDA"] * 3,
            "accuracy_mean": [0.7, 0.65, 0.68],
            "accuracy_std": [0.1, 0.1, 0.1],
            "n_units": [5, 5, 5],
        }
    )
    made = plot_feature_set_comparison(summary, out)
    assert made and all(p.exists() for p in made)

    intra_freq = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "comparison": ["rest1_vs_type1", "rest1_vs_type1"],
            "feature": ["BP__FP1__alpha", "COH__FP1__F3__theta"],
            "selection_count": [2, 3],
        }
    )
    inter_freq = pd.DataFrame(
        {
            "comparison": ["rest1_vs_type1", "rest1_vs_type1"],
            "feature": ["COH__FP1__F3__theta", "COH__F3__F4__alpha"],
            "selection_count": [4, 2],
        }
    )
    made = plot_sfs_top_features(intra_freq, inter_freq, out, top_n=5)
    assert made and all(p.exists() for p in made)
    made = plot_connectivity_networks(inter_freq, out, top_n=5)
    assert made and all(p.exists() for p in made)

    behavior = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "task_type": ["Type1", "Type2", "Type1", "Type2"],
            "accuracy": [0.9, 1.0, 0.85, 1.0],
            "reaction_time_ms_median_correct": [1800, 1200, 2000, 1300],
        }
    )
    made = plot_behavioral(behavior, out)
    assert made and all(p.exists() for p in made)


def test_collect_feature_set_summary(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "intra").mkdir(parents=True)
    (root / "inter").mkdir(parents=True)
    for feature_set in ["bp", "coh"]:
        (root / "feature_sets" / feature_set / "intra").mkdir(parents=True)
        (root / "feature_sets" / feature_set / "inter").mkdir(parents=True)

    intra = pd.DataFrame(
        {
            "comparison": ["rest1_vs_type1"],
            "classifier": ["LDA"],
            "participant_accuracy_mean": [0.7],
            "participant_accuracy_std": [0.1],
            "n_subjects": [5],
        }
    )
    inter = pd.DataFrame(
        {
            "comparison": ["rest1_vs_type1"],
            "classifier": ["LDA"],
            "lopo_accuracy_mean": [0.52],
            "lopo_accuracy_std": [0.05],
            "n_held_out_subjects": [5],
            "pooled_epoch_accuracy": [0.52],
        }
    )

    intra.to_csv(root / "intra" / "intra_overall_summary.csv", index=False)
    inter.to_csv(root / "inter" / "inter_summary.csv", index=False)
    for feature_set in ["bp", "coh"]:
        intra.to_csv(
            root / "feature_sets" / feature_set / "intra" / "intra_overall_summary.csv",
            index=False,
        )
        inter.to_csv(
            root / "feature_sets" / feature_set / "inter" / "inter_summary.csv",
            index=False,
        )

    summary = collect_feature_set_summary(root)
    assert len(summary) == 6
    assert set(summary["feature_set"]) == {"all", "bp", "coh"}
    assert set(summary["scope"]) == {"intra", "inter"}
