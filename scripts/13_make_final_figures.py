from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


CLASSIFIERS = ["KFDA", "LDA", "RBF-SVM"]
CHANNEL_POSITIONS = {
    "FP1": (-0.42, 0.78),
    "FP2": (0.42, 0.78),
    "F7": (-0.88, 0.28),
    "F3": (-0.42, 0.34),
    "Fz": (0.00, 0.38),
    "F4": (0.42, 0.34),
    "F8": (0.88, 0.28),
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required result file: {path}\n"
            "Run scripts/12_run_final_validation.py first and keep the formal result tree intact."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _grouped_accuracy(
    table: pd.DataFrame,
    experiments: list[str],
    labels: dict[str, str],
    title: str,
    out_base: Path,
) -> list[Path]:
    work = table.loc[
        (table["comparison"] == "rest1_vs_type1")
        & (table["experiment"].isin(experiments))
        & (table["classifier"].isin(CLASSIFIERS))
    ].copy()
    if work.empty:
        raise ValueError(f"No rows available for {title}")

    x = np.arange(len(experiments), dtype=float)
    width = 0.8 / len(CLASSIFIERS)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(experiments)), 5.2))

    for i, clf in enumerate(CLASSIFIERS):
        vals = []
        errs = []
        for exp in experiments:
            row = work.loc[(work["experiment"] == exp) & (work["classifier"] == clf)]
            vals.append(float(row["accuracy_mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["accuracy_std"].iloc[0]) if len(row) else np.nan)
        offset = (i - (len(CLASSIFIERS) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=clf)

    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(e, e) for e in experiments], rotation=18, ha="right")
    ax.set_ylim(0.35, 0.9)
    ax.set_ylabel("LOPO accuracy")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_base)


def plot_analysis_story(
    extension_summary: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    # Keep only the canonical rows needed for the report story.  Duplicate
    # references across tables are intentionally resolved by the first source.
    pieces = []

    ext = extension_summary.loc[
        (extension_summary["scope"] == "inter")
        & (extension_summary["comparison"] == "rest1_vs_type1")
        & (extension_summary["experiment"].isin(["poster_baseline", "e1_asymmetry", "e2_plv"]))
    ].copy()
    pieces.append(ext[["experiment", "comparison", "classifier", "accuracy_mean", "accuracy_std"]])

    dec = decomposition_summary.loc[
        (decomposition_summary["scope"] == "inter")
        & (decomposition_summary["comparison"] == "rest1_vs_type1")
        & (decomposition_summary["experiment"] == "e2a_plv_only")
    ].copy()
    pieces.append(dec[["experiment", "comparison", "classifier", "accuracy_mean", "accuracy_std"]])

    fin = final_summary.loc[
        (final_summary["comparison"] == "rest1_vs_type1")
        & (final_summary["experiment"].isin(["plv_alpha_only", "iplv_only"]))
    ].copy()
    pieces.append(fin[["experiment", "comparison", "classifier", "accuracy_mean", "accuracy_std"]])

    story = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["experiment", "comparison", "classifier"], keep="first"
    )

    experiments = [
        "poster_baseline",
        "e1_asymmetry",
        "e2_plv",
        "e2a_plv_only",
        "plv_alpha_only",
        "iplv_only",
    ]
    labels = {
        "poster_baseline": "Poster baseline\nBP+COH",
        "e1_asymmetry": "+ Asymmetry",
        "e2_plv": "+ PLV",
        "e2a_plv_only": "PLV only",
        "plv_alpha_only": "Alpha PLV only",
        "iplv_only": "iPLV only",
    }
    return _grouped_accuracy(
        story,
        experiments,
        labels,
        "Rest 1 vs Type 1 — analysis progression",
        out_dir / "final_story_type1_inter",
    )


def plot_band_ranking(final_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    mapping = {
        "plv_delta_only": "delta",
        "plv_theta_only": "theta",
        "plv_alpha_only": "alpha",
        "plv_beta_low_only": "beta-low",
        "plv_beta_high_only": "beta-high",
        "plv_gamma_only": "gamma",
    }
    work = final_summary.loc[
        (final_summary["comparison"] == "rest1_vs_type1")
        & (final_summary["experiment"].isin(mapping))
    ].copy()
    work["band"] = work["experiment"].map(mapping)
    order = ["delta", "theta", "alpha", "beta-low", "beta-high", "gamma"]

    x = np.arange(len(order), dtype=float)
    width = 0.8 / len(CLASSIFIERS)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, clf in enumerate(CLASSIFIERS):
        vals, errs = [], []
        for band in order:
            row = work.loc[(work["band"] == band) & (work["classifier"] == clf)]
            vals.append(float(row["accuracy_mean"].iloc[0]))
            errs.append(float(row["accuracy_std"].iloc[0]))
        offset = (i - (len(CLASSIFIERS) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=clf)

    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0.35, 0.9)
    ax.set_ylabel("LOPO accuracy")
    ax.set_title("Rest 1 vs Type 1 — single-band PLV comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_dir / "final_plv_band_comparison_type1")


def plot_alpha_subjects(delta_by_unit: pd.DataFrame, out_dir: Path) -> list[Path]:
    work = delta_by_unit.loc[
        (delta_by_unit["experiment"] == "plv_alpha_only")
        & (delta_by_unit["comparison"] == "rest1_vs_type1")
        & (delta_by_unit["classifier"].isin(CLASSIFIERS))
    ].copy()
    if work.empty:
        raise ValueError("No alpha-only paired unit rows found")

    # Descriptive subject-level summary.  Mean across classifiers is used only
    # to keep this figure readable; classifier-specific values remain in CSV.
    subject = (
        work.groupby("unit_id", as_index=False)[["accuracy_reference", "accuracy_experiment"]]
        .mean()
        .sort_values("unit_id")
    )
    x = np.arange(len(subject), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.bar(x - width / 2, subject["accuracy_reference"], width=width, label="Poster baseline")
    ax.bar(x + width / 2, subject["accuracy_experiment"], width=width, label="Alpha PLV only")
    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(subject["unit_id"].astype(str))
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Accuracy (mean across 3 classifiers)")
    ax.set_title("Rest 1 vs Type 1 — held-out subject comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_dir / "final_alpha_vs_baseline_by_subject")


def _parse_plv_feature(name: str) -> tuple[str, str, str] | None:
    parts = str(name).split("__")
    if len(parts) != 4 or parts[0] != "PLV":
        return None
    return parts[1], parts[2], parts[3]


def plot_alpha_network(freq: pd.DataFrame, out_dir: Path, top_n: int = 12) -> list[Path]:
    work = freq.loc[
        (freq["comparison"] == "rest1_vs_type1")
        & freq["feature"].astype(str).str.startswith("PLV__")
    ].copy()
    if work.empty:
        raise ValueError("No alpha-only PLV SFS frequency rows found")

    work = work.sort_values(["selection_count", "feature"], ascending=[False, True]).head(top_n)
    parsed = []
    for _, row in work.iterrows():
        p = _parse_plv_feature(row["feature"])
        if p and p[0] in CHANNEL_POSITIONS and p[1] in CHANNEL_POSITIONS:
            parsed.append((*p, float(row["selection_count"])))
    if not parsed:
        raise ValueError("No drawable PLV connections found")

    max_count = max(x[3] for x in parsed)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.add_patch(Circle((0.0, 0.15), 1.05, fill=False, linewidth=1.5))
    ax.plot([-0.12, 0.0, 0.12], [1.15, 1.28, 1.15], linewidth=1.2)

    for ch, (x, y) in CHANNEL_POSITIONS.items():
        ax.scatter([x], [y], s=90, zorder=3)
        ax.text(x, y + 0.08, ch, ha="center", va="bottom", fontsize=10)

    for ch1, ch2, band, count in parsed:
        x1, y1 = CHANNEL_POSITIONS[ch1]
        x2, y2 = CHANNEL_POSITIONS[ch2]
        lw = 1.0 + 3.0 * (count / max_count)
        ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=0.55, zorder=1)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(mx, my, f"{band}×{int(count)}", fontsize=7, ha="center", va="center")

    ax.set_title("Alpha-PLV SFS network — Rest 1 vs Type 1")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.95, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    return _save(fig, out_dir / "final_alpha_plv_sfs_network")


def plot_phase_robustness(final_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    work = final_summary.loc[
        (final_summary["comparison"] == "rest1_vs_type1")
        & (final_summary["experiment"].isin([
            "poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"
        ]))
    ].copy()
    experiments = ["poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"]
    labels = {
        "poster_baseline": "Poster baseline",
        "plv_all_bands": "PLV all bands",
        "plv_alpha_only": "Alpha PLV",
        "iplv_only": "iPLV",
    }
    return _grouped_accuracy(
        work,
        experiments,
        labels,
        "Rest 1 vs Type 1 — phase-connectivity validation",
        out_dir / "final_phase_connectivity_validation",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Create the final report figures from frozen formal results")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/final_figures")
    p.add_argument("--network-top-n", type=int, default=12)
    args = p.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)

    extension_summary = _read(root / "extensions" / "extension_summary.csv")
    decomposition_summary = _read(root / "plv_decomposition" / "plv_decomposition_summary.csv")
    final_summary = _read(root / "final_validation" / "final_validation_summary.csv")
    final_delta = _read(root / "final_validation" / "delta_vs_poster_by_unit.csv")
    alpha_freq = _read(
        root / "final_validation" / "plv_alpha_only" / "inter" / "inter_feature_frequency.csv"
    )

    made: list[Path] = []
    made += plot_analysis_story(extension_summary, decomposition_summary, final_summary, out)
    made += plot_band_ranking(final_summary, out)
    made += plot_alpha_subjects(final_delta, out)
    made += plot_phase_robustness(final_summary, out)
    made += plot_alpha_network(alpha_freq, out, top_n=args.network_top_n)

    print("=== FINAL FIGURES ===")
    for path in made:
        print(path)
    print(f"count={len(made)} files ({len(made)//2} figures, PNG+SVG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
