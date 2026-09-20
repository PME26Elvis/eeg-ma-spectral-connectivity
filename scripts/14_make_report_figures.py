from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


CLASSIFIERS = ["KFDA", "LDA", "RBF-SVM"]
COMPARISON = "rest1_vs_type1"
CHANNEL_POSITIONS = {
    "FP1": (-0.42, 0.78),
    "FP2": (0.42, 0.78),
    "F7": (-0.88, 0.28),
    "F3": (-0.42, 0.34),
    "Fz": (0.00, 0.38),
    "F4": (0.42, 0.34),
    "F8": (0.88, 0.28),
}


@dataclass(frozen=True)
class Variant:
    name: str
    font_size: float
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    scale: float
    dpi: int


VARIANTS = {
    "report": Variant("report", 10.5, 13.0, 11.0, 9.5, 9.5, 1.00, 300),
    "slide": Variant("slide", 13.0, 17.0, 14.0, 12.0, 12.0, 1.20, 240),
    "compact": Variant("compact", 9.0, 11.5, 9.5, 8.0, 8.0, 0.88, 260),
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required result file: {path}\n"
            "Run the formal analysis scripts first and keep the canonical result tree intact."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def _require(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _configure(v: Variant) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": v.font_size,
        "axes.titlesize": v.title_size,
        "axes.labelsize": v.label_size,
        "xtick.labelsize": v.tick_size,
        "ytick.labelsize": v.tick_size,
        "legend.fontsize": v.legend_size,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str, v: Variant) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=v.dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _chance(ax: plt.Axes) -> None:
    ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.65, color="0.35")


def _grouped_bars(
    ax: plt.Axes,
    categories: list[str],
    values: dict[str, list[float]],
    errors: dict[str, list[float]] | None = None,
    ylim: tuple[float, float] = (0.35, 0.90),
    ylabel: str = "Accuracy",
    legend: bool = True,
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.78 / len(CLASSIFIERS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, clf in enumerate(CLASSIFIERS):
        offset = (i - (len(CLASSIFIERS) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[clf],
            width=width,
            yerr=None if errors is None else errors[clf],
            capsize=3 if errors is not None else 0,
            label=clf,
            color=colors[i % len(colors)],
            alpha=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    if legend:
        ax.legend(frameon=False, ncol=3)


def _pull_rows(
    table: pd.DataFrame,
    categories: list[str],
    category_col: str,
    value_col: str,
    error_col: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    values = {c: [] for c in CLASSIFIERS}
    errors = {c: [] for c in CLASSIFIERS}
    for clf in CLASSIFIERS:
        for cat in categories:
            row = table.loc[(table[category_col] == cat) & (table["classifier"] == clf)]
            if len(row) != 1:
                raise ValueError(f"Expected exactly one row for {cat=} {clf=}; got {len(row)}")
            values[clf].append(float(row[value_col].iloc[0]))
            errors[clf].append(float(row[error_col].iloc[0]))
    return values, errors


def fig01_baseline_intra_inter(root: Path, out: Path, v: Variant) -> list[Path]:
    intra = _read(root / "intra" / "intra_overall_summary.csv")
    inter = _read(root / "inter" / "inter_summary.csv")
    _require(intra, ["comparison", "classifier", "participant_accuracy_mean", "participant_accuracy_std"], "intra summary")
    _require(inter, ["comparison", "classifier", "lopo_accuracy_mean", "lopo_accuracy_std"], "inter summary")

    rows = []
    for comp, short in [("rest1_vs_type1", "Type1"), ("rest2_vs_type2", "Type2")]:
        for _, r in intra.loc[intra["comparison"] == comp].iterrows():
            rows.append({
                "category": f"{short}\nIntra",
                "classifier": r["classifier"],
                "mean": r["participant_accuracy_mean"],
                "std": r["participant_accuracy_std"],
            })
        for _, r in inter.loc[inter["comparison"] == comp].iterrows():
            rows.append({
                "category": f"{short}\nLOPO",
                "classifier": r["classifier"],
                "mean": r["lopo_accuracy_mean"],
                "std": r["lopo_accuracy_std"],
            })

    data = pd.DataFrame(rows)
    cats = ["Type1\nIntra", "Type1\nLOPO", "Type2\nIntra", "Type2\nLOPO"]
    vals, errs = _pull_rows(data, cats, "category", "mean", "std")

    fig, ax = plt.subplots(figsize=(8.6 * v.scale, 4.9 * v.scale), constrained_layout=True)
    _grouped_bars(ax, cats, vals, errs, ylim=(0.35, 1.0), ylabel="Accuracy")
    _chance(ax)
    ax.set_title("Baseline classification: intra-subject vs inter-subject")
    ax.text(
        0.99, 0.02,
        "BP + COH → training-only SFS → classifier",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="0.35",
    )
    return _save(fig, out, "fig01_baseline_intra_vs_inter", v)


def fig02_baseline_decomposition_asym(root: Path, out: Path, v: Variant) -> list[Path]:
    fs = _read(root / "feature_set_comparison.csv")
    ext = _read(root / "extensions" / "extension_summary.csv")

    base = fs.loc[
        (fs["scope"] == "inter")
        & (fs["comparison"] == COMPARISON)
        & (fs["feature_set"].isin(["bp", "coh", "all"]))
    ].copy()
    label_map = {"bp": "BP", "coh": "COH", "all": "BP+COH"}
    base["category"] = base["feature_set"].map(label_map)
    base = base.rename(columns={"accuracy_mean": "mean", "accuracy_std": "std"})

    asym = ext.loc[
        (ext["scope"] == "inter")
        & (ext["comparison"] == COMPARISON)
        & (ext["experiment"] == "e1_asymmetry")
    ].copy()
    asym["category"] = "BP+COH\n+ASYM"
    asym = asym.rename(columns={"accuracy_mean": "mean", "accuracy_std": "std"})

    data = pd.concat([
        base[["category", "classifier", "mean", "std"]],
        asym[["category", "classifier", "mean", "std"]],
    ], ignore_index=True)

    cats = ["BP", "COH", "BP+COH", "BP+COH\n+ASYM"]
    vals, errs = _pull_rows(data, cats, "category", "mean", "std")

    fig, ax = plt.subplots(figsize=(8.5 * v.scale, 4.9 * v.scale), constrained_layout=True)
    _grouped_bars(ax, cats, vals, errs, ylim=(0.44, 0.62), ylabel="LOPO accuracy")
    _chance(ax)
    ax.axvline(2.5, color="0.65", linewidth=1.0)
    ax.text(
        1.0, 0.995, "Baseline decomposition",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontweight="bold",
    )
    ax.text(
        3.0, 0.995, "Extension",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontweight="bold",
    )
    ax.set_title("Baseline decomposition and asymmetry extension — Rest1 vs Type1")
    ax.text(
        0.99, 0.02,
        "Each candidate pool uses the same training-only SFS",
        transform=ax.transAxes,
        ha="right", va="bottom", color="0.35",
    )
    return _save(fig, out, "fig02_baseline_decomposition_asymmetry", v)


def _anonymize_subjects(ids: list[str]) -> dict[str, str]:
    return {sid: f"S{i+1}" for i, sid in enumerate(sorted(ids))}


def fig03_plv_gain_subjects(root: Path, out: Path, v: Variant) -> list[Path]:
    summary = _read(root / "extensions" / "extension_summary.csv")
    delta = _read(root / "extensions" / "extension_delta_by_unit.csv")

    work = summary.loc[
        (summary["scope"] == "inter")
        & (summary["comparison"] == COMPARISON)
        & (summary["experiment"].isin(["poster_baseline", "e2_plv"]))
    ].copy()

    categories = ["Baseline\nBP+COH", "BP+COH\n+PLV"]
    exp_map = {"poster_baseline": categories[0], "e2_plv": categories[1]}
    work["category"] = work["experiment"].map(exp_map)
    vals, errs = _pull_rows(work, categories, "category", "accuracy_mean", "accuracy_std")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.0 * v.scale, 4.8 * v.scale),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    ax = axes[0]
    _grouped_bars(ax, categories, vals, errs, ylim=(0.40, 0.82), ylabel="LOPO accuracy")
    _chance(ax)
    ax.set_title("Mean performance")

    d = delta.loc[
        (delta["scope"] == "inter")
        & (delta["comparison"] == COMPARISON)
        & (delta["experiment"] == "e2_plv")
    ].copy()
    subject = d.groupby("unit_id", as_index=False)[
        ["accuracy_mean_baseline", "accuracy_mean_extension"]
    ].mean()
    subject = subject.sort_values("unit_id")
    amap = _anonymize_subjects(subject["unit_id"].astype(str).tolist())

    ax = axes[1]
    x = np.array([0.0, 1.0])
    for _, r in subject.iterrows():
        sid = str(r["unit_id"])
        y = [float(r["accuracy_mean_baseline"]), float(r["accuracy_mean_extension"])]
        ax.plot(x, y, marker="o", linewidth=1.6, alpha=0.8, label=amap[sid])
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "+PLV"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0.42, 0.86)
    ax.set_ylabel("Accuracy\n(mean across 3 classifiers)")
    _chance(ax)
    ax.grid(axis="y", alpha=0.18)
    ax.set_title("Held-out subject consistency")
    ax.legend(frameon=False, ncol=2)

    improved = int(
        (subject["accuracy_mean_extension"] > subject["accuracy_mean_baseline"]).sum()
    )
    ax.text(
        0.98, 0.03,
        f"{improved}/{len(subject)} held-out subjects improved",
        transform=ax.transAxes,
        ha="right", va="bottom", color="0.35",
    )

    fig.suptitle(
        "Initial PLV extension — Rest1 vs Type1",
        fontsize=v.title_size + 0.5,
        fontweight="bold",
    )
    return _save(fig, out, "fig03_plv_initial_gain_subject_consistency", v)


def fig04_plv_decomposition(root: Path, out: Path, v: Variant) -> list[Path]:
    dec = _read(root / "plv_decomposition" / "plv_decomposition_summary.csv")
    work = dec.loc[
        (dec["scope"] == "inter")
        & (dec["comparison"] == COMPARISON)
    ].copy()

    exp_order = [
        "poster_baseline",
        "e2_full_baseline_plv",
        "e2a_plv_only",
        "e2b_bp_plv",
        "e2c_coh_plv",
        "e2d_asym_plv",
        "e3_all_extensions",
    ]
    labels = {
        "poster_baseline": "Baseline\n168 cand.",
        "e2_full_baseline_plv": "BP+COH+PLV\n294 cand.",
        "e2a_plv_only": "PLV only\n126 cand.",
        "e2b_bp_plv": "BP+PLV\n168 cand.",
        "e2c_coh_plv": "COH+PLV\n252 cand.",
        "e2d_asym_plv": "ASYM+PLV\n144 cand.",
        "e3_all_extensions": "All families\n312 cand.",
    }
    work["category"] = work["experiment"].map(labels)
    cats = [labels[x] for x in exp_order]
    vals, errs = _pull_rows(work, cats, "category", "accuracy_mean", "accuracy_std")

    fig, ax = plt.subplots(figsize=(12.0 * v.scale, 5.2 * v.scale), constrained_layout=True)
    _grouped_bars(ax, cats, vals, errs, ylim=(0.45, 0.80), ylabel="LOPO accuracy")
    _chance(ax)
    ax.set_title("PLV decomposition — Rest1 vs Type1")
    ax.text(
        0.99, 0.02,
        "Candidate features → same training-only SFS → classifier",
        transform=ax.transAxes,
        ha="right", va="bottom", color="0.35",
    )
    return _save(fig, out, "fig04_plv_decomposition_all_feature_families", v)


def _parse_plv_feature(name: str) -> tuple[str, str, str] | None:
    parts = str(name).split("__")
    if len(parts) != 4 or parts[0] != "PLV":
        return None
    return parts[1], parts[2], parts[3]


def _draw_alpha_network(ax: plt.Axes, freq: pd.DataFrame) -> None:
    alpha = freq.loc[
        (freq["comparison"] == COMPARISON)
        & freq["feature"].astype(str).str.startswith("PLV__")
        & freq["feature"].astype(str).str.endswith("__alpha")
    ].copy()
    alpha = alpha.sort_values(
        ["selection_count", "feature"],
        ascending=[False, True],
    )
    alpha = alpha.loc[alpha["selection_count"] >= 2].head(8)

    parsed: list[tuple[str, str, str, float]] = []
    for _, row in alpha.iterrows():
        p = _parse_plv_feature(row["feature"])
        if p and p[0] in CHANNEL_POSITIONS and p[1] in CHANNEL_POSITIONS:
            parsed.append((*p, float(row["selection_count"])))

    ax.add_patch(Circle((0.0, 0.15), 1.05, fill=False, linewidth=1.3, color="0.3"))
    ax.plot([-0.12, 0.0, 0.12], [1.15, 1.28, 1.15], linewidth=1.1, color="0.3")

    for ch, (x, y) in CHANNEL_POSITIONS.items():
        ax.scatter([x], [y], s=55, zorder=3, color="0.2")
        ax.text(x, y + 0.08, ch, ha="center", va="bottom")

    max_count = max([p[3] for p in parsed], default=1.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ch1, ch2, band, count in parsed:
        x1, y1 = CHANNEL_POSITIONS[ch1]
        x2, y2 = CHANNEL_POSITIONS[ch2]
        lw = 1.0 + 3.0 * (count / max_count)
        ax.plot(
            [x1, x2], [y1, y2],
            linewidth=lw, alpha=0.55, zorder=1, color=colors[0],
        )
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(
            mx, my, f"×{int(count)}",
            ha="center", va="center",
            fontsize=max(7, plt.rcParams["font.size"] - 2),
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.95, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Alpha PLV recurrence\nin initial +PLV SFS")
    ax.text(
        0.5, -0.02,
        "Edge labels = selection count across 5 LOPO folds",
        transform=ax.transAxes,
        ha="center", va="top", color="0.35",
    )


def fig05_band_validation(root: Path, out: Path, v: Variant) -> list[Path]:
    final = _read(root / "final_validation" / "final_validation_summary.csv")
    freq = _read(
        root / "extensions" / "e2_plv" / "inter" / "inter_feature_frequency.csv"
    )
    mapping = {
        "plv_delta_only": "Delta",
        "plv_theta_only": "Theta",
        "plv_alpha_only": "Alpha",
        "plv_beta_low_only": "Beta-low",
        "plv_beta_high_only": "Beta-high",
        "plv_gamma_only": "Gamma",
    }
    work = final.loc[
        (final["comparison"] == COMPARISON)
        & (final["experiment"].isin(mapping))
    ].copy()
    work["category"] = work["experiment"].map(mapping)
    cats = ["Delta", "Theta", "Alpha", "Beta-low", "Beta-high", "Gamma"]
    vals, errs = _pull_rows(work, cats, "category", "accuracy_mean", "accuracy_std")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.0 * v.scale, 5.2 * v.scale),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 0.8]},
    )

    ax = axes[0]
    _grouped_bars(ax, cats, vals, errs, ylim=(0.35, 0.90), ylabel="LOPO accuracy")
    _chance(ax)
    ax.set_title("Six-band PLV validation")
    ax.text(2, 0.875, "Alpha", ha="center", va="top", fontweight="bold")

    _draw_alpha_network(axes[1], freq)
    fig.suptitle(
        "Band specificity of PLV — Rest1 vs Type1",
        fontsize=v.title_size + 0.5,
        fontweight="bold",
    )
    return _save(fig, out, "fig05_plv_band_validation_alpha_recurrence", v)


def fig06_zero_lag(root: Path, out: Path, v: Variant) -> list[Path]:
    final = _read(root / "final_validation" / "final_validation_summary.csv")
    exp_order = [
        "poster_baseline",
        "plv_all_bands",
        "plv_alpha_only",
        "iplv_only",
    ]
    labels = {
        "poster_baseline": "Baseline",
        "plv_all_bands": "PLV\nall bands",
        "plv_alpha_only": "Alpha PLV",
        "iplv_only": "iPLV",
    }
    work = final.loc[
        (final["comparison"] == COMPARISON)
        & (final["experiment"].isin(exp_order))
    ].copy()
    work["category"] = work["experiment"].map(labels)
    cats = [labels[x] for x in exp_order]
    vals, errs = _pull_rows(work, cats, "category", "accuracy_mean", "accuracy_std")

    fig, ax = plt.subplots(figsize=(8.6 * v.scale, 5.0 * v.scale), constrained_layout=True)
    _grouped_bars(ax, cats, vals, errs, ylim=(0.35, 0.86), ylabel="LOPO accuracy")
    _chance(ax)
    ax.set_title("Zero-lag sensitivity of the PLV finding — Rest1 vs Type1")
    ax.text(
        0.99, 0.02,
        "iPLV suppresses exact zero-/π-lag phase locking",
        transform=ax.transAxes,
        ha="right", va="bottom", color="0.35",
    )
    return _save(fig, out, "fig06_zero_lag_sensitivity", v)


def fig07_behavior(root: Path, out: Path, v: Variant) -> list[Path]:
    b = _read(root / "behavioral_summary.csv")
    _require(
        b,
        [
            "subject_id",
            "task_type",
            "accuracy",
            "reaction_time_ms_median_correct",
            "attempts_to_30_correct",
        ],
        "behavioral summary",
    )

    subjects = sorted(b["subject_id"].astype(str).unique())
    amap = _anonymize_subjects(subjects)
    x = np.arange(len(subjects), dtype=float)
    width = 0.36
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.3 * v.scale, 4.7 * v.scale),
        constrained_layout=True,
    )

    for panel, (col, ylabel, title) in enumerate([
        ("reaction_time_ms_median_correct", "Median correct RT (ms)", "Response time"),
        ("attempts_to_30_correct", "Attempts to reach 30 correct", "Attempts / ceiling context"),
    ]):
        ax = axes[panel]
        for i, task in enumerate(["Type1", "Type2"]):
            vals = []
            for sid in subjects:
                row = b.loc[
                    (b["subject_id"].astype(str) == sid)
                    & (b["task_type"] == task)
                ]
                if len(row) != 1:
                    raise ValueError(f"Expected one behavioral row for {sid=} {task=}")
                vals.append(float(row[col].iloc[0]))

            offset = (-0.5 if i == 0 else 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=task,
                color=colors[i],
                alpha=0.95,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([amap[s] for s in subjects])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False)

    type2 = b.loc[b["task_type"] == "Type2"]
    if len(type2) and np.allclose(
        type2["accuracy"].to_numpy(dtype=float),
        1.0,
    ):
        axes[1].text(
            0.98, 0.95,
            "Type2 accuracy = 100% for all 5 participants",
            transform=axes[1].transAxes,
            ha="right", va="top", color="0.35",
        )

    fig.suptitle(
        "Behavioral context: Type1 vs Type2",
        fontsize=v.title_size + 0.5,
        fontweight="bold",
    )
    return _save(fig, out, "fig07_behavioral_context", v)


def _write_manifest(out: Path, variant: str) -> None:
    rows = [
        (
            "fig01_baseline_intra_vs_inter",
            "Establishes the intra/inter generalization gap that motivates the rest of the analysis.",
        ),
        (
            "fig02_baseline_decomposition_asymmetry",
            "Shows baseline BP/COH decomposition and the asymmetry extension under the same SFS pipeline.",
        ),
        (
            "fig03_plv_initial_gain_subject_consistency",
            "Shows the initial PLV gain and whether it is broad across held-out subjects rather than driven by one participant.",
        ),
        (
            "fig04_plv_decomposition_all_feature_families",
            "Tests where the PLV gain comes from and explicitly includes the all-feature-family candidate pool.",
        ),
        (
            "fig05_plv_band_validation_alpha_recurrence",
            "Connects the initial alpha SFS recurrence to a full six-band validation.",
        ),
        (
            "fig06_zero_lag_sensitivity",
            "Tests whether the PLV gain survives a zero-lag-suppressing representation (iPLV).",
        ),
        (
            "fig07_behavioral_context",
            "Provides behavioral context for the stronger Type1 finding and the Type2 ceiling effect.",
        ),
    ]
    pd.DataFrame(
        rows,
        columns=["figure", "report_role"],
    ).assign(variant=variant).to_csv(
        out / "figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def make_variant(root: Path, out_root: Path, v: Variant) -> list[Path]:
    _configure(v)
    out = out_root / v.name

    made: list[Path] = []
    made += fig01_baseline_intra_inter(root, out, v)
    made += fig02_baseline_decomposition_asym(root, out, v)
    made += fig03_plv_gain_subjects(root, out, v)
    made += fig04_plv_decomposition(root, out, v)
    made += fig05_band_validation(root, out, v)
    made += fig06_zero_lag(root, out, v)
    made += fig07_behavior(root, out, v)
    _write_manifest(out, v.name)
    return made


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate the seven canonical report figures from frozen EEG results."
    )
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/report_figures")
    p.add_argument(
        "--variant",
        choices=["report", "slide", "compact", "all"],
        default="report",
        help="report is the canonical submission layout; all also renders slide/compact alternatives.",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    out_root = Path(args.out_dir)
    names = list(VARIANTS) if args.variant == "all" else [args.variant]

    for name in names:
        made = make_variant(root, out_root, VARIANTS[name])
        print(f"\n=== REPORT FIGURES: {name} ===")
        for path in made:
            print(path)
        print(f"{len(made)//2} figures ({len(made)} PNG/SVG files)")

    print(f"\nDone. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
