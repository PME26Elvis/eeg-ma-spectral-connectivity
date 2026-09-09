from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


COMPARISON_LABELS = {
    "rest1_vs_type1": "Rest 1 vs Type 1",
    "rest2_vs_type2": "Rest 2 vs Type 2",
}
FEATURE_SET_LABELS = {"all": "BP+COH", "bp": "BP only", "coh": "COH only"}
CHANNEL_POSITIONS = {
    "FP1": (-0.42, 0.78),
    "FP2": (0.42, 0.78),
    "F7": (-0.88, 0.28),
    "F3": (-0.42, 0.34),
    "Fz": (0.00, 0.38),
    "F4": (0.42, 0.34),
    "F8": (0.88, 0.28),
}


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _grouped_bars(
    table: pd.DataFrame,
    category_col: str,
    series_col: str,
    value_col: str,
    error_col: str | None,
    title: str,
    ylabel: str,
    out_base: Path,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    categories = list(dict.fromkeys(table[category_col].astype(str)))
    series = list(dict.fromkeys(table[series_col].astype(str)))
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(categories)), 4.8))
    for i, name in enumerate(series):
        vals = []
        errs = []
        for cat in categories:
            row = table.loc[
                (table[category_col].astype(str) == cat)
                & (table[series_col].astype(str) == name)
            ]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
            if error_col is not None:
                errs.append(float(row[error_col].iloc[0]) if len(row) else np.nan)
        offset = (i - (len(series) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs if error_col is not None else None,
            capsize=3 if error_col is not None else 0,
            label=name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out_base)


def plot_all_feature_classification(
    intra_overall: pd.DataFrame,
    inter_summary: pd.DataFrame,
    out_dir: str | Path,
) -> list[Path]:
    out = Path(out_dir)
    made: list[Path] = []

    intra = intra_overall.copy()
    intra["comparison_label"] = intra["comparison"].map(COMPARISON_LABELS).fillna(intra["comparison"])
    made += _grouped_bars(
        intra,
        category_col="comparison_label",
        series_col="classifier",
        value_col="participant_accuracy_mean",
        error_col="participant_accuracy_std",
        title="Intra-subject classification (BP+COH)",
        ylabel="Accuracy",
        out_base=out / "classification_all_intra",
        ylim=(0.0, 1.0),
    )

    inter = inter_summary.copy()
    inter["comparison_label"] = inter["comparison"].map(COMPARISON_LABELS).fillna(inter["comparison"])
    made += _grouped_bars(
        inter,
        category_col="comparison_label",
        series_col="classifier",
        value_col="lopo_accuracy_mean",
        error_col="lopo_accuracy_std",
        title="Inter-subject LOPO classification (BP+COH)",
        ylabel="Accuracy",
        out_base=out / "classification_all_inter",
        ylim=(0.0, 1.0),
    )
    return made


def plot_feature_set_comparison(summary: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    made: list[Path] = []
    if summary.empty:
        return made

    work = summary.copy()
    work["feature_set_label"] = work["feature_set"].map(FEATURE_SET_LABELS).fillna(work["feature_set"])
    for scope in ["intra", "inter"]:
        scoped = work.loc[work["scope"] == scope]
        for comparison in sorted(scoped["comparison"].unique()):
            sub = scoped.loc[scoped["comparison"] == comparison].copy()
            # One x category per classifier, grouped by feature set.
            label = COMPARISON_LABELS.get(str(comparison), str(comparison))
            made += _grouped_bars(
                sub,
                category_col="classifier",
                series_col="feature_set_label",
                value_col="accuracy_mean",
                error_col="accuracy_std",
                title=f"{scope.capitalize()} feature-set comparison — {label}",
                ylabel="Accuracy",
                out_base=out / f"feature_sets_{scope}_{comparison}",
                ylim=(0.0, 1.0),
            )
    return made


def _top_feature_table(freq: pd.DataFrame, mode: str, comparison: str, top_n: int) -> pd.DataFrame:
    sub = freq.loc[freq["comparison"] == comparison].copy()
    if mode == "intra" and "subject_id" in sub.columns:
        sub = sub.groupby("feature", as_index=False)["selection_count"].sum()
    else:
        sub = sub.groupby("feature", as_index=False)["selection_count"].sum()
    return sub.sort_values(["selection_count", "feature"], ascending=[False, True]).head(top_n)


def plot_sfs_top_features(
    intra_freq: pd.DataFrame,
    inter_freq: pd.DataFrame,
    out_dir: str | Path,
    top_n: int = 15,
) -> list[Path]:
    out = Path(out_dir)
    made: list[Path] = []
    for mode, freq in [("intra", intra_freq), ("inter", inter_freq)]:
        if freq.empty:
            continue
        for comparison in sorted(freq["comparison"].unique()):
            top = _top_feature_table(freq, mode, comparison, top_n)
            if top.empty:
                continue
            top = top.sort_values("selection_count", ascending=True)
            fig, ax = plt.subplots(figsize=(8.6, max(4.5, 0.33 * len(top) + 1.5)))
            ax.barh(top["feature"], top["selection_count"])
            ax.set_xlabel("SFS selection count")
            ax.set_title(
                f"Top SFS features — {mode.capitalize()} — "
                f"{COMPARISON_LABELS.get(str(comparison), comparison)}"
            )
            ax.grid(axis="x", alpha=0.2)
            made += _save(fig, out / f"sfs_top_{mode}_{comparison}")
    return made


def _parse_coh_feature(name: str) -> tuple[str, str, str] | None:
    parts = str(name).split("__")
    if len(parts) != 4 or parts[0] != "COH":
        return None
    return parts[1], parts[2], parts[3]


def plot_connectivity_networks(
    inter_freq: pd.DataFrame,
    out_dir: str | Path,
    top_n: int = 10,
) -> list[Path]:
    """Draw a schematic frontal connectivity network from inter-subject SFS frequency.

    This is deliberately a connectivity network, not an interpolated EEG topomap:
    node positions are fixed schematic 10-20 locations and edge width encodes how
    often a COH feature was selected across outer LOPO folds.
    """
    out = Path(out_dir)
    made: list[Path] = []
    coh = inter_freq.loc[inter_freq["feature"].astype(str).str.startswith("COH__")].copy()
    if coh.empty:
        return made

    for comparison in sorted(coh["comparison"].unique()):
        sub = coh.loc[coh["comparison"] == comparison].copy()
        sub = (
            sub.groupby("feature", as_index=False)["selection_count"]
            .sum()
            .sort_values(["selection_count", "feature"], ascending=[False, True])
            .head(top_n)
        )
        parsed = []
        for _, row in sub.iterrows():
            p = _parse_coh_feature(row["feature"])
            if p is not None and p[0] in CHANNEL_POSITIONS and p[1] in CHANNEL_POSITIONS:
                parsed.append((*p, float(row["selection_count"])))
        if not parsed:
            continue

        max_count = max(p[3] for p in parsed)
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.add_patch(Circle((0.0, 0.15), 1.05, fill=False, linewidth=1.5))
        # Small nose marker for orientation.
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

        ax.set_title(
            "Inter-subject SFS COH network — "
            f"{COMPARISON_LABELS.get(str(comparison), comparison)}"
        )
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.95, 1.38)
        ax.set_aspect("equal")
        ax.axis("off")
        made += _save(fig, out / f"connectivity_inter_{comparison}")
    return made


def plot_behavioral(behavior: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    made: list[Path] = []
    if behavior.empty:
        return made

    b = behavior.copy()
    subjects = sorted(b["subject_id"].astype(str).unique())
    types = [t for t in ["Type1", "Type2"] if t in set(b["task_type"].astype(str))]
    x = np.arange(len(subjects), dtype=float)
    width = 0.8 / max(len(types), 1)

    for value_col, ylabel, title, stem, ylim in [
        ("accuracy", "Accuracy", "Behavioral accuracy", "behavioral_accuracy", (0.0, 1.05)),
        (
            "reaction_time_ms_median_correct",
            "Median correct RT (ms)",
            "Behavioral reaction time",
            "behavioral_rt_median",
            None,
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for i, task in enumerate(types):
            vals = []
            for subject in subjects:
                row = b.loc[
                    (b["subject_id"].astype(str) == subject)
                    & (b["task_type"].astype(str) == task)
                ]
                vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
            offset = (i - (len(types) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=task)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        made += _save(fig, out / stem)
    return made


def collect_feature_set_summary(
    results_root: str | Path,
    feature_sets: Iterable[str] = ("all", "bp", "coh"),
) -> pd.DataFrame:
    """Collect canonical all-feature results plus BP-only/COH-only summaries."""
    root = Path(results_root)
    rows: list[dict[str, object]] = []

    for feature_set in feature_sets:
        if feature_set == "all":
            intra_path = root / "intra" / "intra_overall_summary.csv"
            inter_path = root / "inter" / "inter_summary.csv"
        else:
            intra_path = root / "feature_sets" / feature_set / "intra" / "intra_overall_summary.csv"
            inter_path = root / "feature_sets" / feature_set / "inter" / "inter_summary.csv"

        if intra_path.exists():
            intra = pd.read_csv(intra_path, encoding="utf-8-sig")
            for _, r in intra.iterrows():
                rows.append(
                    {
                        "scope": "intra",
                        "feature_set": feature_set,
                        "comparison": r["comparison"],
                        "classifier": r["classifier"],
                        "accuracy_mean": float(r["participant_accuracy_mean"]),
                        "accuracy_std": float(r["participant_accuracy_std"]),
                        "n_units": int(r["n_subjects"]),
                    }
                )

        if inter_path.exists():
            inter = pd.read_csv(inter_path, encoding="utf-8-sig")
            for _, r in inter.iterrows():
                rows.append(
                    {
                        "scope": "inter",
                        "feature_set": feature_set,
                        "comparison": r["comparison"],
                        "classifier": r["classifier"],
                        "accuracy_mean": float(r["lopo_accuracy_mean"]),
                        "accuracy_std": float(r["lopo_accuracy_std"]),
                        "n_units": int(r["n_held_out_subjects"]),
                    }
                )

    return pd.DataFrame(rows)
