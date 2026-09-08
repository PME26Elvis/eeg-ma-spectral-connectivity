from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .extract import feature_columns
from .models import kfda_grid_search, lda_pipeline, svm_grid_search
from .sfs import forward_select_lda


def run_intra_subject(features: pd.DataFrame, cfg: Dict, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe intra-subject evaluation.

    The project/poster specifies the order SFS -> classifier evaluation.  The
    available poster additionally reports an inter-subject *LDA* SFS result,
    but does not provide a separate SFS objective for SVM/KFDA.  The formal
    repo therefore uses training-only LDA inner-CV as the single SFS criterion
    and compares all three classifiers on exactly the same selected subset.
    This interpretation is explicit in docs/method_definition.md and config.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fcols = feature_columns(features, str(cfg.get("feature_set", "all")))
    random_state = int(cfg["cv"]["random_state"])
    outer_folds = int(cfg["cv"]["intra_outer_folds"])
    inner_folds = int(cfg["cv"]["intra_inner_folds"])

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    sfs_rows: List[Dict[str, object]] = []

    for comparison in sorted(features["comparison"].unique()):
        comp_df = features.loc[features["comparison"] == comparison].copy()
        for subject in sorted(comp_df["subject_id"].astype(str).unique()):
            df = comp_df.loc[comp_df["subject_id"].astype(str) == subject].reset_index(drop=True)
            X = df[fcols].to_numpy(dtype=float)
            y = df["class_label"].to_numpy(dtype=int)
            _assert_binary_balance(y, f"intra {subject} {comparison}")

            outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
            for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]

                inner = StratifiedKFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=random_state + outer_fold,
                )
                inner_splits = list(inner.split(Xtr, ytr))
                sfs = forward_select_lda(
                    Xtr, ytr, fcols, inner_splits,
                    max_features=cfg["sfs"].get("max_features"),
                    min_improvement=float(cfg["sfs"].get("min_improvement", 0.0)),
                )
                sel = sfs.selected_indices
                Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]

                selected_set = set(sfs.selected_names)
                for _, row in sfs.path.iterrows():
                    sfs_rows.append({
                        "analysis": "intra",
                        "subject_id": subject,
                        "comparison": comparison,
                        "outer_fold": outer_fold,
                        **row.to_dict(),
                        "accepted": str(row["candidate_name"]) in selected_set,
                    })

                models = []
                lda = lda_pipeline().fit(Xtr_s, ytr)
                models.append(("LDA", lda, {}, np.nan))

                svm = svm_grid_search(cfg, inner_splits).fit(Xtr_s, ytr)
                models.append(("RBF-SVM", svm.best_estimator_, svm.best_params_, float(svm.best_score_)))

                kfda = kfda_grid_search(cfg, inner_splits).fit(Xtr_s, ytr)
                models.append(("KFDA", kfda.best_estimator_, kfda.best_params_, float(kfda.best_score_)))

                for model_name, model, best_params, inner_score in models:
                    pred = model.predict(Xte_s)
                    acc = float(accuracy_score(yte, pred))
                    fold_rows.append({
                        "subject_id": subject,
                        "comparison": comparison,
                        "outer_fold": outer_fold,
                        "classifier": model_name,
                        "accuracy": acc,
                        "n_selected_features": len(sel),
                        "selected_features": ";".join(sfs.selected_names),
                        "sfs_inner_accuracy": sfs.best_score,
                        "grid_inner_accuracy": inner_score,
                        "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                    })
                    for local_i, p in zip(test_idx, pred):
                        pred_rows.append({
                            "subject_id": subject,
                            "comparison": comparison,
                            "outer_fold": outer_fold,
                            "classifier": model_name,
                            "row_index_within_subject_comparison": int(local_i),
                            "true_label": int(y[local_i]),
                            "predicted_label": int(p),
                        })

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    sfs_path = pd.DataFrame(sfs_rows)
    folds.to_csv(out_dir / "intra_folds.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "intra_predictions.csv", index=False, encoding="utf-8-sig")
    sfs_path.to_csv(out_dir / "intra_sfs_path.csv", index=False, encoding="utf-8-sig")
    _write_intra_summaries(folds, preds, out_dir)
    _write_feature_frequency(sfs_path, out_dir / "intra_feature_frequency.csv", mode="intra")
    return folds, preds


def run_inter_subject(features: pd.DataFrame, cfg: Dict, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe outer LOPO-CV with group-aware inner validation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fcols = feature_columns(features, str(cfg.get("feature_set", "all")))

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    sfs_rows: List[Dict[str, object]] = []

    for comparison in sorted(features["comparison"].unique()):
        df = features.loc[features["comparison"] == comparison].reset_index(drop=True)
        X = df[fcols].to_numpy(dtype=float)
        y = df["class_label"].to_numpy(dtype=int)
        groups = df["subject_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise ValueError("Inter-subject LOPO 至少需要 3 位受試者，才能保留 outer test 並在 training groups 做 inner LOPO。")

        outer = LeaveOneGroupOut()
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            gtr = groups[train_idx]
            held_out = str(np.unique(groups[test_idx])[0])

            inner_logo = LeaveOneGroupOut()
            inner_splits = list(inner_logo.split(Xtr, ytr, gtr))
            sfs = forward_select_lda(
                Xtr, ytr, fcols, inner_splits,
                max_features=cfg["sfs"].get("max_features"),
                min_improvement=float(cfg["sfs"].get("min_improvement", 0.0)),
            )
            sel = sfs.selected_indices
            Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]

            selected_set = set(sfs.selected_names)
            for _, row in sfs.path.iterrows():
                sfs_rows.append({
                    "analysis": "inter",
                    "held_out_subject": held_out,
                    "comparison": comparison,
                    "outer_fold": outer_fold,
                    **row.to_dict(),
                    "accepted": str(row["candidate_name"]) in selected_set,
                })

            models = []
            lda = lda_pipeline().fit(Xtr_s, ytr)
            models.append(("LDA", lda, {}, np.nan))

            svm = svm_grid_search(cfg, inner_splits).fit(Xtr_s, ytr)
            models.append(("RBF-SVM", svm.best_estimator_, svm.best_params_, float(svm.best_score_)))

            kfda = kfda_grid_search(cfg, inner_splits).fit(Xtr_s, ytr)
            models.append(("KFDA", kfda.best_estimator_, kfda.best_params_, float(kfda.best_score_)))

            for model_name, model, best_params, inner_score in models:
                pred = model.predict(Xte_s)
                acc = float(accuracy_score(yte, pred))
                fold_rows.append({
                    "held_out_subject": held_out,
                    "comparison": comparison,
                    "outer_fold": outer_fold,
                    "classifier": model_name,
                    "accuracy": acc,
                    "n_selected_features": len(sel),
                    "selected_features": ";".join(sfs.selected_names),
                    "sfs_inner_accuracy": sfs.best_score,
                    "grid_inner_accuracy": inner_score,
                    "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                })
                for local_i, p in zip(test_idx, pred):
                    pred_rows.append({
                        "held_out_subject": held_out,
                        "subject_id": str(groups[local_i]),
                        "comparison": comparison,
                        "outer_fold": outer_fold,
                        "classifier": model_name,
                        "global_row_index": int(local_i),
                        "true_label": int(y[local_i]),
                        "predicted_label": int(p),
                    })

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    sfs_path = pd.DataFrame(sfs_rows)
    folds.to_csv(out_dir / "inter_folds.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "inter_predictions.csv", index=False, encoding="utf-8-sig")
    sfs_path.to_csv(out_dir / "inter_sfs_path.csv", index=False, encoding="utf-8-sig")
    _write_inter_summaries(folds, preds, out_dir)
    _write_feature_frequency(sfs_path, out_dir / "inter_feature_frequency.csv", mode="inter")
    return folds, preds


def _assert_binary_balance(y: np.ndarray, label: str) -> None:
    values, counts = np.unique(y, return_counts=True)
    if len(values) != 2:
        raise ValueError(f"{label}: 不是二分類，labels={values}")
    if np.min(counts) < 5:
        raise ValueError(f"{label}: class sample 太少，counts={counts}")


def _write_intra_summaries(folds: pd.DataFrame, preds: pd.DataFrame, out_dir: Path) -> None:
    if folds.empty:
        pd.DataFrame().to_csv(out_dir / "intra_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "intra_overall_summary.csv", index=False)
        return

    # Per-participant CV summary: mean/std over that participant's five outer folds.
    summary = (
        folds.groupby(["subject_id", "comparison", "classifier"], dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_outer_folds"})
    )
    summary.to_csv(out_dir / "intra_summary.csv", index=False, encoding="utf-8-sig")

    # Group-level descriptive summary across participant-level means.  This is
    # intentionally separate from the per-participant intra-subject result.
    overall = (
        summary.groupby(["comparison", "classifier"], dropna=False)["accuracy_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "participant_accuracy_mean", "std": "participant_accuracy_std", "count": "n_subjects"})
    )
    overall.to_csv(out_dir / "intra_overall_summary.csv", index=False, encoding="utf-8-sig")


def _write_inter_summaries(folds: pd.DataFrame, preds: pd.DataFrame, out_dir: Path) -> None:
    if folds.empty:
        pd.DataFrame().to_csv(out_dir / "inter_summary.csv", index=False)
        return

    # LOPO performance is summarized *across held-out participants*, not one
    # one-row "summary" per participant.  Each fold accuracy has equal weight;
    # with this protocol each participant contributes the same 60 test epochs
    # per comparison, so fold-mean and pooled epoch accuracy coincide up to
    # floating-point rounding.
    rows: List[Dict[str, object]] = []
    for (comparison, classifier), g in folds.groupby(["comparison", "classifier"], dropna=False):
        pred_g = preds.loc[
            (preds["comparison"] == comparison) & (preds["classifier"] == classifier)
        ]
        pooled = float(accuracy_score(pred_g["true_label"], pred_g["predicted_label"])) if len(pred_g) else np.nan
        rows.append({
            "comparison": comparison,
            "classifier": classifier,
            "lopo_accuracy_mean": float(g["accuracy"].mean()),
            "lopo_accuracy_std": float(g["accuracy"].std(ddof=1)) if len(g) > 1 else np.nan,
            "n_held_out_subjects": int(len(g)),
            "pooled_epoch_accuracy": pooled,
        })
    pd.DataFrame(rows).to_csv(out_dir / "inter_summary.csv", index=False, encoding="utf-8-sig")


def _write_feature_frequency(sfs_path: pd.DataFrame, path: Path, mode: str) -> None:
    """Count accepted SFS features once per outer fold.

    Selection is common to all three downstream classifiers, so counting it
    from the classifier-specific fold table would multiply every occurrence by
    three and give a misleading frequency.  The SFS path is the source of truth.
    """
    if sfs_path.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    accepted = sfs_path.loc[sfs_path["accepted"].astype(bool)].copy()
    if accepted.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    accepted = accepted.rename(columns={"candidate_name": "feature"})
    if mode == "intra":
        keys = ["subject_id", "comparison", "feature"]
    elif mode == "inter":
        keys = ["comparison", "feature"]
    else:
        raise ValueError(mode)

    freq = accepted.groupby(keys).size().reset_index(name="selection_count")
    sort_keys = [k for k in keys if k != "feature"] + ["selection_count", "feature"]
    ascending = [True] * (len(sort_keys) - 2) + [False, True]
    freq = freq.sort_values(sort_keys, ascending=ascending)
    freq.to_csv(path, index=False, encoding="utf-8-sig")
