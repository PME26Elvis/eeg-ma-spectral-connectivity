from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .extract import feature_columns
from .classifier_config import classifier_policy, resolve_classifiers
from .models import fit_lightgbm, fit_random_forest, kfda_grid_search, lda_pipeline, svm_grid_search
from .sfs import forward_select_lda


STABLE_SPLIT_POLICY = "stable_epoch_id_v1"
LEGACY_SPLIT_POLICY = "legacy_positional_v1"


def evaluation_contract(scope: str, *, stable_epoch_ids: bool = True, cfg: dict | None = None) -> dict:
    """Describe the estimand and the roles of inner/outer scores explicitly."""
    if scope not in {"intra", "inter"}:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    result = {
        "schema_version": "eeg-evaluation-contract-1",
        "analysis_id": "grouped_lopo_learning_procedure" if scope == "inter" else "within_subject_nested_cv",
        "report_title": ("Grouped LOPO learning-procedure evaluation" if scope == "inter"
                         else "Within-participant nested epoch-CV evaluation"),
        "estimand": ("learning_procedure_at_each_outer_training_subject_set" if scope == "inter"
                     else "within_participant_epoch_generalization"),
        "scope": scope,
        "split_policy": STABLE_SPLIT_POLICY if stable_epoch_ids else LEGACY_SPLIT_POLICY,
        "grouping_unit": "subject_id_all_sessions" if scope == "inter" else "epochs_within_one_subject",
        "performance_column": "accuracy",
        "performance_data": "outer_test_only",
        "selection_diagnostics": {
            "sfs_inner_accuracy": "feature_selection_diagnostic; not_selection_independent",
            "grid_inner_accuracy": "hyperparameter_selection_after_sfs_on_reused_inner_folds; not_selection_independent",
        },
    }
    if cfg is not None and "classifiers" in cfg:
        result["schema_version"] = "eeg-evaluation-contract-2"
        result["classifier_policy"] = classifier_policy(cfg)
    return result


def _write_evaluation_contract(out_dir: Path, scope: str, features: pd.DataFrame, cfg: dict) -> None:
    value = evaluation_contract(scope, stable_epoch_ids="epoch_id" in features, cfg=cfg)
    (out_dir / "evaluation_contract.json").write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_intra_subject(features: pd.DataFrame, cfg: Dict, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe intra-subject evaluation.

    The project/poster specifies the order SFS -> classifier evaluation.  The
    available poster additionally reports an inter-subject *LDA* SFS result,
    but does not provide a separate SFS objective for SVM/KFDA.  The formal
    repo therefore uses training-only LDA inner-CV as the single SFS criterion
    and compares the declared classifiers on exactly the same selected subset.
    This interpretation is explicit in docs/method_definition.md and config.
    """
    features = _validate_evaluation_input(features, cfg, scope="intra")
    out_dir = _new_output_dir(out_dir)
    fcols = feature_columns(features, str(cfg.get("feature_set", "all")))

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    sfs_rows: List[Dict[str, object]] = []
    rf_audits = []
    lightgbm_audits = []

    for comparison in sorted(features["comparison"].unique()):
        comp_df = features.loc[features["comparison"] == comparison].copy()
        for subject in sorted(comp_df["subject_id"].astype(str).unique()):
            df = comp_df.loc[comp_df["subject_id"].astype(str) == subject].reset_index(drop=True)
            X = df[fcols].to_numpy(dtype=float)
            y = df["class_label"].to_numpy(dtype=int)
            _assert_binary_balance(y, f"intra {subject} {comparison}")

            for outer_fold, train_idx, test_idx, inner_splits in _evaluation_indices(df, cfg, "intra"):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]

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

                context = {"scope": "intra", "comparison": str(comparison),
                           "subject": str(subject), "outer_fold": outer_fold}
                models, rf_audit, lightgbm_audit = _fit_classifiers(Xtr_s, ytr, cfg, inner_splits, context=context)

                for model_name, model, best_params, inner_score in models:
                    pred = model.predict(Xte_s)
                    if model_name == "RandomForest":
                        rf_audits.append(_rf_outer_record(rf_audit, df, train_idx, test_idx,
                                                         sfs.selected_names, model, Xte_s, pred))
                    elif model_name == "LightGBM":
                        lightgbm_audits.append(_lightgbm_outer_record(lightgbm_audit, df, train_idx, test_idx,
                                                                    sfs.selected_names, model, Xte_s, pred))
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
                            **_prediction_metadata(df.iloc[local_i]),
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
    _write_rf_evidence(out_dir, "intra", features, cfg, rf_audits)
    _write_lightgbm_evidence(out_dir, "intra", features, cfg, lightgbm_audits)
    _write_evaluation_contract(out_dir, "intra", features, cfg)
    return folds, preds


def run_inter_subject(features: pd.DataFrame, cfg: Dict, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe outer LOPO-CV with group-aware inner validation."""
    features = _validate_evaluation_input(features, cfg, scope="inter")
    out_dir = _new_output_dir(out_dir)
    fcols = feature_columns(features, str(cfg.get("feature_set", "all")))

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    sfs_rows: List[Dict[str, object]] = []
    rf_audits = []
    lightgbm_audits = []

    for comparison in sorted(features["comparison"].unique()):
        df = features.loc[features["comparison"] == comparison].reset_index(drop=True)
        X = df[fcols].to_numpy(dtype=float)
        y = df["class_label"].to_numpy(dtype=int)
        groups = df["subject_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise ValueError("Inter-subject LOPO 至少需要 3 位受試者，才能保留 outer test 並在 training groups 做 inner LOPO。")

        for outer_fold, train_idx, test_idx, inner_splits in _evaluation_indices(df, cfg, "inter"):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            gtr = groups[train_idx]
            held_out = str(np.unique(groups[test_idx])[0])

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

            context = {"scope": "inter", "comparison": str(comparison),
                       "subject": held_out, "outer_fold": outer_fold}
            models, rf_audit, lightgbm_audit = _fit_classifiers(Xtr_s, ytr, cfg, inner_splits,
                                                             context=context, groups=gtr)

            for model_name, model, best_params, inner_score in models:
                pred = model.predict(Xte_s)
                if model_name == "RandomForest":
                    rf_audits.append(_rf_outer_record(rf_audit, df, train_idx, test_idx,
                                                     sfs.selected_names, model, Xte_s, pred))
                elif model_name == "LightGBM":
                    lightgbm_audits.append(_lightgbm_outer_record(lightgbm_audit, df, train_idx, test_idx,
                                                                sfs.selected_names, model, Xte_s, pred))
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
                        **_prediction_metadata(df.iloc[local_i]),
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
    _write_rf_evidence(out_dir, "inter", features, cfg, rf_audits)
    _write_lightgbm_evidence(out_dir, "inter", features, cfg, lightgbm_audits)
    _write_evaluation_contract(out_dir, "inter", features, cfg)
    return folds, preds


def _validate_evaluation_input(features: pd.DataFrame, cfg: Dict, *, scope=None) -> pd.DataFrame:
    """Keep historical learning semantics, but reject ambiguous typed inputs.

    The low-level CV functions retain support for small, schema-less legacy
    fixtures. Trusted CLI runs require schema v2 or an explicit legacy adapter
    before calling these functions; no acquisition identity is invented here.
    """
    policy = classifier_policy(cfg)
    required = {"subject_id", "comparison", "class_label"}
    if features.empty or not required.issubset(features.columns):
        raise ValueError("Evaluation needs non-empty subject/comparison/class labels")
    for key in ("subject_id", "comparison"):
        if features[key].isna().any() or features[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing evaluation identity: {key}")
    if "schema_version" in features or "epoch_id" in features:
        from .schema import validate_feature_table
        validate_feature_table(features)
    labels = pd.to_numeric(features["class_label"], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("class_label must contain only binary labels 0 and 1")
    fcols = feature_columns(features, str(cfg.get("feature_set", "all")))
    if not np.isfinite(features[fcols].to_numpy(dtype=float)).all():
        raise ValueError("Evaluation features contain NaN or Inf")
    if (policy["random_forest"] is not None or policy.get("lightgbm") is not None) and scope is not None:
        sizes = []
        for _, comparison in features.groupby("comparison", sort=True):
            groups = ([part.reset_index(drop=True) for _, part in comparison.groupby("subject_id", sort=True)]
                      if scope == "intra" else [comparison.reset_index(drop=True)])
            for frame in groups:
                for _, outer_train, _, inner in _evaluation_indices(frame, cfg, scope):
                    sizes.extend(len(train) for train, _ in inner)
                    if policy.get("lightgbm") is not None:
                        outer_y = frame["class_label"].to_numpy(dtype=int)[outer_train]
                        if any(np.unique(outer_y[train]).tolist() != [0, 1] for train, _ in inner):
                            raise ValueError("Every LightGBM inner-training partition must contain both classes")
        cap = cfg["sfs"].get("max_features")
        available = min(len(fcols), int(cap)) if cap is not None else len(fcols)
        if policy["random_forest"] is not None:
            from .classifier_config import forest_constructor_parameters
            from .random_forest import HandcraftedRandomForestClassifier
            for candidate in policy["random_forest"]["candidates"]:
                HandcraftedRandomForestClassifier(**forest_constructor_parameters(candidate)).validate_parameters(
                    n_features=available, n_samples=min(sizes))
        if policy.get("lightgbm") is not None:
            from .classifier_config import lightgbm_constructor_parameters
            from .lightgbm import HandcraftedLightGBMClassifier
            for candidate in policy["lightgbm"]["candidates"]:
                HandcraftedLightGBMClassifier(**lightgbm_constructor_parameters(candidate)).validate_parameters(
                    n_features=available, n_samples=min(sizes))
    return features


def _new_output_dir(path: str | Path) -> Path:
    from .provenance import protect_output
    path = protect_output(path)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"Refusing to overwrite evaluation output: {path}; use a new run or validated resume")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prediction_metadata(row: pd.Series) -> Dict[str, object]:
    if "epoch_id" not in row.index:
        return {"identity_status": "legacy_untracked"}
    from .schema import META_COLUMNS
    return {**{key: row[key] for key in META_COLUMNS}, "identity_status": "schema_validated"}


def _fit_classifiers(Xtr, ytr, cfg, inner_splits, *, context, groups=None):
    models, rf_audit, lightgbm_audit = [], None, None
    for name in resolve_classifiers(cfg):
        if name == "LDA":
            models.append((name, lda_pipeline().fit(Xtr, ytr), {}, np.nan))
        elif name == "RBF-SVM":
            search = svm_grid_search(cfg, inner_splits).fit(Xtr, ytr)
            models.append((name, search.best_estimator_, search.best_params_, float(search.best_score_)))
        elif name == "KFDA":
            search = kfda_grid_search(cfg, inner_splits).fit(Xtr, ytr)
            models.append((name, search.best_estimator_, search.best_params_, float(search.best_score_)))
        elif name == "RandomForest":
            search = fit_random_forest(cfg, Xtr, ytr, inner_splits, context=context, groups=groups)
            models.append((name, search.model, search.best_params, search.best_score))
            rf_audit = search.audit
        elif name == "LightGBM":
            search = fit_lightgbm(cfg, Xtr, ytr, inner_splits, context=context, groups=groups)
            models.append((name, search.model, search.best_params, search.best_score))
            lightgbm_audit = search.audit
    return models, rf_audit, lightgbm_audit


def _rf_outer_record(audit, frame, train, test, selected_names, model, Xtest, prediction):
    """Only the completed model sees held-out feature values for prediction."""
    if "epoch_id" in frame:
        identities = frame.epoch_id.astype(str).to_numpy()
        identity_status = "schema_validated"
    else:
        identities = np.array([f"legacy-row-{i}" for i in range(len(frame))])
        identity_status = "legacy_untracked"
    return {**audit, "identity_status": identity_status,
        "selected_features": list(selected_names), "train_epoch_ids": identities[train].tolist(),
        "test_epoch_ids": identities[test].tolist(), "test_classes": model.classes_.tolist(),
        "test_predictions": prediction.tolist(), "test_probabilities": model.predict_proba(Xtest).tolist()}


def _write_rf_evidence(out_dir, scope, features, cfg, records):
    if "RandomForest" not in resolve_classifiers(cfg):
        return
    from .provenance import sha256_file
    source = out_dir / "random_forest_inputs.csv"
    # Exactly the same serialization is hashed by build_run_identity; the
    # complete source table binds selected train/test tensors to that identity.
    features.to_csv(source, index=False, encoding="utf-8", lineterminator="\n")
    audit = {"schema_version": "eeg-random-forest-cv-1", "scope": scope,
             "classifier_policy": classifier_policy(cfg), "input_table_sha256": sha256_file(source),
             "verification_scope": "source_bound_outer_model_prediction_replay_and_inner_selection_record_consistency_not_inner_refit",
             "folds": records}
    (out_dir / "random_forest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _lightgbm_outer_record(audit, frame, train, test, selected_names, model, Xtest, prediction):
    return {**_rf_outer_record(audit, frame, train, test, selected_names, model, Xtest, prediction),
            "test_raw_scores": model.decision_function(Xtest).tolist()}


def _write_lightgbm_evidence(out_dir, scope, features, cfg, records):
    if "LightGBM" not in resolve_classifiers(cfg):
        return
    from .lightgbm_audit import VERIFICATION_SCOPE
    from .provenance import sha256_file
    source = out_dir / "lightgbm_inputs.csv"
    features.to_csv(source, index=False, encoding="utf-8", lineterminator="\n")
    audit = {"schema_version": "eeg-lightgbm-cv-1", "scope": scope,
             "classifier_policy": classifier_policy(cfg), "input_table_sha256": sha256_file(source),
             "verification_scope": VERIFICATION_SCOPE, "folds": records}
    (out_dir / "lightgbm_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _evaluation_indices(df: pd.DataFrame, cfg: Dict, scope: str):
    """Single source of truth for execution and the recorded split manifest.

    Versioned tables use canonical epoch-ID order before applying the original
    splitter and seed. CSV row shuffling then cannot silently change membership
    or training order. Outer indices are mapped back to the supplied frame;
    inner indices remain relative to the yielded outer-training sequence.

    Schema-less legacy calls retain the historical positional ordering and
    splitter parameters. Existing historical results are never rewritten.
    """
    order = (np.argsort(df["epoch_id"].astype(str).to_numpy(), kind="stable")
             if "epoch_id" in df else np.arange(len(df)))
    y = df["class_label"].to_numpy(dtype=int)[order]
    X = np.zeros((len(df), 1))
    if scope == "intra":
        _assert_binary_balance(y, "intra")
        seed = int(cfg["cv"]["random_state"])
        outer = StratifiedKFold(n_splits=int(cfg["cv"]["intra_outer_folds"]), shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(outer.split(X, y), start=1):
            inner = StratifiedKFold(n_splits=int(cfg["cv"]["intra_inner_folds"]), shuffle=True, random_state=seed + fold)
            yield fold, order[train], order[test], list(inner.split(X[train], y[train]))
    elif scope == "inter":
        groups = df["subject_id"].astype(str).to_numpy()[order]
        if len(np.unique(groups)) < 3:
            raise ValueError("Inter-subject LOPO requires at least 3 participants")
        for fold, (train, test) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
            inner = list(LeaveOneGroupOut().split(X[train], y[train], groups[train]))
            yield fold, order[train], order[test], inner
    else:
        raise ValueError(f"Unknown evaluation scope: {scope}")


def make_evaluation_splits(features: pd.DataFrame, cfg: Dict, scope: str) -> List[Dict[str, object]]:
    """Return the exact outer/inner epoch membership used by the CV runners.

    Unlike the low-level numerical compatibility API, manifests require
    versioned epoch identity; legacy tables must first be explicitly mapped.
    """
    from .schema import validate_feature_table
    validate_feature_table(features)
    _validate_evaluation_input(features, cfg, scope=scope)
    return _split_records(features, cfg, scope)


def validate_evaluation_splits(metadata: list[dict], cfg: Dict, scope: str, manifest: list[dict]) -> None:
    """Reconstruct stable-ID partitions from metadata, without reading features.

    This guards archived new-format runs against rehashed split declarations,
    including inner/outer overlap and subjects with sessions on both sides.
    """
    frame = pd.DataFrame(metadata)
    required = {"subject_id", "comparison", "class_label", "epoch_id"}
    if frame.empty or not required.issubset(frame.columns) or frame["epoch_id"].duplicated().any():
        raise ValueError("Split metadata must contain unique stable epoch identities")
    for key in ("subject_id", "comparison", "epoch_id"):
        if any(not isinstance(value, str) or not value.strip() for value in frame[key]):
            raise ValueError(f"Invalid split identity: {key}")
    if not frame["class_label"].isin([0, 1]).all():
        raise ValueError("Split metadata requires binary labels")
    if manifest != _split_records(frame, cfg, scope):
        raise ValueError("Split manifest differs from stable-ID reconstruction")


def _split_records(features: pd.DataFrame, cfg: Dict, scope: str) -> List[Dict[str, object]]:
    manifest = []
    for comparison in sorted(features["comparison"].unique()):
        comp = features.loc[features["comparison"] == comparison]
        if scope == "intra":
            frames = [(s, comp.loc[comp["subject_id"].astype(str) == s].reset_index(drop=True))
                      for s in sorted(comp["subject_id"].astype(str).unique())]
        elif scope == "inter":
            frames = [(None, comp.reset_index(drop=True))]
        else:
            raise ValueError(f"Unknown evaluation scope: {scope}")
        for subject, frame in frames:
            ids = frame["epoch_id"].astype(str).to_numpy()
            groups = frame["subject_id"].astype(str).to_numpy()
            for fold, train, test, inner in _evaluation_indices(frame, cfg, scope):
                train_ids = ids[train]
                train_groups = groups[train]
                row = {"comparison": str(comparison), "outer_fold": fold, "split_policy": STABLE_SPLIT_POLICY,
                       "train_epoch_ids": ids[train].tolist(), "test_epoch_ids": ids[test].tolist(),
                       "train_subject_ids": sorted(set(groups[train])),
                       "test_subject_ids": sorted(set(groups[test])),
                       "inner_splits": [{"train_epoch_ids": train_ids[a].tolist(),
                                         "test_epoch_ids": train_ids[b].tolist(),
                                         "train_subject_ids": sorted(set(train_groups[a])),
                                         "test_subject_ids": sorted(set(train_groups[b]))} for a, b in inner]}
                row["subject_id" if scope == "intra" else "held_out_subject"] = subject if scope == "intra" else str(groups[test][0])
                manifest.append(row)
    return manifest


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

    Selection is common to every declared downstream classifier, so counting it
    from the classifier-specific fold table would multiply every occurrence by
    the classifier count and give a misleading frequency. The SFS path is the source of truth.
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
