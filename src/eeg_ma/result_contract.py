"""Participant estimands derived from explicitly identified epoch predictions.

This module does no fitting, model selection or inference.  A participant is the
unit of reporting even when several sessions contribute epochs.  Model IDs are
retained as provenance, but are deliberately not a grouping key: LOPO fits a
different model for each held-out participant.
"""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .provenance import fingerprint


class ResultContractError(ValueError):
    """Prediction identity, coverage or a derived result is inconsistent."""


RESULT_GROUP_COLUMNS = [
    "analysis_id", "run_id", "cohort_id", "protocol_id", "experiment",
    "classifier", "comparison",
]
PREDICTION_COLUMNS = RESULT_GROUP_COLUMNS + [
    "model_id", "subject_id", "session_id", "epoch_id", "y_true", "y_pred",
    "eligible_epoch_manifest_id",
]
EPOCH_COLUMNS = [
    "cohort_id", "protocol_id", "comparison", "subject_id", "session_id",
    "epoch_id", "y_true", "eligible_epoch_manifest_id",
]
PREDICTION_ID_COLUMNS = [c for c in PREDICTION_COLUMNS if c not in {"y_true", "y_pred"}]
PARTICIPANT_KEYS = RESULT_GROUP_COLUMNS + ["subject_id"]
PAIR_GROUP_COLUMNS = ["analysis_id", "cohort_id", "protocol_id", "classifier", "comparison"]
METRICS = ["accuracy", "balanced_accuracy", "recall_0", "recall_1"]
COUNT_COLUMNS = ["n_epochs", "n_class_0", "n_class_1", "tn", "fp", "fn", "tp"]


def read_result_csv(path: str | Path) -> pd.DataFrame:
    """Read IDs such as ``001``, ``NA`` and ``nan`` without lossy coercion."""
    data = Path(path).read_text(encoding="utf-8-sig")
    header = next(csv.reader(StringIO(data)), [])
    if not header or len(header) != len(set(header)):
        raise ResultContractError("Result CSV header is empty or contains duplicate columns")
    identifiers = set(PREDICTION_ID_COLUMNS + [
        "reference_analysis_id", "reference_run_id", "reference_experiment",
        "eligible_epochs_sha256", "ba_status", "model_ids_json",
        "held_out_subject", "selected_features", "feature",
    ]) & set(header)
    return pd.read_csv(StringIO(data), keep_default_na=False,
        dtype={c: str for c in identifiers},
        na_values={c: [""] for c in header if c not in identifiers})


def _require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ResultContractError(f"{label} must have unique named columns")
    missing = set(columns) - set(frame)
    if missing:
        raise ResultContractError(f"{label} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ResultContractError(f"{label} is empty")


def _ids(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if any(not isinstance(v, str) or not v or v != v.strip() for v in frame[column]):
            raise ResultContractError(f"{label}.{column} requires literal nonempty string IDs without surrounding whitespace")


def _labels(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if (frame[column].isna().any() or not frame[column].isin([0, 1]).all()
                or any(isinstance(value, (bool, np.bool_)) for value in frame[column])):
            raise ResultContractError(f"{label}.{column} must contain only binary labels 0 and 1")


def _records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict]:
    return frame[list(columns)].sort_values(list(columns)).to_dict("records")


def _exact(actual: pd.DataFrame, expected: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if _records(actual, columns) != _records(expected, columns):
        raise ResultContractError(f"{label} differs: require the exact same subjects and eligible epoch identities, labels and coverage")


def validate_result_matrix(actual: pd.DataFrame, expected: pd.DataFrame,
                           *, keys: Sequence[str] = RESULT_GROUP_COLUMNS) -> None:
    """Reject missing, extra and duplicate cells before selecting any figure rows."""
    _require(actual, keys, "result matrix")
    _require(expected, keys, "expected matrix")
    _ids(actual, keys, "result matrix")
    _ids(expected, keys, "expected matrix")
    if actual.duplicated(list(keys)).any() or expected.duplicated(list(keys)).any():
        raise ResultContractError("Result matrix contains duplicate keys")
    _exact(actual, expected, keys, "Result matrix")


def validate_predictions(frame: pd.DataFrame, *, eligible_epochs: pd.DataFrame | None = None,
                         expected_groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate canonical rows and, when supplied, independent expected coverage.

    An eligible epoch manifest may be shared across runs (``EPOCH_COLUMNS``), or
    scoped to exact result groups by also carrying every result-group column.
    Observed predictions alone never establish that a study matrix is complete.
    """
    _require(frame, PREDICTION_COLUMNS, "predictions")
    _ids(frame, PREDICTION_ID_COLUMNS, "predictions")
    _labels(frame, ["y_true", "y_pred"], "predictions")
    work = frame.copy()
    work[["y_true", "y_pred"]] = work[["y_true", "y_pred"]].astype(int)
    if work.duplicated(RESULT_GROUP_COLUMNS + ["epoch_id"]).any():
        raise ResultContractError("Predictions contain duplicate result/epoch keys")
    # An epoch must never acquire a different owner or label in another model.
    epoch_scope = ["cohort_id", "protocol_id", "epoch_id"]
    immutable = ["subject_id", "session_id", "comparison", "y_true", "eligible_epoch_manifest_id"]
    if (work.groupby(epoch_scope, dropna=False)[immutable].nunique(dropna=False) > 1).any().any():
        raise ResultContractError("An epoch has inconsistent participant, session, comparison, label or eligibility identity")
    groups = work[RESULT_GROUP_COLUMNS].drop_duplicates()
    if expected_groups is not None:
        validate_result_matrix(groups, expected_groups)
    for _, group in work.groupby(RESULT_GROUP_COLUMNS, sort=False, dropna=False):
        if group.eligible_epoch_manifest_id.nunique() != 1:
            raise ResultContractError("A result group combines different eligible epoch manifests")
    if eligible_epochs is not None:
        _require(eligible_epochs, EPOCH_COLUMNS, "eligible epochs")
        _ids(eligible_epochs, [c for c in EPOCH_COLUMNS if c != "y_true"], "eligible epochs")
        _labels(eligible_epochs, ["y_true"], "eligible epochs")
        extra_scope = set(RESULT_GROUP_COLUMNS) - set(EPOCH_COLUMNS)
        supplied_scope = extra_scope & set(eligible_epochs)
        if supplied_scope and supplied_scope != extra_scope:
            raise ResultContractError("Eligible epoch manifest has incomplete result-group scope")
        scoped = bool(supplied_scope)
        manifest_keys = (RESULT_GROUP_COLUMNS if scoped else ["cohort_id", "protocol_id", "comparison"]) + ["epoch_id"]
        if eligible_epochs.duplicated(manifest_keys).any():
            raise ResultContractError("Eligible epoch manifest contains duplicate keys")
        scope_columns = RESULT_GROUP_COLUMNS if scoped else ["cohort_id", "protocol_id", "comparison"]
        observed_scope = groups[scope_columns].drop_duplicates()
        expected_scope = eligible_epochs[scope_columns].drop_duplicates()
        validate_result_matrix(observed_scope, expected_scope, keys=scope_columns)
        for key, group in work.groupby(RESULT_GROUP_COLUMNS, sort=False, dropna=False):
            identity = dict(zip(RESULT_GROUP_COLUMNS, key))
            mask = pd.Series(True, index=eligible_epochs.index)
            for column in scope_columns:
                mask &= eligible_epochs[column].eq(identity[column])
            expected = eligible_epochs.loc[mask]
            _exact(group, expected, EPOCH_COLUMNS, "Prediction eligibility coverage")
    return work.sort_values(RESULT_GROUP_COLUMNS + ["subject_id", "session_id", "epoch_id"]).reset_index(drop=True)


def participant_results(frame: pd.DataFrame, *, eligible_epochs: pd.DataFrame | None = None,
                        expected_groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute one score per participant over all of their eligible sessions."""
    work = validate_predictions(frame, eligible_epochs=eligible_epochs, expected_groups=expected_groups)
    rows = []
    for key, group in work.groupby(PARTICIPANT_KEYS, sort=True, dropna=False):
        truth = group.y_true.to_numpy()
        prediction = group.y_pred.to_numpy()
        tn = int(np.sum((truth == 0) & (prediction == 0)))
        fp = int(np.sum((truth == 0) & (prediction == 1)))
        fn = int(np.sum((truth == 1) & (prediction == 0)))
        tp = int(np.sum((truth == 1) & (prediction == 1)))
        n0, n1 = tn + fp, fn + tp
        recall0 = tn / n0 if n0 else np.nan
        recall1 = tp / n1 if n1 else np.nan
        rows.append({**dict(zip(PARTICIPANT_KEYS, key)),
            "model_ids_json": json.dumps(sorted(group.model_id.unique()), separators=(",", ":")),
            "eligible_epoch_manifest_id": group.eligible_epoch_manifest_id.iloc[0],
            "eligible_epochs_sha256": fingerprint(_records(group, EPOCH_COLUMNS)),
            "n_sessions": int(group.session_id.nunique()), "n_epochs": n0 + n1,
            "n_class_0": n0, "n_class_1": n1, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tn + tp) / (n0 + n1), "recall_0": recall0, "recall_1": recall1,
            "balanced_accuracy": (recall0 + recall1) / 2 if n0 and n1 else np.nan,
            "ba_status": "eligible" if n0 and n1 else ("missing_class_0" if not n0 else "missing_class_1")})
    return pd.DataFrame(rows)


def summarize_participants(participants: pd.DataFrame) -> pd.DataFrame:
    """Equal-participant means and sample SDs; no LOPO confidence interval."""
    _require(participants, PARTICIPANT_KEYS + COUNT_COLUMNS + METRICS + ["ba_status"], "participant results")
    _ids(participants, PARTICIPANT_KEYS, "participant results")
    if participants.duplicated(PARTICIPANT_KEYS).any():
        raise ResultContractError("Participant results contain duplicate participant keys")
    counts = participants[COUNT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ResultContractError("Participant class/confusion counts must be finite nonnegative integers")
    for row in participants.to_dict("records"):
        n0, n1 = row["tn"] + row["fp"], row["fn"] + row["tp"]
        if n0 != row["n_class_0"] or n1 != row["n_class_1"] or n0 + n1 != row["n_epochs"] or not n0 + n1:
            raise ResultContractError("Participant confusion and class counts disagree")
        expected = [(row["tn"] + row["tp"]) / (n0 + n1),
                    ((row["tn"] / n0 + row["tp"] / n1) / 2) if n0 and n1 else np.nan,
                    row["tn"] / n0 if n0 else np.nan, row["tp"] / n1 if n1 else np.nan]
        if not np.allclose([row[m] for m in METRICS], expected, rtol=0, atol=1e-12, equal_nan=True):
            raise ResultContractError("Participant metrics do not follow their confusion counts")
        status = "eligible" if n0 and n1 else ("missing_class_0" if not n0 else "missing_class_1")
        if row["ba_status"] != status:
            raise ResultContractError("Participant BA eligibility contradicts class counts")
    rows = []
    for key, group in participants.groupby(RESULT_GROUP_COLUMNS, sort=True, dropna=False):
        row = {**dict(zip(RESULT_GROUP_COLUMNS, key)), "n_subjects": len(group),
               "n_ba_eligible": int(group.balanced_accuracy.notna().sum()),
               "n_ba_ineligible": int(group.balanced_accuracy.isna().sum()),
               "aggregation": "equal_participant", "uncertainty": "sample_sd_descriptive_no_ci"}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        for count in COUNT_COLUMNS:
            row[count] = int(group[count].sum())
        row["pooled_epoch_accuracy"] = (row["tn"] + row["tp"]) / row["n_epochs"]
        row["pooled_recall_0"] = row["tn"] / row["n_class_0"] if row["n_class_0"] else np.nan
        row["pooled_recall_1"] = row["tp"] / row["n_class_1"] if row["n_class_1"] else np.nan
        row["pooled_epoch_balanced_accuracy"] = ((row["pooled_recall_0"] + row["pooled_recall_1"]) / 2
            if row["n_class_0"] and row["n_class_1"] else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_participant_deltas(reference_predictions: pd.DataFrame,
                              target_predictions: pd.DataFrame) -> pd.DataFrame:
    """Pair only identical participant/epoch/label/eligibility sets, never an inner join."""
    reference = validate_predictions(reference_predictions)
    target = validate_predictions(target_predictions)
    reference_groups = reference[RESULT_GROUP_COLUMNS].drop_duplicates()
    target_groups = target[RESULT_GROUP_COLUMNS].drop_duplicates()
    if reference_groups.duplicated(PAIR_GROUP_COLUMNS).any() or target_groups.duplicated(PAIR_GROUP_COLUMNS).any():
        raise ResultContractError("Paired comparison has ambiguous run or experiment groups")
    validate_result_matrix(reference_groups, target_groups, keys=PAIR_GROUP_COLUMNS)
    paired_columns = PAIR_GROUP_COLUMNS + ["subject_id", "session_id", "epoch_id", "y_true", "eligible_epoch_manifest_id"]
    _exact(reference, target, paired_columns, "Paired comparison")
    reference_units = participant_results(reference)
    target_units = participant_results(target)
    keys = PAIR_GROUP_COLUMNS + ["subject_id"]
    reference_lookup = reference_units.set_index(keys, drop=False)
    rows = []
    for value in target_units.to_dict("records"):
        previous = reference_lookup.loc[tuple(value[k] for k in keys)]
        row = {k: value[k] for k in PARTICIPANT_KEYS}
        row.update({f"reference_{k}": previous[k] for k in ["analysis_id", "run_id", "experiment"]})
        row.update({k: value[k] for k in ["eligible_epoch_manifest_id", "eligible_epochs_sha256", "n_epochs"]})
        for metric in ["accuracy", "balanced_accuracy"]:
            row[f"{metric}_reference"] = previous[metric]
            row[f"{metric}_experiment"] = value[metric]
            row[f"delta_{metric}"] = value[metric] - previous[metric]
        rows.append(row)
    return pd.DataFrame(rows)
