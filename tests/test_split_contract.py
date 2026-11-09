"""Stable-ID partition and selection-diagnostic boundaries for Issue #11."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from eeg_ma.evaluation import (
    STABLE_SPLIT_POLICY, _evaluation_indices, evaluation_contract,
    make_evaluation_splits, run_inter_subject, run_intra_subject,
)
from eeg_ma.provenance import (
    ProvenanceError, _artifacts, fingerprint, validate_complete_run,
)
from eeg_ma.schema import epoch_identity, validate_feature_table
from eeg_ma.sfs import forward_select_lda
from test_provenance import completed, completed_base, fixture_data


def multisession_data():
    frame, cfg = fixture_data()
    second = frame.copy()
    second["session_id"] += "-second"
    second["recording_id"] += "-second"
    features = [c for c in frame if c.startswith(("BP__", "COH__", "PLV__", "IPLV__"))]
    for i, name in enumerate(features):
        second[name] += np.arange(len(second)) * 1e-5 + (i + 1) * .013
    for index, row in second.iterrows():
        for key, value in epoch_identity(row, row).items():
            second.loc[index, key] = value
    result = pd.concat([frame, second], ignore_index=True)
    validate_feature_table(result)
    return result, cfg


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_v2_row_reordering_reconstructs_exact_same_outer_and_inner_membership(scope):
    frame, cfg = multisession_data()
    original = make_evaluation_splits(frame, cfg, scope)
    for seed in (17, 314):
        assert make_evaluation_splits(frame.sample(frac=1, random_state=seed), cfg, scope) == original
    assert all(row["split_policy"] == STABLE_SPLIT_POLICY for row in original)
    if scope == "inter":
        records = frame.set_index("epoch_id")
        for outer in original:
            train, test = set(outer["train_epoch_ids"]), set(outer["test_epoch_ids"])
            assert train.isdisjoint(test) and train | test == set(frame.epoch_id)
            held_out = outer["held_out_subject"]
            assert test == set(frame.loc[frame.subject_id == held_out, "epoch_id"])
            assert records.loc[list(test)].session_id.nunique() == 2
            for inner in outer["inner_splits"]:
                a, b = set(inner["train_epoch_ids"]), set(inner["test_epoch_ids"])
                assert a.isdisjoint(b) and a | b == train
                assert (a | b).isdisjoint(test)
                assert set(inner["train_subject_ids"]).isdisjoint(inner["test_subject_ids"])
                assert sorted(set(records.loc[list(a), "subject_id"])) == inner["train_subject_ids"]
                assert sorted(set(records.loc[list(b), "subject_id"])) == inner["test_subject_ids"]


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_actual_v2_evaluation_is_invariant_to_row_reordering(scope, tmp_path):
    frame, cfg = multisession_data()
    run = run_inter_subject if scope == "inter" else run_intra_subject
    first_folds, first_predictions = run(frame, cfg, tmp_path / "first")
    second_folds, second_predictions = run(frame.sample(frac=1, random_state=29), cfg, tmp_path / "second")
    unit = "held_out_subject" if scope == "inter" else "subject_id"
    keys = [unit, "comparison", "outer_fold", "classifier"]
    pd.testing.assert_frame_equal(first_folds.sort_values(keys).reset_index(drop=True),
                                  second_folds.sort_values(keys).reset_index(drop=True))
    keys = ["epoch_id", "classifier"]
    columns = [*keys, "true_label", "predicted_label", "outer_fold", "subject_id", "session_id"]
    pd.testing.assert_frame_equal(first_predictions[columns].sort_values(keys).reset_index(drop=True),
                                  second_predictions[columns].sort_values(keys).reset_index(drop=True))
    saved = json.loads((tmp_path / "first/evaluation_contract.json").read_text())
    assert saved == evaluation_contract(scope)
    assert saved["analysis_id"] != "fixed_model_external"
    assert "not_selection_independent" in saved["selection_diagnostics"]["grid_inner_accuracy"]


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_schema_less_legacy_splitter_order_and_seed_are_preserved(scope):
    frame, cfg = fixture_data()
    frame = frame.drop(columns="epoch_id")
    if scope == "intra":
        frame = frame.loc[frame.subject_id == "001"].reset_index(drop=True)
    X, y = np.zeros((len(frame), 1)), frame.class_label.to_numpy()
    seed = cfg["cv"]["random_state"]
    if scope == "intra":
        outer = StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y)
    else:
        outer = LeaveOneGroupOut().split(X, y, frame.subject_id)
    observed = list(_evaluation_indices(frame, cfg, scope))
    for (fold, train, test, inner), (wanted_train, wanted_test) in zip(observed, outer):
        np.testing.assert_array_equal(train, wanted_train)
        np.testing.assert_array_equal(test, wanted_test)
        if scope == "intra":
            wanted_inner = list(StratifiedKFold(2, shuffle=True, random_state=seed + fold).split(X[train], y[train]))
        else:
            wanted_inner = list(LeaveOneGroupOut().split(X[train], y[train], frame.subject_id.to_numpy()[train]))
        for (a, b), (wanted_a, wanted_b) in zip(inner, wanted_inner):
            np.testing.assert_array_equal(a, wanted_a)
            np.testing.assert_array_equal(b, wanted_b)


def test_sfs_inner_scalers_only_receive_the_declared_inner_training_epochs(monkeypatch):
    frame, cfg = multisession_data()
    frame = frame.loc[frame.subject_id == "001"].reset_index(drop=True)
    # Distinct source values let the spy recover exact epoch membership. The
    # actual SFS still fits its normal scaler and LDA inside each inner fold.
    X = np.arange(len(frame), dtype=float).reshape(-1, 1)
    y = frame.class_label.to_numpy()
    outer = next(iter(_evaluation_indices(frame, cfg, "intra")))
    _, train, test, inner = outer
    expected = [train[a].tolist() for a, _ in inner]
    observed = []
    original = StandardScaler.fit

    def fit(self, values, labels=None, **kwargs):
        positions = values[:, 0].astype(int).tolist()
        assert positions == expected[len(observed)]
        assert set(positions).isdisjoint(test)
        np.testing.assert_array_equal(labels, y[positions])
        observed.append(positions)
        return original(self, values, labels, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    forward_select_lda(X[train], y[train], ["signal"], inner, max_features=1)
    assert observed == expected


@pytest.mark.parametrize("fault", ["diagnostic_label", "analysis_id", "missing_contract", "schema_downgrade"])
def test_rehashed_runs_cannot_change_the_estimand_or_promote_selection_scores(completed, fault):
    root, _, _, _ = completed
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    file = root / "inter/evaluation_contract.json"
    contract = json.loads(file.read_text())
    if fault == "diagnostic_label":
        contract["selection_diagnostics"]["grid_inner_accuracy"] = "selection_independent_performance"
        manifest["identity"]["evaluation_contracts"]["inter"] = contract
        file.write_text(json.dumps(contract))
    elif fault == "analysis_id":
        contract["analysis_id"] = "fixed_model_external"
        manifest["identity"]["evaluation_contracts"]["inter"] = contract
        file.write_text(json.dumps(contract))
    elif fault == "missing_contract":
        file.unlink()
    else:
        manifest["identity"].pop("evaluation_contracts")
        manifest["identity"]["identity_version"] = "evaluation-1"
    manifest["run_id"] = fingerprint(manifest["identity"])
    manifest["artifacts"] = _artifacts(root)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ProvenanceError):
        validate_complete_run(root)


@pytest.mark.parametrize("fault", ["inner_outer_overlap", "inner_overlap", "subject_membership"])
def test_rehashed_split_manifest_cannot_claim_false_training_membership(completed, fault):
    root, _, _, _ = completed
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    outer = manifest["identity"]["splits"]["inter"][0]
    inner = outer["inner_splits"][0]
    if fault == "inner_outer_overlap":
        inner["train_epoch_ids"].append(outer["test_epoch_ids"][0])
    elif fault == "inner_overlap":
        inner["train_epoch_ids"].append(inner["test_epoch_ids"][0])
    else:
        inner["test_subject_ids"] = inner["train_subject_ids"]
    splits = manifest["identity"]["splits"]
    manifest["identity"]["split_sha256"] = fingerprint(splits)
    (root / "split_manifest.json").write_text(json.dumps(splits))
    manifest["run_id"] = fingerprint(manifest["identity"])
    manifest["artifacts"] = _artifacts(root)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ProvenanceError, match="split contract"):
        validate_complete_run(root)
