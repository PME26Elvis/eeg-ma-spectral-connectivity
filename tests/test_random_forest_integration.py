"""Actual manual-forest integration, training boundaries and archive replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from eeg_ma.classifier_config import classifier_policy, resolve_classifiers
from eeg_ma.evaluation import make_evaluation_splits, run_inter_subject, run_intra_subject
from eeg_ma.experiment_runs import execute_evaluation
from eeg_ma.models import fit_random_forest, random_forest_seed
from eeg_ma.provenance import (ProvenanceError, RunTransaction, _artifacts,
    build_run_identity, fingerprint, read_csv_literal, validate_complete_run, validate_reference_coverage)
from eeg_ma.random_forest import HandcraftedRandomForestClassifier
from test_provenance import CODE, ENV, fixture_data
from test_split_contract import multisession_data


def rf_data(*, multi=False, mixed=False):
    frame, cfg = multisession_data() if multi else fixture_data()
    cfg["feature_set"] = "bp"
    cfg["classifiers"] = ["LDA", "RandomForest"] if mixed else ["RandomForest"]
    cfg["random_forest"] = {"parameters": {"n_estimators": 5, "random_state": 403,
        "oob_score": True, "max_depth": 4, "n_jobs": 1}, "grid": {"min_samples_leaf": [1, 2]}}
    return frame, cfg


@pytest.fixture(scope="module")
def rf_completed_base(tmp_path_factory):
    frame, cfg = rf_data(mixed=True)
    identity = build_run_identity(frame, cfg, ["inter", "intra"], code=CODE, environment=ENV)
    root = tmp_path_factory.mktemp("rf-evaluation") / "run"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    return root, frame, cfg, identity


@pytest.fixture
def rf_completed(rf_completed_base, tmp_path):
    source, frame, cfg, identity = rf_completed_base
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    return root, frame.copy(deep=True), copy.deepcopy(cfg), copy.deepcopy(identity)


def _rehash(root):
    file = root / "run_manifest.json"
    value = json.loads(file.read_text())
    value["run_id"] = fingerprint(value["identity"])
    value["artifacts"] = _artifacts(root)
    file.write_text(json.dumps(value, allow_nan=False))


def test_actual_forests_publish_complete_source_bound_replayable_runs(rf_completed_base):
    root, frame, cfg, identity = rf_completed_base
    manifest = validate_complete_run(root)
    assert manifest["identity"]["identity_version"] == "evaluation-3"
    assert manifest["identity"]["classifier_policy"] == classifier_policy(cfg)
    for scope in ("inter", "intra"):
        audit = json.loads((root / scope / "random_forest_audit.json").read_text())
        assert audit["input_table_sha256"] == identity["dataset"]["table_sha256"]
        table = read_csv_literal(root / scope / f"{scope}_predictions.csv")
        assert set(table.classifier) == {"LDA", "RandomForest"}
        assert len(table) == 2 * len(frame)
        assert len(audit["folds"]) == len(identity["splits"][scope])
        for record in audit["folds"]:
            assert len(record["candidates"]) == 2
            assert record["oob"]["scope"] == "training_epoch_diagnostic_not_independent_participant_validation"
            assert record["feature_importance_scope"].startswith("training_impurity_decrease")
            assert record["best_params"]["random_state"] != cfg["random_forest"]["parameters"]["random_state"]


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_rf_search_fits_only_exact_inner_and_outer_training_memberships(scope, tmp_path, monkeypatch):
    frame, cfg = rf_data(multi=True)
    splits = make_evaluation_splits(frame, cfg, scope)
    lookup = {float(row["BP__FP1__alpha"]): row["epoch_id"] for _, row in frame.iterrows()}
    assert len(lookup) == len(frame)
    expected = []
    candidate_count = len(classifier_policy(cfg)["random_forest"]["candidates"])
    for outer in splits:
        for _ in range(candidate_count):
            expected.extend(inner["train_epoch_ids"] for inner in outer["inner_splits"])
        expected.append(outer["train_epoch_ids"])
        if scope == "inter":
            records = frame.set_index("epoch_id")
            assert records.loc[outer["test_epoch_ids"]].session_id.nunique() == 2
            for inner in outer["inner_splits"]:
                assert set(inner["train_subject_ids"]).isdisjoint(inner["test_subject_ids"])
                assert set(inner["train_epoch_ids"]).isdisjoint(outer["test_epoch_ids"])
    calls = []
    original = HandcraftedRandomForestClassifier.fit

    def spy(self, X, y, sample_weight=None):
        observed = [lookup[float(value)] for value in X[:, 0]]
        assert observed == expected[len(calls)]
        np.testing.assert_array_equal(y, frame.set_index("epoch_id").loc[observed].class_label)
        calls.append(observed)
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(HandcraftedRandomForestClassifier, "fit", spy)
    (run_inter_subject if scope == "inter" else run_intra_subject)(frame, cfg, tmp_path / scope)
    assert calls == expected


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_rf_row_shuffling_preserves_models_predictions_and_selected_grid(scope, tmp_path):
    frame, cfg = rf_data(multi=True)
    run = run_inter_subject if scope == "inter" else run_intra_subject
    first_folds, first_preds = run(frame, cfg, tmp_path / "first")
    second_folds, second_preds = run(frame.sample(frac=1, random_state=141), cfg, tmp_path / "second")
    pd.testing.assert_frame_equal(first_folds, second_folds)
    columns = ["epoch_id", "classifier", "predicted_label", "true_label", "outer_fold"]
    a = first_preds[columns].sort_values(["epoch_id", "classifier"]).reset_index(drop=True)
    b = second_preds[columns].sort_values(["epoch_id", "classifier"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    first = json.loads((tmp_path / "first/random_forest_audit.json").read_text())
    second = json.loads((tmp_path / "second/random_forest_audit.json").read_text())
    assert first["folds"] == second["folds"]


def test_held_out_features_cannot_change_that_subjects_trained_forest(tmp_path):
    frame, cfg = rf_data()
    run_inter_subject(frame, cfg, tmp_path / "original")
    changed = frame.copy()
    subject = sorted(frame.subject_id.unique())[0]
    changed.loc[changed.subject_id == subject, "BP__FP1__alpha"] += 10000
    run_inter_subject(changed, cfg, tmp_path / "changed")
    original = json.loads((tmp_path / "original/random_forest_audit.json").read_text())["folds"]
    new = json.loads((tmp_path / "changed/random_forest_audit.json").read_text())["folds"]
    a = next(row for row in original if row["context"]["subject"] == subject)
    b = next(row for row in new if row["context"]["subject"] == subject)
    for field in ("model", "candidates", "best_params", "best_score", "selected_features", "oob"):
        assert a[field] == b[field]


def test_rf_path_never_calls_ready_trees_forests_or_sklearn_grid_search(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    def forbidden(*args, **kwargs):
        pytest.fail("RF delegated its implementation to a ready-made learner/search")
    for cls in (RandomForestClassifier, DecisionTreeClassifier, GridSearchCV):
        monkeypatch.setattr(cls, "fit", forbidden)
    frame, cfg = rf_data()
    run_inter_subject(frame, cfg, tmp_path / "manual")


@pytest.mark.parametrize("change", [
    {"classifiers": []}, {"classifiers": ["RandomForest", "RandomForest"]},
    {"classifiers": ["randomforest"]}, {"classifiers": ["LDA"]},
    {"random_forest": {"extra": True}},
    {"random_forest": {"parameters": {"max_depth": 0}}},
    {"random_forest": {"parameters": {"n_estimators": True}}},
    {"random_forest": {"parameters": {"unknown": 3}}},
    {"random_forest": {"parameters": {"random_state": None}}},
    {"random_forest": {"parameters": {"bootstrap": False, "oob_score": True}}},
    {"random_forest": {"parameters": {"max_samples": 100000}}},
    {"random_forest": {"grid": {"max_depth": []}}},
    {"random_forest": {"grid": {"max_depth": [2, 2]}}},
    {"random_forest": {"grid": {"random_state": [1, 2]}}},
    {"random_forest": {"grid": {"n_jobs": [1, 2]}}},
    {"random_forest": {"grid": {"min_samples_leaf": list(range(1, 66))}}},
])
def test_invalid_classifier_settings_fail_before_selection_or_output(change, tmp_path, monkeypatch):
    import eeg_ma.evaluation as evaluation
    frame, cfg = rf_data()
    cfg.update(copy.deepcopy(change))
    monkeypatch.setattr(evaluation, "forward_select_lda", lambda *a, **k: pytest.fail("selection must not run"))
    with pytest.raises(ValueError):
        run_inter_subject(frame, cfg, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("weight", ["balanced", {"0": 1.0, "1": 3.0}])
def test_class_weights_are_training_local_and_source_replayable(weight, tmp_path):
    frame, cfg = rf_data()
    cfg["random_forest"]["parameters"]["class_weight"] = weight
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    root = tmp_path / "weighted"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    assert validate_complete_run(root)["status"] == "complete"


def test_inner_candidate_ties_keep_first_and_do_not_use_oob():
    X = np.ones((12, 1))
    y = np.tile([0, 1], 6)
    groups = np.repeat(["one", "two", "three"], 4)
    splits = [(np.flatnonzero(groups != value), np.flatnonzero(groups == value)) for value in np.unique(groups)]
    _, cfg = rf_data()
    cfg["random_forest"]["parameters"].update(bootstrap=False, oob_score=False)
    context = {"scope": "inter", "comparison": "rest1_vs_type1", "subject": "held", "outer_fold": 1}
    result = fit_random_forest(cfg, X, y, splits, context=context, groups=groups)
    assert result.audit["best_candidate_index"] == 0
    assert result.best_score == .5
    assert result.best_params["min_samples_leaf"] == 1
    assert result.best_params["random_state"] == random_forest_seed(403, context, "refit")
    assert all(not fold["oob_score_requested_during_search"] for row in result.audit["candidates"] for fold in row["folds"])


def test_mathematically_equal_inner_means_keep_first_despite_float_reduction(monkeypatch):
    """Only validation decisions are controlled; CART fitting still executes.

    The two candidates have 3/30 correct validation decisions, distributed
    differently across three equally sized folds. Naive float means rank the
    second candidate higher even though the stipulated objective is equal.
    """
    X, y = np.arange(30, dtype=float).reshape(-1, 1), np.tile([0, 1], 15)
    groups = np.repeat(["one", "two", "three"], 10)
    splits = [(np.flatnonzero(groups != value), np.flatnonzero(groups == value)) for value in np.unique(groups)]
    context = {"scope": "inter", "comparison": "rest1_vs_type1", "subject": "held", "outer_fold": 1}
    _, cfg = rf_data()
    cfg["random_forest"]["parameters"].update(n_estimators=1, bootstrap=False, oob_score=False)
    seeds = {random_forest_seed(403, context, "inner", index): index - 1 for index in (1, 2, 3)}
    correct = {1: (0, 0, 3), 2: (0, 1, 2)}
    assert sum(correct[1]) == sum(correct[2]) == 3

    def controlled_predictions(self, values):
        truth = y[values[:, 0].astype(int)]
        predicted = 1 - truth
        count = correct[self.min_samples_leaf][seeds[self.random_state]]
        predicted[:count] = truth[:count]
        return predicted

    def controlled_probabilities(self, values):
        predicted = controlled_predictions(self, values)
        return np.column_stack((predicted == 0, predicted == 1)).astype(float)

    monkeypatch.setattr(HandcraftedRandomForestClassifier, "predict", controlled_predictions)
    monkeypatch.setattr(HandcraftedRandomForestClassifier, "predict_proba", controlled_probabilities)
    result = fit_random_forest(cfg, X, y, splits, context=context, groups=groups)
    assert [row["mean_inner_accuracy"] for row in result.audit["candidates"]] == [0.1, 0.1]
    assert result.audit["best_candidate_index"] == 0
    assert result.best_params["min_samples_leaf"] == 1
    assert result.best_score == 0.1


@pytest.mark.parametrize("fault", ["missing_fold", "duplicate_index", "participant_leakage", "boolean_index", "invalid_context"])
def test_invalid_inner_partitions_or_seed_context_fail_before_any_fit(fault, monkeypatch):
    X, y = np.ones((12, 1)), np.tile([0, 1], 6)
    groups = np.repeat(["one", "two", "three"], 4)
    splits = [(np.flatnonzero(groups != value), np.flatnonzero(groups == value)) for value in np.unique(groups)]
    context = {"scope": "inter", "comparison": "rest1_vs_type1", "subject": "held", "outer_fold": 1}
    if fault == "missing_fold":
        splits.pop()
    elif fault == "duplicate_index":
        splits[0][0][1] = splits[0][0][0]
    elif fault == "participant_leakage":
        splits = [(np.arange(6, 12), np.arange(6)), (np.arange(6), np.arange(6, 12))]
    elif fault == "boolean_index":
        splits[0] = (np.ones(12, dtype=bool), splits[0][1])
    else:
        context["outer_fold"] = True
    _, cfg = rf_data()
    monkeypatch.setattr(HandcraftedRandomForestClassifier, "fit", lambda *a, **k: pytest.fail("invalid input must not fit"))
    with pytest.raises(ValueError):
        fit_random_forest(cfg, X, y, splits, context=context, groups=groups)


@pytest.mark.parametrize("fault", ["source", "missing_audit", "inner_seed", "inner_membership", "inner_score", "boolean_probability", "winner", "best_params", "scope", "outer_probability", "tree_source", "coherent_oob", "missing_classifier_rows", "downgrade"])
def test_rehashed_rf_archives_reject_inconsistent_source_search_or_models(rf_completed, fault):
    root, _, _, _ = rf_completed
    path = root / "inter/random_forest_audit.json"
    audit = json.loads(path.read_text())
    record = audit["folds"][0]
    if fault == "source":
        source = root / "inter/random_forest_inputs.csv"
        frame = read_csv_literal(source, float_precision="round_trip")
        frame["BP__FP1__alpha"] += .25
        frame.to_csv(source, index=False, lineterminator="\n")
    elif fault == "missing_audit":
        path.unlink()
    elif fault == "inner_seed":
        record["candidates"][0]["folds"][0]["seed"] += 1
    elif fault == "inner_membership":
        inner = record["candidates"][0]["folds"][0]
        inner["train_indices"], inner["test_indices"] = inner["test_indices"], inner["train_indices"]
    elif fault == "inner_score":
        record["candidates"][0]["folds"][0]["accuracy"] = .123456
    elif fault == "boolean_probability":
        record["test_probabilities"][0] = [True, False]
        record["test_predictions"][0] = 0
    elif fault == "winner":
        record["best_candidate_index"] = 1 - record["best_candidate_index"]
    elif fault == "best_params":
        record["best_params"]["random_state"] += 1
    elif fault == "scope":
        record["feature_importance_scope"] = "causal_neural_importance"
    elif fault == "outer_probability":
        record["test_probabilities"][0] = [.123, .877]
        record["test_predictions"][0] = 1
    elif fault == "tree_source":
        tree = next(t for t in record["model"]["trees"] if t["feature"][0] >= 0)
        tree["threshold"][0] += 10000
    elif fault == "coherent_oob":
        classes = record["model"]["classes"]["values"]
        assert classes == [0, 1]
        covered = np.flatnonzero(np.array(record["oob"]["counts"]) > 0)
        for row in covered:
            record["oob"]["probabilities"][row] = [1.0, 0.0]
            record["model"]["oob"]["decision_function"][row] = [1.0, 0.0]
        truth = read_csv_literal(root / "inter/random_forest_inputs.csv").set_index("epoch_id").loc[record["train_epoch_ids"]].class_label.to_numpy()
        score = float(np.mean(truth[covered] == 0))
        record["oob"]["accuracy"] = record["model"]["oob"]["score"] = score
    elif fault == "missing_classifier_rows":
        for name in ("inter_folds.csv", "inter_predictions.csv"):
            table_path = root / "inter" / name
            data = read_csv_literal(table_path)
            data = data.loc[data.classifier != "RandomForest"]
            data.to_csv(table_path, index=False)
    else:
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["identity_version"] = "evaluation-2"
        manifest_path.write_text(json.dumps(manifest))
    if fault != "missing_audit":
        path.write_text(json.dumps(audit, allow_nan=False))
    _rehash(root)
    with pytest.raises(ProvenanceError):
        validate_complete_run(root)


def test_mismatched_classifier_reference_is_rejected(rf_completed_base):
    root, _, _, identity = rf_completed_base
    changed = copy.deepcopy(identity)
    changed["config"]["classifiers"] = ["LDA"]
    changed["config"].pop("random_forest")
    with pytest.raises(ProvenanceError, match="settings differ"):
        validate_reference_coverage(root, changed)


@pytest.mark.parametrize("replacement", ["boolean", "string"])
def test_rehashed_oob_probabilities_reject_nonnumeric_types_even_when_values_match(rf_completed, replacement):
    root, _, _, _ = rf_completed
    path = root / "inter/random_forest_audit.json"
    audit = json.loads(path.read_text())
    changed = False
    for record in audit["folds"]:
        for count, row in zip(record["oob"]["counts"], record["oob"]["probabilities"]):
            if count and all(value in (0.0, 1.0) for value in row):
                row[:] = [bool(value) if replacement == "boolean" else str(value) for value in row]
                changed = True
                break
        if changed:
            break
    assert changed, "Fixture must include a covered OOB row with pure probabilities"
    path.write_text(json.dumps(audit, allow_nan=False))
    _rehash(root)
    with pytest.raises(ProvenanceError, match="OOB source replay must contain numeric values"):
        validate_complete_run(root)


def test_rehashed_outer_probabilities_reject_numeric_strings_without_changing_values(rf_completed):
    root, _, _, _ = rf_completed
    path = root / "inter/random_forest_audit.json"
    audit = json.loads(path.read_text())
    row = audit["folds"][0]["test_probabilities"][0]
    row[:] = [str(value) for value in row]
    path.write_text(json.dumps(audit, allow_nan=False))
    _rehash(root)
    with pytest.raises(ProvenanceError, match="outer predictions probabilities must contain numeric values"):
        validate_complete_run(root)


def test_undeclared_rf_artifacts_cannot_be_attached_to_an_lda_only_run(tmp_path, rf_completed_base):
    rf_root, frame, cfg, _ = rf_completed_base
    cfg = copy.deepcopy(cfg)
    cfg["classifiers"] = ["LDA"]
    cfg.pop("random_forest")
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    root = tmp_path / "lda-only"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    for name in ("random_forest_inputs.csv", "random_forest_audit.json"):
        shutil.copyfile(rf_root / "inter" / name, root / "inter" / name)
    _rehash(root)
    with pytest.raises(ProvenanceError, match="Unexpected RandomForest evidence"):
        validate_complete_run(root)


def test_legacy_roster_and_contract_remain_unchanged():
    frame, cfg = fixture_data()
    assert resolve_classifiers(cfg) == ("LDA", "RBF-SVM", "KFDA")
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    assert identity["identity_version"] == "evaluation-2"
    assert "classifier_policy" not in identity
    assert identity["evaluation_contracts"]["inter"]["schema_version"] == "eeg-evaluation-contract-1"
