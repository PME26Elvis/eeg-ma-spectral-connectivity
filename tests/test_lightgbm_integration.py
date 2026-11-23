"""Actual boosting CV, train-only boundaries and source-bound archive replay."""
from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from eeg_ma.classifier_config import classifier_policy
from eeg_ma.evaluation import make_evaluation_splits, run_inter_subject, run_intra_subject
from eeg_ma.experiment_runs import execute_evaluation
from eeg_ma.lightgbm import HandcraftedLightGBMClassifier
from eeg_ma.lightgbm_audit import VERIFICATION_SCOPE, _verify_training
from eeg_ma.models import fit_lightgbm, lightgbm_seed, random_forest_seed
from eeg_ma.provenance import (
    ProvenanceError, RunTransaction, _artifacts, build_run_identity, fingerprint,
    read_csv_literal, validate_complete_run,
)
from test_provenance import CODE, ENV, fixture_data
from test_split_contract import multisession_data


def lightgbm_data(*, multi=False, mixed=False):
    frame, cfg = multisession_data() if multi else fixture_data()
    cfg["feature_set"] = "bp"
    cfg["classifiers"] = ["LDA", "RandomForest", "LightGBM"] if mixed else ["LightGBM"]
    if mixed:
        cfg["random_forest"] = {"parameters": {"n_estimators": 3, "max_depth": 3,
            "oob_score": True, "random_state": 403, "n_jobs": 1}}
    cfg["lightgbm"] = {"parameters": {"n_estimators": 4, "random_state": 405,
        "max_bin": 7, "min_data_in_leaf": 1, "lambda_l2": .2, "n_jobs": 1},
        "grid": {"num_leaves": [2, 3]}}
    return frame, cfg


@pytest.fixture(scope="module")
def lightgbm_completed_base(tmp_path_factory):
    frame, cfg = lightgbm_data(mixed=True)
    identity = build_run_identity(frame, cfg, ["inter", "intra"], code=CODE, environment=ENV)
    root = tmp_path_factory.mktemp("lightgbm-evaluation") / "run"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    return root, frame, cfg, identity


@pytest.fixture
def lightgbm_completed(lightgbm_completed_base, tmp_path):
    source, frame, cfg, identity = lightgbm_completed_base
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    return root, frame.copy(deep=True), copy.deepcopy(cfg), copy.deepcopy(identity)


def _rehash(root):
    path = root / "run_manifest.json"
    value = json.loads(path.read_text())
    value["run_id"] = fingerprint(value["identity"])
    value["artifacts"] = _artifacts(root)
    path.write_text(json.dumps(value, allow_nan=False))


def test_actual_boosting_publishes_both_scopes_with_source_replay(lightgbm_completed_base):
    root, frame, cfg, identity = lightgbm_completed_base
    manifest = validate_complete_run(root)
    assert manifest["identity"]["identity_version"] == "evaluation-3"
    assert manifest["identity"]["classifier_policy"] == classifier_policy(cfg)
    assert classifier_policy(cfg)["schema_version"] == "eeg-classifier-policy-2"
    for scope in ("inter", "intra"):
        audit = json.loads((root / scope / "lightgbm_audit.json").read_text())
        assert audit["verification_scope"] == VERIFICATION_SCOPE
        assert audit["input_table_sha256"] == identity["dataset"]["table_sha256"]
        table = read_csv_literal(root / scope / f"{scope}_predictions.csv")
        assert set(table.classifier) == {"LDA", "RandomForest", "LightGBM"}
        assert len(table) == 3 * len(frame)
        assert set(read_csv_literal(root / scope / "lightgbm_inputs.csv").subject_id) == {"001", "1", "NA"}
        assert len(audit["folds"]) == len(identity["splits"][scope])
        for record in audit["folds"]:
            assert len(record["candidates"]) == 2
            assert record["best_params"]["early_stopping_rounds"] is None
            assert record["model"]["validation_loss"] is None
            assert len(record["model"]["training_loss"]) == len(record["model"]["iterations"]) + 1
            assert record["feature_importance_scope"].startswith("training_newton_gain")
            assert record["best_params"]["random_state"] != 405


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_every_boosting_fit_and_binner_sees_only_declared_training_rows(scope, tmp_path, monkeypatch):
    from eeg_ma.lightgbm_histogram import BinMapper
    frame, cfg = lightgbm_data(multi=True)
    splits = make_evaluation_splits(frame, cfg, scope)
    lookup = {float(row["BP__FP1__alpha"]): row["epoch_id"] for _, row in frame.iterrows()}
    expected = []
    for outer in splits:
        for _ in classifier_policy(cfg)["lightgbm"]["candidates"]:
            expected.extend(inner["train_epoch_ids"] for inner in outer["inner_splits"])
        expected.append(outer["train_epoch_ids"])
        for inner in outer["inner_splits"]:
            assert set(inner["train_epoch_ids"]).isdisjoint(outer["test_epoch_ids"])
            if scope == "inter":
                assert set(inner["train_subject_ids"]).isdisjoint(inner["test_subject_ids"])
    calls, bin_calls = [], []
    original_fit = HandcraftedLightGBMClassifier.fit
    original_bins = BinMapper.fit

    def fit_spy(self, X, y, sample_weight=None, **kwargs):
        observed = [lookup[float(value)] for value in X[:, 0]]
        assert observed == expected[len(calls)]
        assert not kwargs.get("eval_set") and sample_weight is None
        np.testing.assert_array_equal(y, frame.set_index("epoch_id").loc[observed].class_label)
        calls.append(observed)
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    def bin_spy(cls, X, **kwargs):
        observed = [lookup[float(value)] for value in X[:, 0]]
        assert observed == expected[len(bin_calls)]
        bin_calls.append(observed)
        return original_bins(X, **kwargs)

    monkeypatch.setattr(HandcraftedLightGBMClassifier, "fit", fit_spy)
    monkeypatch.setattr(BinMapper, "fit", classmethod(bin_spy))
    (run_inter_subject if scope == "inter" else run_intra_subject)(frame, cfg, tmp_path / scope)
    assert calls == expected == bin_calls


@pytest.mark.parametrize("scope", ["inter", "intra"])
def test_stable_epoch_row_shuffle_preserves_boosters_and_candidate_selection(scope, tmp_path):
    frame, cfg = lightgbm_data(multi=True)
    run = run_inter_subject if scope == "inter" else run_intra_subject
    a_folds, a_preds = run(frame, cfg, tmp_path / "first")
    b_folds, b_preds = run(frame.sample(frac=1, random_state=141), cfg, tmp_path / "second")
    pd.testing.assert_frame_equal(a_folds, b_folds)
    columns = ["epoch_id", "classifier", "predicted_label", "true_label", "outer_fold"]
    pd.testing.assert_frame_equal(
        a_preds[columns].sort_values(["epoch_id", "classifier"]).reset_index(drop=True),
        b_preds[columns].sort_values(["epoch_id", "classifier"]).reset_index(drop=True))
    a = json.loads((tmp_path / "first/lightgbm_audit.json").read_text())
    b = json.loads((tmp_path / "second/lightgbm_audit.json").read_text())
    assert a["folds"] == b["folds"]


def test_held_out_features_cannot_change_bins_bundles_search_or_final_booster(tmp_path):
    frame, cfg = lightgbm_data()
    run_inter_subject(frame, cfg, tmp_path / "original")
    changed = frame.copy()
    subject = sorted(frame.subject_id.unique())[0]
    changed.loc[changed.subject_id == subject, "BP__FP1__alpha"] += 100000
    run_inter_subject(changed, cfg, tmp_path / "changed")
    original = json.loads((tmp_path / "original/lightgbm_audit.json").read_text())["folds"]
    mutated = json.loads((tmp_path / "changed/lightgbm_audit.json").read_text())["folds"]
    a = next(row for row in original if row["context"]["subject"] == subject)
    b = next(row for row in mutated if row["context"]["subject"] == subject)
    for field in ("model", "candidates", "best_params", "best_score", "selected_features"):
        assert a[field] == b[field]


def test_boosting_path_never_delegates_to_ready_learners_or_grid_search(tmp_path, monkeypatch):
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    def forbidden(*args, **kwargs):
        pytest.fail("LightGBM delegated to a ready-made learner or search")
    for cls in (AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                RandomForestClassifier, DecisionTreeClassifier, DecisionTreeRegressor, GridSearchCV):
        monkeypatch.setattr(cls, "fit", forbidden)
    frame, cfg = lightgbm_data()
    run_inter_subject(frame, cfg, tmp_path / "manual")


@pytest.mark.parametrize("change", [
    {"classifiers": []}, {"classifiers": ["LightGBM", "LightGBM"]},
    {"classifiers": ["lightgbm"]}, {"classifiers": ["LDA"]},
    {"lightgbm": {"extra": True}},
    {"lightgbm": {"parameters": {"max_depth": 0}}},
    {"lightgbm": {"parameters": {"n_estimators": True}}},
    {"lightgbm": {"parameters": {"unknown": 3}}},
    {"lightgbm": {"parameters": {"random_state": None}}},
    {"lightgbm": {"parameters": {"random_state": -1}}},
    {"lightgbm": {"parameters": {"early_stopping_rounds": 2}}},
    {"lightgbm": {"parameters": {"early_stopping_min_delta": .01}}},
    {"lightgbm": {"parameters": {"class_weight": {"0": 1}}}},
    {"lightgbm": {"parameters": {"class_weight": {"0": 0, "1": 1}}}},
    {"lightgbm": {"parameters": {"class_weight": {"0": True, "1": 1}}}},
    {"lightgbm": {"parameters": {"data_sample_strategy": "goss", "bagging_fraction": .5}}},
    {"lightgbm": {"grid": {"num_leaves": []}}},
    {"lightgbm": {"grid": {"num_leaves": [2, 2]}}},
    {"lightgbm": {"grid": {"random_state": [1, 2]}}},
    {"lightgbm": {"grid": {"n_jobs": [1, 2]}}},
    {"lightgbm": {"grid": {"early_stopping_rounds": [None]}}},
    {"lightgbm": {"grid": {"early_stopping_min_delta": [0, .1]}}},
    {"lightgbm": {"grid": {"num_leaves": list(range(2, 67))}}},
    {"lightgbm": {"grid": {"lambda_l2": [0, float("inf")]}}},
])
def test_invalid_settings_fail_before_feature_selection_or_output(change, tmp_path, monkeypatch):
    import eeg_ma.evaluation as evaluation
    frame, cfg = lightgbm_data()
    cfg.update(copy.deepcopy(change))
    monkeypatch.setattr(evaluation, "forward_select_lda", lambda *a, **k: pytest.fail("selection must not run"))
    with pytest.raises(ValueError):
        run_inter_subject(frame, cfg, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("weight,strategy", [(None, "bagging"), ("balanced", "goss"), ({"0": 1., "1": 3.}, "goss")])
def test_training_local_class_weights_and_sampling_are_source_replayable(weight, strategy, tmp_path):
    frame, cfg = lightgbm_data()
    if weight == "balanced":
        # Training participant mixtures now have different class frequencies.
        frame = frame.loc[~((frame.subject_id == "001") & (frame.class_label == 1)
                            & (frame.epoch_index <= 3))].reset_index(drop=True)
    cfg["lightgbm"]["parameters"].update(class_weight=weight, data_sample_strategy=strategy,
        bagging_fraction=.75 if strategy == "bagging" else 1., top_rate=.3, other_rate=.2)
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    root = tmp_path / "weighted"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    assert validate_complete_run(root)["status"] == "complete"


@pytest.mark.parametrize("strategy", ["bagging", "goss"])
def test_multifeature_bundles_and_feature_subsampling_replay_from_source(strategy):
    X = np.zeros((40, 4))
    for row in range(len(X)):
        X[row, row % 4] = 1. + row / 40
    y = (np.arange(len(X)) % 4 >= 2).astype(int)
    model = HandcraftedLightGBMClassifier(n_estimators=5, min_data_in_leaf=1,
        max_bin=7, num_leaves=4, feature_fraction=.5, data_sample_strategy=strategy,
        bagging_fraction=.8 if strategy == "bagging" else 1., random_state=44).fit(X, y)
    restored = HandcraftedLightGBMClassifier.from_dict(model.to_dict())
    assert len(restored.bundle_layout_.bundles_) == 1
    assert all(len(features) == 2 for features in restored.feature_subsets_)
    _verify_training(restored, X, y)


@pytest.mark.parametrize("changed", ["labels", "feature_values"])
def test_structurally_valid_models_from_different_training_sources_are_rejected(changed):
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.repeat([0, 1], 12)
    trained = HandcraftedLightGBMClassifier(n_estimators=3, min_data_in_leaf=1, max_bin=7,
        num_leaves=3, random_state=82).fit(X + .01 if changed == "feature_values" else X,
                                        1 - y if changed == "labels" else y)
    restored = HandcraftedLightGBMClassifier.from_dict(trained.to_dict())
    with pytest.raises(ProvenanceError, match="gradient|training-only bin"):
        _verify_training(restored, X, y)


def test_prior_only_replay_bounds_seed_allocation_by_materialized_iterations(monkeypatch):
    import eeg_ma.lightgbm as implementation
    X, y = np.ones((12, 1)), np.tile([0, 1], 6)
    model = HandcraftedLightGBMClassifier(n_estimators=2**50, min_data_in_leaf=1).fit(X, y)
    assert model.n_estimators_ == 0
    original = implementation.iteration_seed_sequence
    calls = []
    def bounded(seed, count):
        calls.append(count)
        assert count == len(model.estimators_)
        return original(seed, count)
    monkeypatch.setattr(implementation, "iteration_seed_sequence", bounded)
    restored = HandcraftedLightGBMClassifier.from_dict(model.to_dict())
    _verify_training(restored, X, y)
    assert calls == [0, 0]


def test_source_replay_keeps_finite_weighted_losses_and_gain_importances_at_extreme_scale():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.asarray([0] * 16 + [1] * 4 + [0] * 4 + [1] * 16)
    model = HandcraftedLightGBMClassifier(n_estimators=6, learning_rate=100,
        num_leaves=2, max_bin=2, min_data_in_leaf=1,
        class_weight={0: 1e305, 1: 1e305}, random_state=66).fit(X, y)
    assert model.n_estimators_ == 6
    assert np.isfinite(model.training_loss_).all()
    assert np.max(model.training_loss_) > 500
    _verify_training(HandcraftedLightGBMClassifier.from_dict(model.to_dict()), X, y)


def test_exact_inner_accuracy_ties_keep_first_and_seed_domains_are_isolated():
    X, y = np.ones((12, 1)), np.tile([0, 1], 6)
    groups = np.repeat(["one", "two", "three"], 4)
    splits = [(np.flatnonzero(groups != group), np.flatnonzero(groups == group)) for group in np.unique(groups)]
    _, cfg = lightgbm_data()
    context = {"scope": "inter", "comparison": "rest1_vs_type1", "subject": "held", "outer_fold": 1}
    result = fit_lightgbm(cfg, X, y, splits, context=context, groups=groups)
    assert result.audit["best_candidate_index"] == 0
    assert result.best_score == .5
    assert result.best_params["num_leaves"] == 2
    assert result.best_params["random_state"] == lightgbm_seed(405, context, "refit")
    assert lightgbm_seed(405, context, "refit") != random_forest_seed(405, context, "refit")
    assert all(not fold["early_stopping_used"] for candidate in result.audit["candidates"] for fold in candidate["folds"])


def test_mathematically_equal_inner_means_keep_first_despite_float_reduction(monkeypatch):
    X, y = np.arange(30, dtype=float).reshape(-1, 1), np.tile([0, 1], 15)
    groups = np.repeat(["one", "two", "three"], 10)
    splits = [(np.flatnonzero(groups != group), np.flatnonzero(groups == group)) for group in np.unique(groups)]
    context = {"scope": "inter", "comparison": "rest1_vs_type1", "subject": "held", "outer_fold": 1}
    _, cfg = lightgbm_data()
    cfg["lightgbm"]["parameters"]["n_estimators"] = 1
    seeds = {lightgbm_seed(405, context, "inner", index): index - 1 for index in (1, 2, 3)}
    correct = {2: (0, 0, 3), 3: (0, 1, 2)}

    def controlled_predictions(self, values):
        truth = y[values[:, 0].astype(int)]
        predicted = 1 - truth
        params = self.get_params(deep=False)
        count = correct[params["num_leaves"]][seeds[params["random_state"]]]
        predicted[:count] = truth[:count]
        return predicted

    def controlled_raw(self, values):
        return np.where(controlled_predictions(self, values) == 1, 1., -1.)

    def controlled_probabilities(self, values):
        p = 1 / (1 + np.exp(-controlled_raw(self, values)))
        return np.column_stack((1 - p, p))

    monkeypatch.setattr(HandcraftedLightGBMClassifier, "predict", controlled_predictions)
    monkeypatch.setattr(HandcraftedLightGBMClassifier, "decision_function", controlled_raw)
    monkeypatch.setattr(HandcraftedLightGBMClassifier, "predict_proba", controlled_probabilities)
    result = fit_lightgbm(cfg, X, y, splits, context=context, groups=groups)
    assert [candidate["mean_inner_accuracy"] for candidate in result.audit["candidates"]] == [.1, .1]
    assert result.audit["best_candidate_index"] == 0
    assert result.best_params["num_leaves"] == 2
    assert result.best_score == .1


@pytest.mark.parametrize("fault", ["missing_fold", "duplicate_index", "participant_leakage", "boolean_index", "invalid_context", "single_class_training"])
def test_invalid_inner_partitions_fail_before_any_boosting_fit(fault, monkeypatch):
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
    elif fault == "single_class_training":
        y[:] = 0
    else:
        context["outer_fold"] = True
    _, cfg = lightgbm_data()
    monkeypatch.setattr(HandcraftedLightGBMClassifier, "fit", lambda *a, **k: pytest.fail("invalid input must not fit"))
    with pytest.raises(ValueError):
        fit_lightgbm(cfg, X, y, splits, context=context, groups=groups)


@pytest.mark.parametrize("fault", [
    "source", "missing_audit", "inner_seed", "inner_membership", "inner_score", "inner_probability_boolean",
    "inner_raw_string", "winner", "best_params", "scope", "outer_probability", "outer_raw", "node_gradient",
    "node_hessian", "node_value", "node_count_boolean", "sampling", "feature_subset", "training_loss",
    "bin_threshold", "bundle_state", "source_class_mass", "source_class_weights", "missing_classifier_rows", "downgrade",
])
def test_rehashed_archives_reject_inconsistent_source_search_training_and_models(lightgbm_completed, fault):
    root, _, _, _ = lightgbm_completed
    path = root / "inter/lightgbm_audit.json"
    audit = json.loads(path.read_text())
    record = audit["folds"][0]
    inner = record["candidates"][0]["folds"][0]
    iteration = record["model"]["iterations"][0]
    if fault == "source":
        source = root / "inter/lightgbm_inputs.csv"
        frame = read_csv_literal(source, float_precision="round_trip")
        frame["BP__FP1__alpha"] += .25
        frame.to_csv(source, index=False, lineterminator="\n")
    elif fault == "missing_audit":
        path.unlink()
    elif fault == "inner_seed":
        inner["seed"] += 1
    elif fault == "inner_membership":
        inner["train_indices"][0] = inner["test_indices"][0]
    elif fault == "inner_score":
        inner["accuracy"] += .1
    elif fault == "inner_probability_boolean":
        inner["probabilities"][0][0] = True
    elif fault == "inner_raw_string":
        inner["raw_scores"][0] = str(inner["raw_scores"][0])
    elif fault == "winner":
        record["best_candidate_index"] = 1 - record["best_candidate_index"]
    elif fault == "best_params":
        record["best_params"]["learning_rate"] = .9
    elif fault == "scope":
        audit["verification_scope"] = "inner_models_were_refitted"
    elif fault == "outer_probability":
        record["test_probabilities"][0][0] += .01
    elif fault == "outer_raw":
        record["test_raw_scores"][0] += .01
    elif fault in {"node_gradient", "node_hessian", "node_value"}:
        field = {"node_gradient": "sum_gradient", "node_hessian": "sum_hessian", "node_value": "value"}[fault]
        iteration["tree"][field][0] += .25
    elif fault == "node_count_boolean":
        iteration["tree"]["n_node_samples"][0] = True
    elif fault == "sampling":
        iteration["sampling"]["multipliers"][0] *= 2
    elif fault == "feature_subset":
        iteration["feature_indices"] = []
    elif fault == "training_loss":
        record["model"]["training_loss"][-1] += .1
    elif fault == "bin_threshold":
        feature = record["model"]["bin_mapper"]["features"][0]
        old = feature["thresholds"][0]
        new = old - .01
        feature["thresholds"][0] = new
        for item in record["model"]["iterations"]:
            tree = item["tree"]
            for index, (split_feature, threshold_bin) in enumerate(zip(tree["feature"], tree["threshold_bin"])):
                if split_feature == 0 and threshold_bin == 0:
                    tree["threshold"][index] = new
    elif fault == "bundle_state":
        record["model"]["bundle_layout"]["n_samples_fit"] += 1
    elif fault == "source_class_mass":
        record["model"]["training_class_mass"][0] += .1
    elif fault == "source_class_weights":
        record["model"]["class_weight_factors"][0] += .1
    elif fault == "missing_classifier_rows":
        output = root / "inter/inter_predictions.csv"
        table = read_csv_literal(output)
        table.loc[table.classifier != "LightGBM"].to_csv(output, index=False)
    elif fault == "downgrade":
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["classifier_policy"]["schema_version"] = "eeg-classifier-policy-1"
        manifest_path.write_text(json.dumps(manifest))
    if path.exists():
        path.write_text(json.dumps(audit, allow_nan=False))
    _rehash(root)
    with pytest.raises(ProvenanceError):
        validate_complete_run(root)


def test_unused_lightgbm_evidence_cannot_be_smuggled_into_a_legacy_run(tmp_path):
    frame, cfg = fixture_data()
    cfg["classifiers"] = ["LDA"]
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    root = tmp_path / "legacy"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    (root / "inter/lightgbm_audit.json").write_text("{}")
    _rehash(root)
    with pytest.raises(ProvenanceError, match="undeclared classifier"):
        validate_complete_run(root)


@pytest.mark.parametrize("roster", [None, ["LDA", "RBF-SVM", "KFDA"], ["LDA", "RBF-SVM", "KFDA", "RandomForest"]])
def test_three_and_four_classifier_policies_retain_exact_legacy_shape(roster):
    _, cfg = fixture_data()
    if roster is not None:
        cfg["classifiers"] = roster
    policy = classifier_policy(cfg)
    assert policy["schema_version"] == "eeg-classifier-policy-1"
    assert set(policy) == {"schema_version", "classifiers", "feature_selection", "random_forest"}
    assert "lightgbm" not in policy
    assert policy["feature_selection"] == "shared_training_only_lda_sfs"
