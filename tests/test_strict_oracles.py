"""Independent tiny-oracle and strict budget/contract regression tests."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from eeg_ma.lightgbm import HandcraftedLightGBMClassifier
from eeg_ma.random_forest import HandcraftedRandomForestClassifier
from eeg_ma.strict_oracles import (
    OracleMismatch, OracleNotApplicable, run_tiny_oracle_canary,
    verify_booster_exhaustive_tiny, verify_cart_exhaustive_tiny,
)
from eeg_ma.strict_replay import (
    ABS_TOL, REL_TOL, ReplayBudget, ReplayResourceLimit, StrictReplayError,
    _equal, _preflight,
)


def test_independent_bounded_oracle_canary():
    result = run_tiny_oracle_canary()
    assert result["status"] == "verified_independent_tiny_oracles"
    assert result["n_fixtures"] == 6
    assert set(record["algorithm"] for record in result["coverage"]) == {
        "handcrafted_CART", "handcrafted_histogram_Newton"
    }
    assert sum(record["proof"]["split_nodes_checked"] for record in result["coverage"]) > 0


@pytest.mark.parametrize("seed", range(7))
def test_independent_cart_exhaustive_random_integer_matrices(seed):
    rng = np.random.default_rng(989 + seed)
    X = rng.integers(0, 5, size=(10, 3)).astype(float)
    y = (2 * X[:, 0] - X[:, 1] + X[:, 2] > 4).astype(int)
    # At least two distinct classes; high-entropy fixtures include impossible
    # improvements, exact ties, and shallow terminal nodes.
    if len(set(y)) != 2:
        y[0], y[-1] = 0, 1
    w = rng.integers(1, 4, size=len(y)).astype(float)
    model = HandcraftedRandomForestClassifier(
        n_estimators=1, bootstrap=False, max_features=None,
        max_depth=4, min_samples_leaf=1, random_state=seed,
    ).fit(X, y, sample_weight=w)
    result = verify_cart_exhaustive_tiny(model, X, y, sample_weight=w)
    assert result["nodes_checked"] == model.estimators_[0].node_count


@pytest.mark.parametrize("seed", range(5))
def test_independent_booster_exhaustive_integer_matrices(seed):
    rng = np.random.default_rng(150 + seed)
    X = rng.integers(0, 4, size=(12, 3)).astype(float)
    y = (2 * X[:, 0] + X[:, 1] - X[:, 2] > 3).astype(int)
    if len(set(y)) != 2:
        y[0], y[-1] = 0, 1
    model = HandcraftedLightGBMClassifier(
        n_estimators=3, num_leaves=3, max_depth=3,
        max_bin=7, min_data_in_leaf=1, lambda_l2=1.0,
        random_state=seed, enable_bundle=True,
    ).fit(X, y)
    result = verify_booster_exhaustive_tiny(model, X, y)
    assert result["rounds_checked"] == model.n_estimators_


def test_oracle_rejects_non_applicable_complexity_instead_of_claiming_pass():
    X = np.arange(48, dtype=float).reshape(16, 3)
    y = np.array([0, 1] * 8)
    model = HandcraftedRandomForestClassifier(
        n_estimators=2, bootstrap=True, random_state=12,
    ).fit(X, y)
    with pytest.raises(OracleNotApplicable):
        verify_cart_exhaustive_tiny(model, X, y)
    model = HandcraftedLightGBMClassifier(n_estimators=2, num_leaves=3).fit(X, y)
    with pytest.raises(OracleNotApplicable):
        verify_booster_exhaustive_tiny(model, X, y, max_rows=10)


@pytest.mark.parametrize("kwargs", [
    {"max_model_fits": 0}, {"max_candidates": True},
    {"max_outer_folds": 1.0}, {"max_seconds": float("nan")},
    {"max_projected_nodes": -1},
])
def test_resource_limits_fail_closed_on_invalid_user_inputs(kwargs):
    with pytest.raises(ValueError):
        ReplayBudget(**kwargs)


def test_strict_comparator_checks_schema_types_and_fixed_float_tolerance():
    _equal({"model": [1.0, None, True]}, {"model": [1.0 + 0.5 * ABS_TOL, None, True]}, "test")
    with pytest.raises(StrictReplayError):
        _equal({"model": [1]}, {"model": [True]}, "test")
    with pytest.raises(StrictReplayError):
        _equal({"probabilities": [0.6, 0.4]}, {"probabilities": [0.5, 0.5]}, "test")
    with pytest.raises(StrictReplayError):
        _equal({"a": 1}, {"b": 1}, "test")


def test_resource_preflight_detects_expensive_plans_without_training():
    from eeg_ma.classifier_config import classifier_policy
    cfg = {
        "classifiers": ["RandomForest", "LightGBM"],
        "random_forest": {"parameters": {"n_estimators": 12, "max_depth": 4},
                          "grid": {"min_samples_leaf": [1, 2]}},
        "lightgbm": {"parameters": {"n_estimators": 8, "num_leaves": 3},
                     "grid": {"lambda_l2": [1.0, 2.0]}},
    }
    policy = classifier_policy(cfg)
    assert policy["feature_selection"] == "shared_training_only_lda_sfs"
    split = {
        "train_epoch_ids": [str(i) for i in range(24)],
        "inner_splits": [
            {"train_epoch_ids": [str(i) for i in range(12)],
             "test_epoch_ids": [str(i) for i in range(12, 24)]},
            {"train_epoch_ids": [str(i) for i in range(12, 24)],
             "test_epoch_ids": [str(i) for i in range(12)]},
        ],
    }
    identity = {"config": cfg, "scopes": ["inter"],
                "splits": {"inter": [deepcopy(split) for _ in range(3)]},
                "dataset": {"metadata": []}}
    plan = _preflight(identity, None, ReplayBudget(max_model_fits=64))
    assert plan["complete_model_fits"] == 30
    with pytest.raises(ReplayResourceLimit, match="model_fits"):
        _preflight(identity, None, ReplayBudget(max_model_fits=8))
    with pytest.raises(ReplayResourceLimit, match="outer_folds"):
        _preflight(identity, None, ReplayBudget(max_outer_folds=2))
    with pytest.raises(ReplayResourceLimit, match="projected_nodes"):
        _preflight(identity, None, ReplayBudget(max_projected_nodes=100))
    assert plan["projected_nodes_upper_bound"] > 100
