"""Parameter, failure-atomicity and serialization-boundary contracts."""
from copy import deepcopy
from fractions import Fraction
import json

import numpy as np
import pytest

import eeg_ma.random_forest as module
from eeg_ma.random_forest import HandcraftedRandomForestClassifier as RF


@pytest.mark.parametrize("params", [
    {"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 1.5},
    {"max_depth": 0}, {"max_depth": False}, {"min_samples_split": 1},
    {"min_samples_leaf": 0}, {"min_samples_leaf": 1.5},
    {"max_features": True}, {"max_features": 0}, {"max_features": 0.0},
    {"max_features": 1.1}, {"max_features": "auto"},
    {"bootstrap": 1}, {"oob_score": 1}, {"bootstrap": False, "oob_score": True},
    {"bootstrap": False, "max_samples": 1}, {"max_samples": 0},
    {"max_samples": True}, {"max_samples": 1.1},
    {"class_weight": {}}, {"class_weight": "balanced_subsample"},
    {"class_weight": {0: float("nan")}}, {"class_weight": {0: -1}},
    {"min_impurity_decrease": -0.1}, {"min_impurity_decrease": float("nan")},
    {"random_state": None}, {"random_state": True}, {"random_state": -1},
    {"random_state": 2**64}, {"n_jobs": 0}, {"n_jobs": -2}, {"n_jobs": True},
])
def test_invalid_parameters_are_rejected_before_fit(params):
    with pytest.raises(ValueError):
        RF(**params)


def test_context_bounds_use_actual_features_and_eligible_rows():
    X = np.arange(8.).reshape(4, 2)
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="max_features"):
        RF(max_features=3).fit(X, y)
    with pytest.raises(ValueError, match="max_samples"):
        RF(max_samples=4).fit(X, y, sample_weight=[1, 0, 1, 1])
    m = RF(n_estimators=3, max_samples=0.5).fit(X, y, sample_weight=[1, 0, 1, 1])
    assert all(len(rows) == 2 for rows in m.bootstrap_indices_)
    assert all(1 not in rows for rows in m.bootstrap_indices_)


@pytest.mark.parametrize("X,y,weights", [
    ([], [], None),
    ([[0], [float("nan")]], [0, 1], None),
    ([[0], [float("inf")]], [0, 1], None),
    ([["0"], ["1"]], [0, 1], None),
    ([[0j], [1j]], [0, 1], None),
    ([[0], [1]], [[0], [1]], None),
    ([[0], [1]], [0, 1], [0, 0]),
    ([[0], [1]], [0, 1], [-1, 1]),
    ([[0], [1]], [0, 1], [1, float("inf")]),
    ([[0], [1]], [0, 1], [float("nan"), 1]),
    ([[0], [1]], [0, 1], [True, False]),
    ([[0], [1]], [0, 1], [1e308, 1e308]),
])
def test_invalid_training_arrays_fail_without_creating_a_fitted_model(X, y, weights):
    model = RF(n_estimators=2)
    with pytest.raises(ValueError):
        model.fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0]])


def test_class_weight_coverage_and_underflow_are_explicit_errors():
    X, y = [[0], [1]], [0, 1]
    for mapping in ({0: 1}, {0: 1, 1: 1, 2: 1}, {0: 0, 1: 0}):
        with pytest.raises(ValueError):
            RF(n_estimators=1, class_weight=mapping).fit(X, y)
    with pytest.raises(ValueError, match="underflow"):
        RF(n_estimators=1, class_weight={0: 1e-200, 1: 1e-200}).fit(X, y, sample_weight=[1e-200, 1e-200])


def test_repeated_bootstrap_draws_reject_overflow_before_numerical_operations():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError, match="Bootstrap draw weight mass"):
            RF(n_estimators=10, random_state=0).fit(
                [[0], [1], [2]], [0, 1, 1], sample_weight=[3e307, 1e-100, 1e-100])


@pytest.mark.parametrize("labels,mapping", [
    ([0, 0, 1, 1], {0: 2, 1: 0.5}),
    (["a", "a", "b", "b"], {"a": 2, "b": 0.5}),
])
def test_explicit_class_weight_keys_survive_json_without_string_coercion(labels, mapping):
    model = RF(n_estimators=7, oob_score=True, class_weight=mapping).fit([[0], [1], [2], [3]], labels)
    state = model.to_dict()
    assert RF.from_dict(json.loads(json.dumps(state, allow_nan=False))).to_dict() == state


def test_tiny_positive_impurity_threshold_does_not_admit_zero_gain_xor_split():
    X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    model = RF(n_estimators=1, bootstrap=False, max_features=None,
               min_impurity_decrease=1e-17).fit(X, y)
    assert model.estimators_[0].node_count == 1


def _rational_global_band_choice(X, labels, weights):
    """Fresh split masks and exact arithmetic, independent of prefix scans."""
    weights = [Fraction.from_float(float(weight)) for weight in weights]

    def cost(rows):
        masses = {}
        for index in rows:
            label = labels[index]
            masses[label] = masses.get(label, Fraction()) + weights[index]
        total = sum(masses.values(), Fraction())
        return total - sum((mass**2 for mass in masses.values()), Fraction()) / total

    all_rows = list(range(len(labels)))
    parent, total = cost(all_rows), sum(weights, Fraction())
    candidates = []
    for feature in range(X.shape[1]):
        distinct = sorted(set(X[:, feature].tolist()))
        for lower, upper in zip(distinct[:-1], distinct[1:]):
            left = [index for index in all_rows if X[index, feature] <= lower]
            right = [index for index in all_rows if X[index, feature] > lower]
            gain = (parent - cost(left) - cost(right)) / total
            candidates.append((gain, feature, lower, upper))
    largest = max(item[0] for item in candidates)
    tolerance = Fraction.from_float(float(64 * np.finfo(float).eps))
    eligible = [item for item in candidates if item[0] >= largest - tolerance]
    return min(eligible, key=lambda item: item[1:3]), largest


def test_feature_ties_use_global_maximum_without_pairwise_tolerance_drift():
    X = np.zeros((5, 3))
    X[:3] = np.eye(3)
    labels = [0, 0, 0, 1, 1]
    weights = [1., 1. + 1.2e-13, 1. + 2.4e-13, 1., 1.]
    chosen, largest = _rational_global_band_choice(X, labels, weights)
    assert chosen[1] == 1  # feature 0 is outside the global tolerance band
    for seed in range(15):
        tree = RF(n_estimators=1, bootstrap=False, max_features=None,
                  max_depth=1, random_state=seed).fit(X, labels, sample_weight=weights).estimators_[0]
        assert tree.feature_[0] == chosen[1]
        assert chosen[2] <= tree.threshold_[0] < chosen[3]
        assert float(largest - chosen[0]) <= 64 * np.finfo(float).eps


def test_local_boundary_and_feature_ties_share_one_global_tolerance_budget():
    X = np.array([[0, 0], [4, 0], [2, 1], [1, 0], [3, 0]], dtype=float)
    labels = [0, 0, 0, 1, 1]
    weights = [1., 1. + 1.2e-13, 1. + 2.4e-13, 1., 1.]
    chosen, _ = _rational_global_band_choice(X, labels, weights)
    assert chosen[1:] == (0, 3., 4.)
    # Feature 0 has a local near-tie at boundary 0/1. That earlier boundary is
    # outside the global band, even though its difference from feature 0's
    # own maximum is within tolerance. Spending the tolerance twice is wrong.
    for seed in range(15):
        tree = RF(n_estimators=1, bootstrap=False, max_features=None,
                  max_depth=1, random_state=seed).fit(X, labels, sample_weight=weights).estimators_[0]
        assert tree.feature_[0] == 0
        assert 3 <= tree.threshold_[0] < 4


def test_failed_refit_does_not_publish_partial_trees(monkeypatch):
    X, y = np.arange(24.).reshape(12, 2), [0, 1] * 6
    model = RF(n_estimators=3, random_state=12).fit(X, y)
    before = json.dumps(model.to_dict(), sort_keys=True, allow_nan=False)
    original = module._fit_tree
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("deliberate tree construction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_tree", fail_second)
    with pytest.raises(ValueError, match="deliberate"):
        model.fit(X + 1, y)
    assert json.dumps(model.to_dict(), sort_keys=True, allow_nan=False) == before


def test_parameter_updates_are_atomic_and_valid_updates_invalidate_fit():
    model = RF(n_estimators=3).fit([[0], [1]], [0, 1])
    params = model.get_params()
    before = model.to_dict()
    with pytest.raises(ValueError):
        model.set_params(n_estimators=4, max_depth=0)
    assert model.get_params() == params and model.to_dict() == before
    with pytest.raises(ValueError, match="Unknown"):
        model.set_params(criterion="entropy")
    model.set_params(n_estimators=4)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0]])


def test_export_retains_fit_time_parameters_after_direct_attribute_edits():
    model = RF(n_estimators=3, random_state=18).fit([[0], [1], [2]], [0, 1, 1])
    before = model.to_dict()
    model.n_estimators, model.random_state = 70, 99
    assert model.to_dict() == before
    assert model.get_params()["n_estimators"] == 70


def test_numpy_scalar_parameters_and_label_kinds_are_plain_json():
    model = RF(n_estimators=np.int64(3), max_features=np.float64(0.5),
               random_state=np.uint64(7), min_impurity_decrease=np.longdouble(0),
               bootstrap=np.bool_(True), n_jobs=np.int32(1)).fit([[0], [1]], np.array([0., 1.]))
    encoded = json.dumps(model.to_dict(), allow_nan=False)
    np.testing.assert_array_equal(RF.from_dict(json.loads(encoded)).predict([[0], [1]]), model.predict([[0], [1]]))


def test_fitted_forest_does_not_alias_training_inputs():
    X = np.arange(24.).reshape(12, 2)
    query = X.copy()
    y, weights = np.array([0, 1] * 6), np.ones(12)
    model = RF(n_estimators=5, oob_score=True).fit(X, y, sample_weight=weights)
    before = model.predict_proba(query)
    X[:] = -10
    y[:] = 5
    weights[:] = 0
    np.testing.assert_array_equal(model.predict_proba(query), before)


def test_single_class_empty_prediction_and_input_dimension_contracts():
    model = RF(n_estimators=3, oob_score=True).fit([[0, 1], [2, 3]], ["only", "only"])
    np.testing.assert_array_equal(model.predict_proba([[9, 2]]), [[1.]])
    assert model.predict(np.empty((0, 2))).shape == (0,)
    assert model.predict_proba(np.empty((0, 2))).shape == (0, 1)
    with pytest.raises(ValueError, match="features"):
        model.predict([[1]])
    with pytest.raises(ValueError):
        model.score(np.empty((0, 2)), [])


@pytest.mark.parametrize("field,value", [
    ("n_features_in", 10**12), ("training_n_samples", 10**12),
    ("n_estimators", 10**12),
])
def test_json_dimensions_are_bound_before_allocating_from_untrusted_metadata(field, value):
    state = RF(n_estimators=3, oob_score=True).fit([[0], [1], [2]], [0, 1, 1]).to_dict()
    if field == "n_estimators":
        state["params"][field] = value
    else:
        state[field] = value
    with pytest.raises(ValueError):
        RF.from_dict(state)


def test_json_rejects_boolean_tree_indices_and_forged_importances():
    state = RF(n_estimators=1, bootstrap=False, max_features=None).fit([[0], [1], [2], [3]], [0, 0, 1, 1]).to_dict()
    index_forgery = deepcopy(state)
    index_forgery["trees"][0]["children_left"][0] = True
    with pytest.raises(ValueError, match="integer"):
        RF.from_dict(index_forgery)
    state["feature_importances"][0] = 0.5
    with pytest.raises(ValueError, match="importances"):
        RF.from_dict(state)
