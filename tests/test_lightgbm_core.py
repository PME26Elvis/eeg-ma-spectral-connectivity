"""Behavioral, numerical, lifecycle, and hostile-JSON tests for the handwritten learner."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

import eeg_ma.lightgbm as core
from eeg_ma.lightgbm import HandcraftedLightGBMClassifier as LGB


@pytest.fixture
def data():
    rng = np.random.default_rng(1805)
    X = rng.normal(size=(54, 5))
    y = (X[:, 0] + 0.45 * X[:, 2] > 0).astype(np.int64)
    return X, y


@pytest.fixture
def fitted(data):
    X, y = data
    return LGB(n_estimators=5, num_leaves=5, min_data_in_leaf=2, random_state=71).fit(X, y)


@pytest.mark.parametrize("params", [
    {"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 1.2},
    {"num_leaves": 1}, {"num_leaves": False}, {"max_depth": 0},
    {"max_bin": 1}, {"max_bin": 65536}, {"max_bin": 31.0},
    {"min_data_in_bin": 0}, {"min_data_in_leaf": 0},
    {"learning_rate": 0}, {"learning_rate": "0.1"}, {"learning_rate": math.inf},
    {"learning_rate": 10**400}, {"learning_rate": True},
    {"max_delta_step": -1}, {"max_delta_step": math.inf},
    {"lambda_l1": -1}, {"lambda_l2": math.nan},
    {"min_sum_hessian_in_leaf": -1}, {"min_gain_to_split": -1},
    {"feature_fraction": 0}, {"feature_fraction": 1.01},
    {"bagging_fraction": 0}, {"top_rate": 0}, {"other_rate": 0},
    {"top_rate": 1}, {"top_rate": 0.9, "other_rate": 0.2},
    {"data_sample_strategy": "uniform"}, {"data_sample_strategy": "goss", "bagging_fraction": 0.8},
    {"enable_bundle": 1}, {"random_state": None}, {"random_state": -1},
    {"random_state": 2**64}, {"n_jobs": 0}, {"n_jobs": True}, {"n_jobs": -2},
    {"class_weight": "auto"}, {"class_weight": {}}, {"class_weight": {0: -1}},
    {"class_weight": {float("nan"): 1}}, {"class_weight": {"": 1}},
    {"early_stopping_rounds": 0}, {"early_stopping_min_delta": -1},
])
def test_constructor_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        LGB(**params)


@pytest.mark.parametrize("X,y,weights", [
    ([], [], None), ([[1], [2]], [0, 0], None), ([[1], [2], [3]], [0, 1, 2], None),
    ([[1], [2]], [0, "1"], None), ([[1], [2]], [0.0, 1], None),
    ([[1], [2]], [0.0, np.nan], None), ([[1], [2]], ["", "b"], None),
    ([[1], [2]], [0, 1], [0, 0]), ([[1], [2]], [0, 1], [1, 0]),
    ([[1], [2]], [0, 1], [1, -1]), ([[1], [2]], [0, 1], [True, True]),
    ([[1], [2]], [0, 1], [1, math.inf]), ([[1], [2]], [0, 1], [1]),
    ([[1j], [2j]], [0, 1], None), ([["1"], ["2"]], [0, 1], None),
    ([[1], [math.inf]], [0, 1], None), ([[1], [math.nan]], [0, 1], None),
    ([[1], [2**54 + 1]], [0, 1], None),
])
def test_fit_rejects_unsupported_data_and_weights(X, y, weights):
    with pytest.raises(ValueError):
        LGB(n_estimators=2).fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("labels", [
    np.array([False, True]), np.array([-8, 13], dtype=np.int64),
    np.array([2**63 + 7, 2**64 - 1], dtype=np.uint64),
    np.array([-0.5, 0.7]), np.array(["left", "right"]),
])
def test_label_types_roundtrip_without_loss(labels):
    X = np.arange(12.0)[:, None]
    y = labels[(X[:, 0] >= 6).astype(int)]
    model = LGB(n_estimators=3, min_data_in_leaf=1).fit(X, y)
    loaded = LGB.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(loaded.classes_, labels)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.score(X, y) == 1


def test_probability_score_and_loss_contract(fitted, data):
    X, y = data
    probabilities = fitted.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(fitted.predict(X), fitted.classes_[np.argmax(probabilities, axis=1)])
    assert np.all(np.diff(fitted.training_loss_) <= 1e-14)
    assert fitted.training_loss_[-1] < fitted.training_loss_[0]
    assert fitted.n_classes_ == 2
    assert fitted.n_samples_fit_ == len(X)
    assert fitted.n_estimators_ == fitted.best_iteration_ == 5
    assert fitted.validation_loss_ is None
    assert fitted.feature_importances_.sum() == pytest.approx(1)
    assert fitted.feature_split_counts_.sum() == sum(np.sum(tree.feature_ >= 0) for tree in fitted.estimators_)
    weights = np.arange(1, len(X) + 1)
    expected = np.average(fitted.predict(X) == y, weights=weights)
    assert fitted.score(X, y, sample_weight=weights) == pytest.approx(expected)
    assert fitted.predict(np.empty((0, X.shape[1]))).shape == (0,)
    assert fitted.predict_proba(np.empty((0, X.shape[1]))).shape == (0, 2)
    with pytest.raises(ValueError):
        fitted.predict(np.zeros((2, X.shape[1] + 1)))


def test_constant_and_zero_gain_features_produce_a_valid_base_only_model():
    for X in (np.ones((12, 3)), np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (3, 1))):
        y = np.tile([0, 1, 1, 0], 3)
        model = LGB(n_estimators=3, min_data_in_leaf=1, lambda_l2=0).fit(X, y)
        assert model.n_estimators_ == model.best_iteration_ == 0
        assert model.stopping_reason_ == "no_positive_split_gain"
        np.testing.assert_array_equal(model.predict_proba(X), np.full((len(X), 2), 0.5))
        np.testing.assert_array_equal(model.predict(X), np.zeros(len(X)))
        np.testing.assert_array_equal(model.feature_importances_, np.zeros(X.shape[1]))
        assert LGB.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_no_split_constraints_and_unbounded_newton_steps():
    X = np.arange(20.0)[:, None]
    y = (X[:, 0] > 9).astype(int)
    for params in ({"min_data_in_leaf": 11}, {"min_sum_hessian_in_leaf": 3},
                   {"min_gain_to_split": 100}, {"lambda_l1": 100}):
        model = LGB(n_estimators=3, **params).fit(X, y)
        assert model.n_estimators_ == 0
    uncapped = LGB(n_estimators=1, min_data_in_leaf=1, lambda_l2=0, max_delta_step=0).fit(X, y)
    capped = LGB(n_estimators=1, min_data_in_leaf=1, lambda_l2=0, max_delta_step=0.25).fit(X, y)
    assert max(abs(uncapped.estimators_[0].value_)) == 2
    assert max(abs(capped.estimators_[0].value_)) == 0.25
    assert uncapped.estimators_[0].gain_[0] > capped.estimators_[0].gain_[0]
    assert LGB.from_dict(uncapped.to_dict()).to_dict() == uncapped.to_dict()


def test_transactional_refit_and_parameter_invalidation(fitted, data):
    X, y = data
    snapshot = fitted.to_dict()
    with pytest.raises(ValueError):
        fitted.fit(X, np.zeros(len(y)))
    assert fitted.to_dict() == snapshot
    with pytest.raises(ValueError):
        fitted.set_params(learning_rate=-1)
    assert fitted.to_dict() == snapshot
    with pytest.raises(ValueError):
        fitted.set_params(unknown=True)
    assert fitted.to_dict() == snapshot
    fitted.learning_rate = 100  # configuration mutation cannot alter existing predictions/export
    assert fitted.to_dict() == snapshot
    fitted.set_params(learning_rate=0.1)
    with pytest.raises(ValueError, match="not fitted"):
        fitted.predict(X)
    assert fitted.get_params()["learning_rate"] == 0.1


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning")
def test_fitted_arrays_and_metadata_are_immutable_and_detached(fitted, data):
    X, _ = data
    before = fitted.predict_proba(X)
    arrays = [fitted.classes_, fitted.training_eligible_indices_, fitted.class_weight_factors_,
              fitted.training_class_mass_, fitted.iteration_seeds_, fitted.training_loss_,
              fitted.feature_importances_, fitted.feature_split_counts_, *fitted.feature_subsets_,
              fitted.estimators_[0].feature_, fitted.estimators_[0].value_,
              fitted.estimators_[0].sum_gradient_]
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
        array.shape = (array.size, 1)
    assert fitted.classes_.shape == (2,)
    assert fitted.training_loss_.shape == (6,)
    assert fitted.estimators_[0].feature_.ndim == 1
    snapshot = fitted.to_dict()
    altered = fitted.to_dict()
    altered["params"]["learning_rate"] = 5
    altered["iterations"][0]["sampling"]["sampled_indices"][0] = 10000
    record = fitted.sampling_records_[0]
    record["multipliers"][0] = 900
    assert fitted.to_dict() == snapshot
    np.testing.assert_array_equal(before, fitted.predict_proba(X))


def test_numpy_parameters_export_as_json_and_class_weight_is_copied(data):
    X, y = data
    weights = {0: np.float64(1.5), 1: np.float64(2.5)}
    model = LGB(n_estimators=np.int64(2), min_data_in_leaf=np.int64(2),
                enable_bundle=np.bool_(True), learning_rate=np.float64(0.1), class_weight=weights)
    weights[0] = 99
    assert model.get_params()["class_weight"][0] == 1.5
    params = model.get_params()
    params["class_weight"][1] = 100
    model.fit(X, y)
    assert model.class_weight_factors_.tolist() == [1.5, 2.5]
    serialized = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert LGB.from_dict(serialized).to_dict() == serialized


def test_balanced_factors_use_eligible_input_counts_and_compose_sample_weights():
    X = np.arange(8.0)[:, None]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    sample_weights = np.array([0, 1, 2, 3, 4, 1, 0, 5], dtype=float)
    model = LGB(n_estimators=2, min_data_in_leaf=1, class_weight="balanced").fit(X, y, sample_weights)
    factors = np.array([6 / (2 * 4), 6 / (2 * 2)])
    np.testing.assert_array_equal(model.class_weight_factors_, factors)
    np.testing.assert_array_equal(model.training_eligible_indices_, [1, 2, 3, 4, 5, 7])
    expected = np.array([sum(sample_weights[y == k] * factors[k]) for k in (0, 1)])
    np.testing.assert_array_equal(model.training_class_mass_, expected)
    assert model.base_score_ == math.log(expected[1]) - math.log(expected[0])
    assert LGB.from_dict(model.to_dict()).to_dict() == model.to_dict()


@pytest.mark.parametrize("mapping", [{0: 1}, {0: 1, 1: 1, 2: 1}, {0: 0, 1: 1},
                                      {False: 1, True: 1}, {0.0: 1, 1.0: 1}])
def test_class_weight_requires_exact_classes_and_positive_effective_mass(data, mapping):
    X, y = data
    with pytest.raises(ValueError):
        LGB(class_weight=mapping).fit(X, y)


def test_mixed_signed_unsigned_evaluation_labels_do_not_alias_through_float64():
    X = np.arange(12.0)[:, None]
    classes = np.array([2**63 + 7, 2**64 - 1], dtype=np.uint64)
    y = classes[(X[:, 0] >= 6).astype(int)]
    model = LGB(n_estimators=1, min_data_in_leaf=1).fit(X, y)
    unseen = np.full(len(X), 2**63 - 1, dtype=np.int64)
    with pytest.raises(ValueError, match="absent"):
        model.score(X, unseen)
    with pytest.raises(ValueError, match="absent"):
        model.fit(X, y, eval_set=(X, unseen))


@pytest.mark.parametrize("strategy", ["bagging", "goss"])
def test_prefix_and_parallel_histogram_determinism(data, strategy):
    X, y = data
    params = dict(min_data_in_leaf=2, num_leaves=4, feature_fraction=0.6,
                  data_sample_strategy=strategy, random_state=101,
                  bagging_fraction=0.8 if strategy == "bagging" else 1)
    short = LGB(n_estimators=3, n_jobs=1, **params).fit(X, y)
    long = LGB(n_estimators=6, n_jobs=2, **params).fit(X, y)
    assert short.n_estimators_ == 3
    assert short.to_dict()["iterations"] == long.to_dict()["iterations"][:3]
    assert short.training_loss_.tolist() == long.training_loss_[:4].tolist()
    np.testing.assert_array_equal(short.iteration_seeds_, core.iteration_seed_sequence(101, 6)[:3])
    assert all(len(features) == math.ceil(0.6 * X.shape[1]) for features in long.feature_subsets_)


def test_bagging_without_replacement_and_rounding(data):
    X, y = data
    weights = np.ones(len(X))
    weights[[2, 7, 31]] = 0
    model = LGB(n_estimators=3, min_data_in_leaf=1, bagging_fraction=0.51,
                feature_fraction=0.21).fit(X, y, weights)
    expected_count = math.ceil(0.51 * np.count_nonzero(weights))
    for record, features in zip(model.sampling_records_, model.feature_subsets_):
        rows = record["sampled_indices"]
        assert len(rows) == len(set(rows)) == expected_count
        assert rows == sorted(rows)
        assert not set(rows).intersection({2, 7, 31})
        assert record["multipliers"] == [1.0] * expected_count
        assert record["top_indices"] == []
        assert len(features) == 2


def test_early_stopping_can_select_the_initial_model(data):
    X, y = data
    model = LGB(n_estimators=20, min_data_in_leaf=2, learning_rate=0.3,
                early_stopping_rounds=2).fit(X, y, eval_set=(X, 1 - y))
    assert model.stopping_reason_ == "early_stopping"
    assert model.n_estimators_ == 2
    assert model.best_iteration_ == 0
    assert len(model.estimators_) == 2
    assert len(model.training_loss_) == len(model.validation_loss_) == 3
    np.testing.assert_array_equal(model.decision_function(X), np.full(len(X), model.base_score_))
    np.testing.assert_array_equal(model.feature_importances_, np.zeros(X.shape[1]))
    np.testing.assert_array_equal(model.feature_split_counts_, np.zeros(X.shape[1]))
    loaded = LGB.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_early_stopping_selects_prefix_and_min_delta_is_absolute(data):
    X, y = data
    # One first step improves enough; all later improvements remain below the
    # deliberately large minimum delta relative to the last accepted best.
    model = LGB(n_estimators=12, min_data_in_leaf=2, learning_rate=0.5,
                early_stopping_rounds=2, early_stopping_min_delta=0.15).fit(X, y, eval_set=(X, y))
    best, best_loss, stale = 0, model.validation_loss_[0], 0
    for index, loss in enumerate(model.validation_loss_[1:], 1):
        if loss < best_loss - 0.15:
            best, best_loss, stale = index, loss, 0
        else:
            stale += 1
    assert model.best_iteration_ == best
    assert model.stopping_reason_ == "early_stopping"
    assert stale == 2
    expected = np.full(len(X), model.base_score_)
    for tree in model.estimators_[:best]:
        expected += 0.5 * tree.predict(X)
    np.testing.assert_array_equal(model.decision_function(X), expected)
    assert LGB.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_validation_never_influences_bins_bundles_or_derivatives(data):
    X, y = data
    other_X = X * 1000 + 1e6
    params = dict(n_estimators=3, min_data_in_leaf=2, class_weight="balanced")
    train_only = LGB(**params).fit(X, y)
    with_valid = LGB(**params).fit(X, y, eval_set=(other_X, y), eval_sample_weight=np.arange(1, len(y) + 1))
    a, b = train_only.to_dict(), with_valid.to_dict()
    for field in ("bin_mapper", "bundle_layout", "iterations", "training_loss", "base_score"):
        assert a[field] == b[field]
    assert with_valid.best_iteration_ == with_valid.n_estimators_
    assert len(with_valid.validation_loss_) == 4
    raw = with_valid.decision_function(other_X)
    weights = np.arange(1, len(y) + 1) * with_valid.class_weight_factors_[y]
    expected = np.average(np.logaddexp(0, np.where(y == 1, -raw, raw)), weights=weights)
    assert with_valid.validation_loss_[-1] == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("fit_kwargs", [
    {"eval_sample_weight": [1, 1]},
    {"eval_set": []}, {"eval_set": [(np.zeros((2, 5)), [0, 1])]},
    {"eval_set": (np.zeros((2, 6)), [0, 1])},
    {"eval_set": (np.zeros((2, 5)), [0, 2])},
    {"eval_set": (np.zeros((2, 5)), ["0", "1"])},
    {"eval_set": (np.zeros((2, 5)), [0, 1]), "eval_sample_weight": [0, 0]},
])
def test_validation_input_errors_preserve_existing_fit(fitted, data, fit_kwargs):
    X, y = data
    before = fitted.to_dict()
    with pytest.raises(ValueError):
        fitted.fit(X, y, **fit_kwargs)
    assert fitted.to_dict() == before


def test_early_stopping_requires_explicit_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="eval_set"):
        LGB(early_stopping_rounds=2).fit(X, y)


def test_hessian_floor_is_before_weighting_and_sigmoid_is_stable():
    raw = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0])
    labels = np.array([0, 1, 1, 1, 0])
    weights = np.array([0.0, 2.0, 4.0, 8.0, 16.0])
    g, h = core.logistic_gradients(raw, labels, weights)
    assert np.isfinite(g).all() and np.isfinite(h).all()
    assert g[0] == h[0] == 0
    assert g[2] == -2 and h[2] == 1
    assert g[3] < 0  # unlike rounded sigmoid(40)-1, the gradient is retained
    assert g[3] == pytest.approx(-8 * math.exp(-40), rel=1e-15)
    assert h[[1, 3, 4]].tolist() == (weights[[1, 3, 4]] * core.HESSIAN_FLOOR).tolist()
    with pytest.raises(ValueError, match="dynamic range"):
        core.logistic_gradients(np.array([1000.0]), np.array([1]), np.array([1e-320]))


@pytest.mark.parametrize("same_bin", [False, True])
def test_histogram_cancellation_recovers_a_small_true_positive_gain(same_bin):
    X = np.array([0, 0, 1, 2, 3, 3] if same_bin else list(range(6)), dtype=float)[:, None]
    y = np.array([0, 0, 1, 1, 1, 0])
    weights = np.array([2e16, 2, 2e16, 2e16, 2, 2e16])
    model = LGB(n_estimators=1, min_data_in_leaf=3, lambda_l2=0,
                min_sum_hessian_in_leaf=0).fit(X, y, weights)
    assert model.n_estimators_ == 1
    tree = model.estimators_[0]
    assert tree.sum_gradient_.tolist() == [0, 1, -1]
    assert tree.gain_[0] == pytest.approx(2e-16, rel=1e-15)
    assert LGB.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_hessian_threshold_refinement_recovers_small_positive_weight_terms():
    n = 201
    X = np.repeat([0.0, 1.0], n)[:, None]
    y = np.repeat([0, 1], n)
    weights = np.tile(np.r_[4e16, np.full(n - 1, 4.0)], 2)
    model = LGB(n_estimators=1, min_data_in_leaf=n, lambda_l2=0,
                min_sum_hessian_in_leaf=1e16 + 200).fit(X, y, weights)
    assert model.n_estimators_ == 1
    assert model.estimators_[0].sum_hessian_[1:].tolist() == [1e16 + 200, 1e16 + 200]


def test_bin_rounding_does_not_defeat_global_exact_gain_ties():
    # Both features have the same sole legal row partition. Feature 0 groups
    # [1e16,1] into a bin before a negative bin, losing one rounded low bit;
    # feature 1 groups opposite large terms before the small positive term.
    # Comparing only approximate gains unfairly favors feature 1 by >64 eps.
    X = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [3, 3], [3, 2]], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 0])
    weights = np.array([2e16, 2.0, 2 * (1e16 - 1e13), 2e16, 2.0, 2 * (1e16 - 1e13)])
    model = LGB(n_estimators=1, min_data_in_leaf=3, lambda_l2=0,
                min_sum_hessian_in_leaf=0).fit(X, y, weights)
    assert model.estimators_[0].feature_[0] == 0
    assert model.estimators_[0].sum_gradient_[1:].tolist() == [1e13 + 1, -1e13 - 1]


def test_histogram_subtraction_is_used_and_safe_fallback_is_equivalent(monkeypatch, data):
    X, y = data
    original = core.subtract_histograms
    calls = []

    def record(parent, child):
        calls.append(1)
        return original(parent, child)

    monkeypatch.setattr(core, "subtract_histograms", record)
    expected = LGB(n_estimators=3, num_leaves=5, min_data_in_leaf=2).fit(X, y)
    assert len(calls) == sum(np.sum(tree.feature_ >= 0) for tree in expected.estimators_)

    def rebuild(parent, child):
        raise core.HistogramSubtractionError("deliberate numerical fallback test")

    monkeypatch.setattr(core, "subtract_histograms", rebuild)
    rebuilt = LGB(n_estimators=3, num_leaves=5, min_data_in_leaf=2).fit(X, y)
    np.testing.assert_array_equal(rebuilt.predict(X), expected.predict(X))
    np.testing.assert_allclose(rebuilt.predict_proba(X), expected.predict_proba(X), rtol=2e-14, atol=0)
    for left, right in zip(expected.estimators_, rebuilt.estimators_):
        np.testing.assert_array_equal(left.feature_, right.feature_)
        np.testing.assert_array_equal(left.threshold_, right.threshold_)


def test_weight_overflow_and_underflow_fail_transactionally(fitted, data):
    X, y = data
    before = fitted.to_dict()
    with pytest.raises(ValueError):
        fitted.fit(X, y, np.full(len(X), 1e308))
    assert fitted.to_dict() == before
    model = LGB(n_estimators=1, min_data_in_leaf=1, class_weight={0: 1e-300, 1: 1e-300})
    with pytest.raises(ValueError, match="underflow"):
        model.fit(X, y, np.full(len(X), 1e-300))


def _corrupt(payload, path, value):
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "old"), (("algorithm_version",), "other"),
    (("n_features_in",), True), (("n_features_in",), 10**12),
    (("training_n_samples",), "54"), (("training_n_samples",), 1),
    (("base_score",), True), (("base_score",), math.nan), (("base_score",), 20),
    (("classes", "values"), [0, True]), (("classes", "values"), [1, 0]),
    (("classes", "values"), [0, 0]), (("classes", "kind"), "mixed"),
    (("params", "learning_rate"), "0.1"), (("params", "enable_bundle"), 1),
    (("params", "n_estimators"), 1), (("params", "random_state"), 72),
    (("params", "lambda_l2"), 99), (("params", "min_gain_to_split"), 1e99),
    (("class_weight_factors",), [2, 1]), (("training_class_mass",), [0, 1]),
    (("training_eligible_indices",), [0, 0]), (("training_eligible_indices",), [1, 0]),
    (("training_eligible_indices",), [0, 1, 10000]),
    (("iterations", 0, "seed"), 0), (("iterations", 0, "seed"), True),
    (("iterations", 0, "feature_indices"), [0]),
    (("iterations", 0, "sampling", "strategy"), "goss"),
    (("iterations", 0, "sampling", "sampled_indices", 0), True),
    (("iterations", 0, "sampling", "multipliers", 0), 1.1),
    (("iterations", 0, "sampling", "top_indices"), [1]),
    (("iterations", 0, "tree", "children_left", 0), 0),
    (("iterations", 0, "tree", "children_right", 0), 1),
    (("iterations", 0, "tree", "depth", 1), 0),
    (("iterations", 0, "tree", "feature", 0), 10**12),
    (("iterations", 0, "tree", "threshold_bin", 0), 10**12),
    (("iterations", 0, "tree", "threshold", 0), 1000),
    (("iterations", 0, "tree", "sum_gradient", 1), 1e90),
    (("iterations", 0, "tree", "sum_hessian", 1), -1),
    (("iterations", 0, "tree", "n_node_samples", 0), 1),
    (("iterations", 0, "tree", "value", 0), 20),
    (("iterations", 0, "tree", "gain", 0), 0),
    (("iterations", 0, "tree", "gain", 0), math.inf),
    (("iterations", 0, "tree", "n_features_in"), 4),
    (("best_iteration",), 0), (("best_iteration",), 6),
    (("training_loss",), [0]), (("training_loss", 0), 4),
    (("training_loss", 2), -1), (("validation_loss",), []),
    (("stopping_reason",), "early_stopping"), (("stopping_reason",), "no_positive_split_gain"),
    (("feature_importances",), [0] * 5), (("feature_split_counts",), [0] * 5),
])
def test_json_rejects_typed_structural_and_objective_tampering(fitted, path, value):
    payload = fitted.to_dict()
    _corrupt(payload, path, value)
    with pytest.raises(ValueError):
        LGB.from_dict(payload)


def test_json_rejects_unknown_fields_at_every_layer(fitted):
    paths = [(), ("params",), ("classes",), ("iterations", 0),
             ("iterations", 0, "sampling"), ("iterations", 0, "tree")]
    for path in paths:
        payload = fitted.to_dict()
        target = payload
        for key in path:
            target = target[key]
        target["unexpected"] = 1
        with pytest.raises(ValueError):
            LGB.from_dict(payload)


def test_json_checks_goss_low_draw_and_correction(data):
    X, y = data
    model = LGB(n_estimators=2, min_data_in_leaf=1, data_sample_strategy="goss").fit(X, y)
    assert model.n_estimators_ == 2
    assert LGB.from_dict(model.to_dict()).to_dict() == model.to_dict()
    for key in ("sampled_indices", "top_indices", "multipliers"):
        payload = model.to_dict()
        record = payload["iterations"][0]["sampling"]
        if key == "multipliers":
            record[key][0] += 0.5
        else:
            record[key][0] = len(X) - 1
        with pytest.raises(ValueError):
            LGB.from_dict(payload)


def test_json_checks_early_stopping_curve_and_active_prefix(data):
    X, y = data
    model = LGB(n_estimators=10, min_data_in_leaf=2, early_stopping_rounds=2).fit(X, y, eval_set=(X, 1 - y))
    for path, value in [(("best_iteration",), 1), (("stopping_reason",), "n_estimators"),
                        (("validation_loss",), None), (("params", "early_stopping_rounds"), 1),
                        (("validation_loss", 1), 0.0)]:
        payload = model.to_dict()
        _corrupt(payload, path, value)
        with pytest.raises(ValueError):
            LGB.from_dict(payload)


def test_hostile_declared_size_is_rejected_without_large_array_allocation(fitted, monkeypatch):
    payload = fitted.to_dict()
    payload["n_features_in"] = 10**12
    original = np.zeros

    def bounded(shape, *args, **kwargs):
        assert np.prod(shape) < 10000
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", bounded)
    with pytest.raises(ValueError):
        LGB.from_dict(payload)
    # A huge configured future iteration count must not allocate that seed
    # schedule; a legitimate early-stopped prefix can still be loaded.
    payload = fitted.to_dict()
    payload["params"]["n_estimators"] = 10**12
    payload["stopping_reason"] = "no_positive_split_gain"
    loaded = LGB.from_dict(payload)
    assert loaded.n_estimators_ == fitted.n_estimators_


def test_prediction_uses_probability_ties_even_for_tiny_positive_logits(monkeypatch, fitted):
    monkeypatch.setattr(fitted, "decision_function", lambda X: np.array([1e-20, -1e-20, 1.0]))
    np.testing.assert_array_equal(fitted.predict(np.zeros((3, 5))), fitted.classes_[[0, 0, 1]])
