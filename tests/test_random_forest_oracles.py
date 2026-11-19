"""Independent mathematical and state-contract oracles for the handwritten RF.

These tests use direct enumeration, exact rational arithmetic, and retained
bootstrap memberships.  No ready-made decision tree or forest is an oracle.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from fractions import Fraction
import inspect
import json
import random
import sys
import warnings

import numpy as np
import pytest

from eeg_ma.random_forest import HandcraftedRandomForestClassifier
import eeg_ma.random_forest as forest_module


RF = HandcraftedRandomForestClassifier
TREE_ARRAYS = (
    "children_left_", "children_right_", "feature_", "threshold_",
    "n_node_samples_", "weighted_n_node_samples_", "class_counts_",
    "impurity_", "depth_",
)


def _cart(X, y, *, sample_weight=None, **params):
    options = dict(n_estimators=1, bootstrap=False, max_features=None,
                   random_state=731, n_jobs=1)
    options.update(params)
    return RF(**options).fit(X, y, sample_weight=sample_weight)


def _weighted_cost(y, weights):
    """W * Gini as exact Fractions; deliberately not cumulative scan code."""
    masses = {}
    for label, weight in zip(y, weights):
        mass = Fraction(int(weight))
        masses[label] = masses.get(label, Fraction()) + mass
    total = sum(masses.values(), Fraction())
    return total - sum((mass * mass for mass in masses.values()), Fraction()) / total


def _exhaustive_root(X, y, weights, min_leaf):
    """Evaluate every distinct split by fresh masks and exact impurity sums."""
    active = weights > 0
    X, y, weights = X[active], y[active], weights[active]
    candidates = []
    for feature in range(X.shape[1]):
        distinct = sorted(set(X[:, feature].tolist()))
        for lower, upper in zip(distinct[:-1], distinct[1:]):
            left = X[:, feature] <= lower
            if np.count_nonzero(left) < min_leaf or np.count_nonzero(~left) < min_leaf:
                continue
            cost = _weighted_cost(y[left], weights[left]) + _weighted_cost(y[~left], weights[~left])
            candidates.append((cost, feature, lower, upper, left))
    if not candidates:
        return None
    return min(candidates, key=lambda value: value[:3])


def _assert_tree_identical(first, second):
    for name in TREE_ARRAYS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_core_depends_only_on_numpy_and_python_standard_library():
    """The requested hand implementation cannot delegate to model packages."""
    source = inspect.getsource(forest_module)
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level, "The numerical core must not hide delegation in a sibling module"
            imported.append(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"numpy", "__future__"}
    assert set(imported) <= allowed
    assert not {"sklearn", "scipy", "xgboost", "lightgbm", "catboost"} & set(imported)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("min_leaf", [1, 2, 3])
def test_weighted_multiclass_root_matches_exact_exhaustive_oracle(seed, min_leaf):
    rng = np.random.default_rng(872 + seed)
    X = rng.integers(-2, 5, size=(12, 4)).astype(float)
    y = np.array(["amber", "blue", "copper"] * 4)
    weights = rng.integers(1, 7, size=12)
    # A zero-weight observation must not change a split or satisfy min_leaf.
    weights[seed % len(weights)] = 0
    expected = _exhaustive_root(X, y, weights, min_leaf)
    model = _cart(X, y, sample_weight=weights, max_depth=1, min_samples_leaf=min_leaf)
    tree = model.estimators_[0]
    active = weights > 0
    if expected is None:
        assert tree.node_count == 1
        return
    _, feature, lower, upper, left = expected
    assert tree.feature_[0] == feature
    assert lower <= tree.threshold_[0] < upper
    np.testing.assert_array_equal(X[active, feature] <= tree.threshold_[0], left)
    assert tree.n_node_samples_[0] == np.count_nonzero(active)
    root_counts = [int(weights[(y == label) & active].sum()) for label in model.classes_]
    np.testing.assert_allclose(tree.class_counts_[0], root_counts, rtol=0, atol=1e-13)
    expected_gini = float(_weighted_cost(y[active], weights[active]) / int(weights[active].sum()))
    assert tree.impurity_[0] == pytest.approx(expected_gini, abs=2e-15)


def test_exact_ties_choose_lowest_feature_then_lowest_split_boundary():
    X = np.column_stack([np.arange(4.0), np.arange(4.0), -np.arange(4.0)])
    y = np.array([0, 1, 1, 0])
    for seed in (0, 1, 2, 9, 731):
        tree = _cart(X, y, max_depth=1, random_state=seed).estimators_[0]
        assert tree.feature_[0] == 0
        assert 0.0 <= tree.threshold_[0] < 1.0


def test_zero_gain_first_split_can_learn_xor_without_special_cases():
    X = np.repeat(np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]), 4, axis=0)
    y = np.repeat(np.array([0, 1, 1, 0]), 4)
    model = _cart(X, y, max_depth=2, min_samples_leaf=2)
    tree = model.estimators_[0]
    assert tree.node_count == 7
    assert tree.max_depth == 2
    np.testing.assert_array_equal(model.predict(X), y)
    left, right = tree.children_left_[0], tree.children_right_[0]
    assert tree.impurity_[0] == pytest.approx(0.5)
    assert tree.impurity_[left] == pytest.approx(0.5)
    assert tree.impurity_[right] == pytest.approx(0.5)
    stopped = _cart(X, y, max_depth=2, min_impurity_decrease=1e-6)
    assert stopped.estimators_[0].node_count == 1


@pytest.mark.parametrize("lo, hi", [
    (1.0, np.nextafter(1.0, np.inf)),
    (np.nextafter(np.finfo(float).max, 0.0), np.finfo(float).max),
    (-np.finfo(float).max, np.nextafter(-np.finfo(float).max, 0.0)),
    (-np.finfo(float).max, np.finfo(float).max),
    (0.0, np.nextafter(0.0, 1.0)),
    (-np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
])
def test_adjacent_and_extreme_floats_keep_a_finite_correct_partition(lo, hi):
    X = np.array([[lo], [lo], [hi], [hi]])
    y = np.array([0, 0, 1, 1])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model = _cart(X, y, max_depth=1)
        threshold = model.estimators_[0].threshold_[0]
        assert np.isfinite(threshold)
        assert lo <= threshold < hi
        np.testing.assert_array_equal(model.predict(X), y)
        np.testing.assert_array_equal(model.predict_proba(X), [[1, 0], [1, 0], [0, 1], [0, 1]])


def test_constant_candidate_features_do_not_force_an_avoidable_dead_end():
    X = np.zeros((12, 12))
    X[6:, -1] = 1
    y = np.array([0] * 6 + [1] * 6)
    for seed in (0, 8, 29):
        model = _cart(X, y, max_features=1, random_state=seed)
        assert model.estimators_[0].feature_[0] == 11
        np.testing.assert_array_equal(model.predict(X), y)


def test_weighted_leaf_probabilities_keep_zero_mass_classes_and_literal_labels():
    X = np.zeros((5, 2))
    y = np.array(["001", "1", "1", "NA", "zero"])
    weights = np.array([1., 2., 3., 4., 0.])
    model = _cart(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, ["001", "1", "NA", "zero"])
    np.testing.assert_allclose(model.predict_proba([[0, 0]]), [[0.1, 0.5, 0.4, 0]], atol=1e-15)
    assert model.predict([[0, 0]]).tolist() == ["1"]
    assert model.estimators_[0].n_node_samples_[0] == 4
    assert model.estimators_[0].node_count == 1
    np.testing.assert_array_equal(model.feature_importances_, [0., 0.])


def test_class_weights_multiply_sample_weights_before_leaf_normalization():
    X = np.zeros((4, 1))
    y = np.array(["a", "a", "a", "b"])
    model = _cart(X, y, sample_weight=[1, 2, 3, 4], class_weight={"a": 3, "b": 0.25})
    np.testing.assert_allclose(model.predict_proba([[0]]), [[18 / 19, 1 / 19]], atol=1e-15)
    balanced = _cart(X, y, class_weight="balanced")
    np.testing.assert_allclose(balanced.predict_proba([[0]]), [[0.5, 0.5]], atol=1e-15)
    assert balanced.predict([[0]]).tolist() == ["a"]


def test_balanced_class_weights_count_positive_input_weights_only():
    X = np.zeros((5, 1))
    y = np.array(["a", "a", "a", "b", "c"])
    # Positive input rows: a twice, b once; c remains a declared zero-mass class.
    model = _cart(X, y, sample_weight=[1, 2, 0, 1, 0], class_weight="balanced")
    # a multiplier = 3/(2*2), b multiplier = 3/(2*1).
    np.testing.assert_allclose(model.predict_proba([[0]]), [[0.6, 0.4, 0]], atol=1e-15)


@pytest.mark.parametrize("scale", [1e-200, 1e-12, 1e12, 1e200])
def test_uniform_weight_rescaling_preserves_splits_probabilities_and_importance(scale):
    rng = np.random.default_rng(393)
    X = rng.normal(size=(25, 5))
    y = np.array([0, 1, 2, 1, 0] * 5)
    weights = np.array([1., 3., 7., 2., 4.] * 5)
    ordinary = _cart(X, y, sample_weight=weights, max_depth=4, min_samples_leaf=2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rescaled = _cart(X, y, sample_weight=weights * scale, max_depth=4, min_samples_leaf=2)
    first, second = ordinary.estimators_[0], rescaled.estimators_[0]
    for name in ("children_left_", "children_right_", "feature_", "threshold_", "n_node_samples_", "depth_"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_allclose(ordinary.predict_proba(X), rescaled.predict_proba(X), rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(ordinary.feature_importances_, rescaled.feature_importances_, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(second.weighted_n_node_samples_ / scale,
                               first.weighted_n_node_samples_, rtol=2e-13, atol=2e-13)


def test_seed_streams_are_schedule_independent_and_stable_when_forest_extends():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(37, 6))
    y = np.argmax(np.column_stack([X[:, 0], X[:, 1] + X[:, 2], -X[:, 3]]), axis=1)
    common = dict(max_depth=4, max_features=0.5, random_state=432, oob_score=True)
    serial = RF(n_estimators=13, n_jobs=1, **common).fit(X, y)
    threaded = RF(n_estimators=13, n_jobs=2, **common).fit(X, y)
    prefix = RF(n_estimators=5, n_jobs=1, **common).fit(X, y)
    np.testing.assert_array_equal(serial.tree_seeds_, threaded.tree_seeds_)
    np.testing.assert_array_equal(prefix.tree_seeds_, serial.tree_seeds_[:5])
    for i, (one, two) in enumerate(zip(serial.estimators_, threaded.estimators_)):
        _assert_tree_identical(one, two)
        np.testing.assert_array_equal(serial.bootstrap_indices_[i], threaded.bootstrap_indices_[i])
        if i < len(prefix.estimators_):
            _assert_tree_identical(one, prefix.estimators_[i])
            np.testing.assert_array_equal(serial.bootstrap_indices_[i], prefix.bootstrap_indices_[i])
    np.testing.assert_array_equal(serial.predict_proba(X), threaded.predict_proba(X))
    np.testing.assert_array_equal(serial.oob_decision_function_, threaded.oob_decision_function_)
    np.testing.assert_array_equal(serial.feature_importances_, threaded.feature_importances_)


def test_fit_does_not_consume_process_global_random_streams():
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    RF(n_estimators=5, random_state=97, n_jobs=2).fit(np.arange(24.).reshape(12, 2), [0, 1] * 6)
    now = np.random.get_state()
    assert numpy_state[0] == now[0] and numpy_state[2:] == now[2:]
    np.testing.assert_array_equal(numpy_state[1], now[1])
    assert random.getstate() == python_state


def test_public_arrays_cannot_change_predictions_or_fitted_tree_state():
    X = np.arange(60.).reshape(20, 3)
    y = np.array([0, 1, 2, 1, 0] * 4)
    model = RF(n_estimators=5, max_depth=3, random_state=51, oob_score=True).fit(X, y)
    before = model.predict_proba(X)
    arrays = [model.classes_, model.feature_importances_, model.tree_seeds_,
              model.oob_counts_, model.oob_decision_function_, model.oob_eligible_mask_]
    arrays.extend(model.bootstrap_indices_)
    arrays.extend(getattr(tree, key) for tree in model.estimators_ for key in TREE_ARRAYS)
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(model.predict_proba(X), before)


def test_public_array_shape_metadata_is_detached_from_the_fitted_state():
    X = np.arange(60.).reshape(20, 3)
    model = RF(n_estimators=5, max_depth=3, random_state=51, oob_score=True).fit(X, [0, 1, 2, 1, 0] * 4)
    prediction = model.predict_proba(X)
    serialized = json.dumps(model.to_dict(), sort_keys=True, allow_nan=False)
    getters = [lambda name=name: getattr(model, name) for name in
               ("classes_", "feature_importances_", "tree_seeds_", "oob_counts_",
                "oob_decision_function_", "oob_eligible_mask_")]
    getters += [lambda index=index: model.bootstrap_indices_[index] for index in range(len(model.estimators_))]
    getters += [lambda tree=tree, name=name: getattr(tree, name)
                for tree in model.estimators_ for name in TREE_ARRAYS]
    with warnings.catch_warnings():
        # New NumPy warns about this API, but deliberately attempting metadata
        # mutation is the behavior under test, not a recommended reshape API.
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message="Setting the shape on a NumPy array has been deprecated.*")
        for getter in getters:
            exposed = getter()
            shape = exposed.shape
            exposed.shape = (exposed.size, 1) if exposed.ndim == 1 else (exposed.size,)
            assert getter().shape == shape
    np.testing.assert_array_equal(model.predict_proba(X), prediction)
    assert json.dumps(model.to_dict(), sort_keys=True, allow_nan=False) == serialized


def test_json_roundtrip_is_transparent_finite_and_preserves_predictions():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(35, 4))
    y = np.where(X[:, 1] > 0, "positive", "negative")
    model = RF(n_estimators=11, max_depth=3, random_state=98, oob_score=True).fit(X, y)
    encoded = json.dumps(model.to_dict(), allow_nan=False, sort_keys=True)
    loaded = RF.from_dict(json.loads(encoded))
    query = np.concatenate([X, rng.normal(size=(17, 4))])
    np.testing.assert_array_equal(model.classes_, loaded.classes_)
    np.testing.assert_array_equal(model.predict(query), loaded.predict(query))
    np.testing.assert_array_equal(model.predict_proba(query), loaded.predict_proba(query))
    np.testing.assert_array_equal(model.feature_importances_, loaded.feature_importances_)
    assert json.dumps(loaded.to_dict(), allow_nan=False, sort_keys=True) == encoded


def _node_paths(tree, X):
    paths = []
    for row in X:
        node, path = 0, [0]
        while tree.children_left_[node] != -1:
            node = int(tree.children_left_[node] if row[tree.feature_[node]] <= tree.threshold_[node]
                       else tree.children_right_[node])
            path.append(node)
            assert len(path) <= tree.node_count, "A fitted tree cannot contain a cycle"
        paths.append(path)
    return paths


def test_bootstrap_counts_leaf_probabilities_oob_and_importance_from_original_rows():
    rng = np.random.default_rng(847)
    X = rng.normal(size=(28, 4))
    y = np.array(["amber", "blue", "copper", "blue"] * 7)
    sample_weights = np.array([0, 1, 2, 3, 1, 2, 1] * 4)
    class_weights = {"amber": 2, "blue": 1, "copper": 3}
    effective = sample_weights * np.array([class_weights[label] for label in y])
    model = RF(n_estimators=11, max_depth=4, min_samples_leaf=2, max_features=2,
               max_samples=0.55, random_state=37, oob_score=True,
               class_weight=class_weights).fit(X, y, sample_weight=sample_weights)
    active = effective > 0
    wanted_draws = int(np.ceil(0.55 * np.count_nonzero(active)))
    all_probabilities, tree_importances = [], []
    oob_sum = np.zeros((len(y), len(model.classes_)))
    oob_counts = np.zeros(len(y), dtype=int)
    for tree, draws in zip(model.estimators_, model.bootstrap_indices_):
        assert len(draws) == wanted_draws
        assert np.all(active[draws])
        paths = _node_paths(tree, X)
        counts = np.zeros((tree.node_count, len(model.classes_)), dtype=int)
        n_draws = np.zeros(tree.node_count, dtype=int)
        for index in draws:
            class_index = model.classes_.tolist().index(y[index])
            for node in paths[index]:
                n_draws[node] += 1
                counts[node, class_index] += int(effective[index])
        np.testing.assert_array_equal(tree.n_node_samples_, n_draws)
        np.testing.assert_allclose(tree.class_counts_, counts, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tree.weighted_n_node_samples_, counts.sum(axis=1), rtol=0, atol=1e-12)
        assert np.all(n_draws[tree.children_left_ == -1] >= 2)
        probabilities = np.array([counts[path[-1]] / counts[path[-1]].sum() for path in paths])
        np.testing.assert_allclose(tree.predict_proba(X), probabilities, atol=1e-15)
        all_probabilities.append(probabilities)
        absent = np.array([row not in set(draws.tolist()) for row in range(len(y))])
        oob_sum[absent] += probabilities[absent]
        oob_counts[absent] += 1

        # Independently calculate impurity reduction from source class masses.
        costs = []
        for row in counts:
            total = sum(map(int, row))
            costs.append(Fraction(total) - sum((Fraction(int(v) ** 2, total) for v in row), Fraction()))
        importance = np.zeros(X.shape[1])
        for node in range(tree.node_count):
            if tree.children_left_[node] != -1:
                left, right = tree.children_left_[node], tree.children_right_[node]
                decrease = costs[node] - costs[left] - costs[right]
                assert decrease >= 0
                importance[tree.feature_[node]] += float(decrease / int(counts[0].sum()))
        if importance.sum() > 0:
            importance /= importance.sum()
        tree_importances.append(importance)

    # Fixed-order averaging, not a majority of the individual hard decisions.
    expected_probabilities = sum(all_probabilities) / len(all_probabilities)
    np.testing.assert_allclose(model.predict_proba(X), expected_probabilities, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(model.predict(X), model.classes_[expected_probabilities.argmax(axis=1)])
    expected_importance = np.mean(tree_importances, axis=0)
    if expected_importance.sum() > 0:
        expected_importance /= expected_importance.sum()
    np.testing.assert_allclose(model.feature_importances_, expected_importance, atol=3e-15)
    expected_oob = np.full(oob_sum.shape, np.nan)
    covered = oob_counts > 0
    expected_oob[covered] = oob_sum[covered] / oob_counts[covered, None]
    np.testing.assert_array_equal(model.oob_counts_, oob_counts)
    np.testing.assert_array_equal(model.oob_eligible_mask_, active)
    np.testing.assert_allclose(model.oob_decision_function_, expected_oob, atol=2e-15, equal_nan=True)
    assert np.all(model.oob_counts_[~active] == model.n_estimators)
    eligible_covered = active & covered
    expected_score = np.mean(model.classes_[expected_oob[eligible_covered].argmax(axis=1)] == y[eligible_covered])
    assert model.oob_score_ == pytest.approx(expected_score)
    assert model.oob_coverage_ == pytest.approx(np.count_nonzero(eligible_covered) / np.count_nonzero(active))


def test_zero_oob_votes_remain_unknown_and_never_count_as_class_zero():
    X = np.arange(12.).reshape(-1, 1)
    y = np.array([0, 1] * 6)
    model = RF(n_estimators=1, max_samples=0.5, oob_score=True, random_state=911).fit(X, y)
    votes = model.oob_counts_
    assert np.any(votes == 0) and np.any(votes == 1)
    assert np.isnan(model.oob_decision_function_[votes == 0]).all()
    assert np.isfinite(model.oob_decision_function_[votes > 0]).all()
    assert model.oob_coverage_ == pytest.approx(np.mean(votes > 0))
    alone = RF(n_estimators=3, oob_score=True, random_state=0).fit([[7.0]], ["only"])
    assert np.isnan(alone.oob_score_)
    assert alone.oob_coverage_ == 0
    np.testing.assert_array_equal(alone.oob_counts_, [0])
    assert np.isnan(alone.oob_decision_function_).all()
    # JSON must encode the unknown diagnostic explicitly, never nonstandard NaN.
    restored = RF.from_dict(json.loads(json.dumps(alone.to_dict(), allow_nan=False)))
    assert np.isnan(restored.oob_score_)
    assert np.isnan(restored.oob_decision_function_).all()


def test_bootstrap_leaf_sample_counts_include_repeated_draws():
    X = np.array([[0.0], [1.0]])
    model = RF(n_estimators=13, min_samples_leaf=2, max_samples=2,
               random_state=14).fit(X, ["left", "right"])
    assert any(len(set(indices.tolist())) < len(indices) for indices in model.bootstrap_indices_)
    for tree, draws in zip(model.estimators_, model.bootstrap_indices_):
        assert tree.n_node_samples_[0] == 2
        # With only two positive draws, min_leaf=2 forbids every binary split.
        assert tree.node_count == 1
        expected = np.bincount(draws, minlength=2) / 2
        np.testing.assert_allclose(tree.predict_proba([[0.5]])[0], expected, atol=0)


def test_soft_probability_voting_can_disagree_with_hard_majority():
    rng = np.random.default_rng(301)
    X = rng.normal(size=(18, 2))
    y = (X[:, 0] + 0.2 * rng.normal(size=18) > 0).astype(int)
    query = rng.normal(size=(25, 2))[2:3]
    model = RF(n_estimators=3, max_depth=1, max_features=1, random_state=0).fit(X, y)
    tree_probabilities = np.array([tree.predict_proba(query)[0] for tree in model.estimators_])
    # Two uncertain trees vote class 1; one pure leaf votes class 0.
    expected_trees = np.array([[5 / 13, 8 / 13], [1 / 4, 3 / 4], [1, 0]])
    np.testing.assert_allclose(tree_probabilities, expected_trees, atol=1e-15)
    np.testing.assert_array_equal(tree_probabilities.argmax(axis=1), [1, 1, 0])
    expected = np.array([[85 / 156, 71 / 156]])
    np.testing.assert_allclose(model.predict_proba(query), expected, atol=1e-15)
    assert model.predict(query).tolist() == [0]


@pytest.mark.parametrize("labels", [[1, "1"], [0, "0"], [True, "True"]])
def test_mixed_string_and_numeric_labels_cannot_silently_merge(labels):
    with pytest.raises(ValueError):
        _cart([[0], [1]], labels)


def test_extended_precision_labels_are_rejected_or_losslessly_serializable():
    # Accepting a dtype obliges the public JSON exporter to support it.  An
    # explicit dtype rejection is also a valid, documented contract boundary.
    labels = np.array([0, 1], dtype=np.longdouble)
    try:
        model = _cart([[0], [1]], labels)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("label", "float", "precision", "dtype"))
    else:
        loaded = RF.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
        np.testing.assert_array_equal(loaded.predict([[0], [1]]), labels)


@pytest.fixture
def serialized_forest():
    rng = np.random.default_rng(726)
    X = rng.normal(size=(24, 3))
    y = np.array([0, 1, 2] * 8)
    model = RF(n_estimators=5, max_depth=3, random_state=712, oob_score=True).fit(X, y)
    value = model.to_dict()
    assert any(max(tree["depth"]) > 1 for tree in value["trees"])
    return value


def _corrupt_model(value, case):
    tree = value["trees"][0]
    if case == "schema":
        value["schema_version"] += "-unrecognized"
    elif case == "algorithm":
        value["algorithm_version"] += "-unrecognized"
    elif case == "missing_tree_field":
        del tree["depth"]
    elif case == "extra_tree_field":
        tree["execute_on_load"] = "not executable content"
    elif case == "boolean_dimensions":
        value["n_features_in"] = True
    elif case == "unsorted_classes":
        value["classes"]["values"].reverse()
    elif case == "duplicate_classes":
        value["classes"]["values"][1] = value["classes"]["values"][0]
    elif case == "cycle":
        tree["children_left"][0] = 0
    elif case == "shared_child":
        tree["children_right"][0] = tree["children_left"][0]
    elif case == "unknown_feature":
        tree["feature"][0] = value["n_features_in"]
    elif case == "nan_threshold":
        tree["threshold"][0] = float("nan")
    elif case == "infinite_threshold":
        tree["threshold"][0] = float("inf")
    elif case == "negative_class_mass":
        tree["class_counts"][0][0] = -1
    elif case == "coherent_parent_mass_but_inconsistent_children":
        # This is numerically coherent for the parent considered alone.
        tree["class_counts"][0][0] += 1
        total = sum(tree["class_counts"][0])
        tree["weighted_n_node_samples"][0] = total
        tree["impurity"][0] = 1 - sum((x / total) ** 2 for x in tree["class_counts"][0])
    elif case == "wrong_gini":
        tree["impurity"][0] += 0.01
    elif case == "wrong_child_depth":
        tree["depth"][1] += 1
    elif case == "fractional_child_index":
        tree["children_left"][0] += 0.5
    elif case == "boolean_child_index":
        assert tree["children_left"][0] == 1
        tree["children_left"][0] = True
    elif case == "boolean_leaf_threshold":
        leaf = next(i for i, child in enumerate(tree["children_left"]) if child == -1)
        tree["threshold"][leaf] = False
    elif case == "configured_depth":
        value["params"]["max_depth"] = 1
    elif case == "configured_min_leaf":
        value["params"]["min_samples_leaf"] = 100
    elif case == "configured_min_split":
        value["params"]["min_samples_split"] = 100
    elif case == "configured_min_gain":
        value["params"]["min_impurity_decrease"] = 1.0
    elif case == "seed_identity":
        value["tree_seeds"][0] ^= 1
    elif case == "bootstrap_membership":
        value["bootstrap_indices"][0][0] = (value["bootstrap_indices"][0][0] + 1) % value["training_n_samples"]
    elif case == "eligible_membership":
        value["training_eligible_indices"].pop()
    elif case == "oob_membership":
        value["oob"]["counts"][0] = (value["oob"]["counts"][0] + 1) % (value["params"]["n_estimators"] + 1)
    elif case == "oob_probability_sum":
        row = next(i for i, count in enumerate(value["oob"]["counts"]) if count > 0)
        value["oob"]["decision_function"][row] = [0.0] * len(value["classes"]["values"])
    elif case == "oob_coverage":
        value["oob"]["coverage"] = 0.0
    elif case == "oob_score_missing":
        value["oob"]["score"] = None
    else:
        raise AssertionError(f"Unknown corruption case: {case}")


@pytest.mark.parametrize("case", [
    "schema", "algorithm", "missing_tree_field", "extra_tree_field", "boolean_dimensions",
    "unsorted_classes", "duplicate_classes", "cycle", "shared_child", "unknown_feature",
    "nan_threshold", "infinite_threshold", "negative_class_mass",
    "coherent_parent_mass_but_inconsistent_children", "wrong_gini", "wrong_child_depth",
    "fractional_child_index", "boolean_child_index", "boolean_leaf_threshold",
    "configured_depth", "configured_min_leaf", "configured_min_split", "configured_min_gain",
    "seed_identity", "bootstrap_membership", "eligible_membership", "oob_membership",
    "oob_probability_sum", "oob_coverage", "oob_score_missing",
])
def test_serialized_models_reject_structural_numeric_and_rng_contradictions(serialized_forest, case):
    broken = deepcopy(serialized_forest)
    _corrupt_model(broken, case)
    with pytest.raises(ValueError):
        RF.from_dict(broken)


def test_serialized_dictionary_mutation_does_not_mutate_fitted_state(serialized_forest):
    model = RF.from_dict(serialized_forest)
    X = np.array([[0., 0., 0.], [1., 2., 3.], [-1., 2., 0.]])
    before = model.predict_proba(X)
    exported = model.to_dict()
    exported["trees"][0]["feature"][0] = -100
    exported["classes"]["values"].clear()
    exported["params"]["n_estimators"] = 0
    exported["oob"]["decision_function"][0].clear()
    np.testing.assert_array_equal(model.predict_proba(X), before)
    assert model.to_dict() == serialized_forest
