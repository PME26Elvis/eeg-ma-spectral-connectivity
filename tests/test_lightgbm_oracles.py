"""Independent exact-arithmetic and fresh-mask boosting oracles.

No ready-made tree/boosting learner is used as an expected answer. The first
Newton tree is solved with Fraction arithmetic and exhaustive raw-value masks,
rather than using the implementation's histograms, split scorer, or routing.
"""
from __future__ import annotations

import ast
from fractions import Fraction as F
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eeg_ma.lightgbm import HandcraftedLightGBMClassifier


def leaf_objective(y, weights, rows, l1, l2, cap):
    """Balanced weighted prior => exact p=1/2, g=w(1/2-y), h=w/4."""
    gradient = sum((F(int(weights[i])) * (F(1, 2) - int(y[i])) for i in rows), F())
    hessian = sum((F(int(weights[i]), 4) for i in rows), F())
    magnitude = max(abs(gradient) - l1, F())
    value = -(1 if gradient > 0 else -1 if gradient < 0 else 0) * magnitude / (hessian + l2)
    value = min(max(value, -cap), cap)
    score = -2 * gradient * value - (hessian + l2) * value * value - 2 * l1 * abs(value)
    return gradient, hessian, value, score


def split_oracle(X, y, weights, rows, l1, l2, cap, min_leaf=1):
    parent_score = leaf_objective(y, weights, rows, l1, l2, cap)[3]
    candidates = []
    for feature in range(X.shape[1]):
        # One training-only mapper is fixed for the whole tree. Enumerate
        # complete-X raw cuts directly, including bins empty in this node.
        unique = sorted(set(X[:, feature]))
        for lo, hi in zip(unique[:-1], unique[1:]):
            threshold = float(lo / 2 + hi / 2)
            left = rows[X[rows, feature] <= threshold]
            right = rows[X[rows, feature] > threshold]
            if min(len(left), len(right)) < min_leaf:
                continue
            gain = (leaf_objective(y, weights, left, l1, l2, cap)[3]
                    + leaf_objective(y, weights, right, l1, l2, cap)[3] - parent_score)
            if gain > 0:
                candidates.append((gain, feature, threshold, left, right))
    if not candidates:
        return None
    return min(candidates, key=lambda row: (-row[0], row[1], row[2]))


def tree_oracle(X, y, weights, *, leaves, depth, l1, l2, cap, min_leaf):
    nodes = [{"rows": np.arange(len(y)), "depth": 0, "split": None}]
    open_leaves = {0}
    while len(open_leaves) < leaves:
        candidates = []
        for node_id in open_leaves:
            node = nodes[node_id]
            if depth is not None and node["depth"] >= depth:
                continue
            split = split_oracle(X, y, weights, node["rows"], l1, l2, cap, min_leaf)
            if split is not None:
                candidates.append((node_id, split))
        if not candidates:
            break
        node_id, split = min(candidates, key=lambda item: (-item[1][0], item[1][1], item[1][2], item[0]))
        node = nodes[node_id]
        children = (len(nodes), len(nodes) + 1)
        node.update(split=split, children=children)
        open_leaves.remove(node_id)
        open_leaves.update(children)
        nodes.extend({"rows": rows, "depth": node["depth"] + 1, "split": None}
                     for rows in split[3:])
    return nodes


@pytest.mark.parametrize("seed", range(9))
@pytest.mark.parametrize("l1,l2,cap", [(F(0), F(0), F(10)), (F(1, 2), F(1), F(10)),
                                       (F(0), F(4), F(1, 5)), (F(2), F(1), F(1))])
def test_first_stump_matches_exhaustive_fraction_objective(seed, l1, l2, cap):
    rng = np.random.default_rng(seed)
    X = rng.integers(-5, 6, size=(12, 4)).astype(float)
    y = rng.permutation([0] * 6 + [1] * 6)
    weights = np.ones(12, dtype=int)
    expected = split_oracle(X, y, weights, np.arange(12), l1, l2, cap)
    model = HandcraftedLightGBMClassifier(n_estimators=1, num_leaves=2, max_bin=31,
        min_data_in_leaf=1, lambda_l1=float(l1), lambda_l2=float(l2),
        max_delta_step=float(cap), random_state=seed).fit(X, y)
    payload = model.to_dict()
    assert payload["base_score"] == 0.0
    if expected is None:
        assert payload["iterations"] == []
        np.testing.assert_array_equal(model.predict_proba(X), np.full((12, 2), 0.5))
        return
    tree = payload["iterations"][0]["tree"]
    gain, feature, threshold, left, right = expected
    assert len(tree["feature"]) == 3
    assert tree["feature"][0] == feature
    assert tree["threshold"][0] == threshold
    assert tree["gain"][0] == pytest.approx(float(gain), rel=1e-13)
    for node, rows in enumerate((np.arange(12), left, right)):
        g, h, value, _ = leaf_objective(y, weights, rows, l1, l2, cap)
        assert tree["n_node_samples"][node] == len(rows)
        assert tree["sum_gradient"][node] == pytest.approx(float(g), abs=1e-14)
        assert tree["sum_hessian"][node] == pytest.approx(float(h), rel=1e-14)
        assert tree["value"][node] == pytest.approx(float(value), abs=1e-14)
    expected_raw = np.empty(12)
    expected_raw[left] = float(leaf_objective(y, weights, left, l1, l2, cap)[2]) * 0.05
    expected_raw[right] = float(leaf_objective(y, weights, right, l1, l2, cap)[2]) * 0.05
    np.testing.assert_allclose(model.decision_function(X), expected_raw, rtol=1e-13, atol=1e-15)
    p = 1 / (1 + np.exp(-expected_raw))
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], p, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("leaves,depth", [(3, None), (5, None), (5, 2)])
def test_leaf_wise_growth_matches_global_best_first_fraction_search(seed, leaves, depth):
    rng = np.random.default_rng(100 + seed)
    X = rng.integers(-4, 5, size=(20, 3)).astype(float)
    y = rng.permutation([0] * 10 + [1] * 10)
    weights = np.ones(20, dtype=int)
    l1, l2, cap = F(1, 4), F(1), F(3, 4)
    nodes = tree_oracle(X, y, weights, leaves=leaves, depth=depth, l1=l1, l2=l2,
                       cap=cap, min_leaf=2)
    model = HandcraftedLightGBMClassifier(n_estimators=1, num_leaves=leaves, max_depth=depth,
        max_bin=31, min_data_in_leaf=2, lambda_l1=float(l1), lambda_l2=float(l2),
        max_delta_step=float(cap), random_state=seed).fit(X, y)
    if len(nodes) == 1:
        assert model.to_dict()["iterations"] == []
        return
    actual = model.to_dict()["iterations"][0]["tree"]
    assert len(actual["feature"]) == len(nodes)
    raw = np.empty(len(y))
    for i, node in enumerate(nodes):
        g, h, value, _ = leaf_objective(y, weights, node["rows"], l1, l2, cap)
        assert actual["n_node_samples"][i] == len(node["rows"])
        assert actual["sum_gradient"][i] == pytest.approx(float(g), abs=1e-14)
        assert actual["sum_hessian"][i] == pytest.approx(float(h), rel=1e-14)
        assert actual["value"][i] == pytest.approx(float(value), abs=1e-14)
        assert actual["depth"][i] == node["depth"]
        if node["split"] is None:
            assert actual["children_left"][i] == actual["children_right"][i] == -1
            raw[node["rows"]] = float(value) * 0.05
        else:
            gain, feature, threshold, _, _ = node["split"]
            assert (actual["children_left"][i], actual["children_right"][i]) == node["children"]
            assert actual["feature"][i] == feature
            assert actual["threshold"][i] == threshold
            assert actual["gain"][i] == pytest.approx(float(gain), rel=2e-13)
    np.testing.assert_allclose(model.decision_function(X), raw, rtol=1e-13, atol=1e-15)


def test_equal_gain_across_leaves_uses_feature_boundary_then_leaf_priority():
    # Independent Fraction enumeration gives root (feature 3, cut .5), then
    # equal gain 2/5 in both leaves. The right leaf's feature 0 outranks the
    # earlier-created left leaf's feature 1.
    rng = np.random.default_rng(47)
    X = rng.integers(0, 3, size=(12, 4)).astype(float)
    y = rng.permutation([0] * 6 + [1] * 6)
    nodes = tree_oracle(X, y, np.ones(12, dtype=int), leaves=3, depth=None,
                       l1=F(0), l2=F(1), cap=F(10), min_leaf=1)
    assert nodes[0]["split"][:3] == (F(16, 5), 3, .5)
    assert nodes[1]["split"] is None
    assert nodes[2]["split"][:3] == (F(2, 5), 0, .5)
    model = HandcraftedLightGBMClassifier(n_estimators=1, num_leaves=3,
        max_bin=31, min_data_in_leaf=1, lambda_l2=1.).fit(X, y)
    tree = model.to_dict()["iterations"][0]["tree"]
    assert tree["feature"] == [3, -1, 0, -1, -1]
    assert tree["gain"][2] == pytest.approx(0.4, rel=1e-14)


@pytest.mark.parametrize("seed", range(5))
def test_nonuniform_sample_weights_match_rational_first_tree(seed):
    rng = np.random.default_rng(202 + seed)
    X = rng.integers(-5, 6, size=(12, 3)).astype(float)
    y = np.array([0] * 6 + [1] * 6)
    weights = np.r_[np.arange(1, 7), rng.permutation(np.arange(1, 7))]
    l1, l2, cap = F(1, 2), F(3, 2), F(2, 3)
    expected = split_oracle(X, y, weights, np.arange(12), l1, l2, cap)
    assert expected is not None
    model = HandcraftedLightGBMClassifier(n_estimators=1, num_leaves=2,
        max_bin=31, min_data_in_leaf=1, lambda_l1=float(l1), lambda_l2=float(l2),
        max_delta_step=float(cap)).fit(X, y, sample_weight=weights)
    tree = model.to_dict()["iterations"][0]["tree"]
    assert tree["feature"][0] == expected[1]
    assert tree["threshold"][0] == expected[2]
    assert tree["gain"][0] == pytest.approx(float(expected[0]), rel=1e-13)
    for node, rows in enumerate((np.arange(12), expected[3], expected[4])):
        g, h, v, _ = leaf_objective(y, weights, rows, l1, l2, cap)
        assert tree["sum_gradient"][node] == pytest.approx(float(g), abs=1e-14)
        assert tree["sum_hessian"][node] == pytest.approx(float(h), rel=1e-14)
        assert tree["value"][node] == pytest.approx(float(v), abs=1e-14)


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1., 1e100, 1e200])
def test_mass_and_regularization_scaling_preserve_predictions(scale):
    X = np.column_stack((np.arange(12), [2, 5, 1, 4, 3, 0, 9, 8, 10, 7, 6, 11])).astype(float)
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    params = dict(n_estimators=4, num_leaves=4, min_data_in_leaf=1,
                  min_sum_hessian_in_leaf=0., max_bin=31, random_state=111)
    reference = HandcraftedLightGBMClassifier(**params, lambda_l1=.25, lambda_l2=1.).fit(X, y)
    model = HandcraftedLightGBMClassifier(**params, lambda_l1=.25 * scale,
        lambda_l2=scale).fit(X, y, sample_weight=np.full(12, scale))
    np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X),
                               rtol=2e-12, atol=1e-14)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    snapshot = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = HandcraftedLightGBMClassifier.from_dict(snapshot)
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def _raw_tree_predictions(tree, X):
    result = np.empty(len(X))
    for row, values in enumerate(X):
        node = 0
        while tree["children_left"][node] >= 0:
            node = (tree["children_left"][node] if values[tree["feature"][node]] <= tree["threshold"][node]
                    else tree["children_right"][node])
        result[row] = tree["value"][node]
    return result


def test_each_goss_round_uses_prefix_weighted_gradients_and_finite_count_correction():
    rng = np.random.default_rng(414)
    X = rng.normal(size=(61, 7))
    y = (X[:, 0] + 0.8 * X[:, 2] > 0).astype(int)
    weights = rng.uniform(0.2, 3, len(y))
    weights[[3, 14, 21]] = 0
    eligible = np.flatnonzero(weights > 0)
    master, rounds, fraction = 193, 8, 0.6
    model = HandcraftedLightGBMClassifier(n_estimators=rounds, learning_rate=0.2,
        num_leaves=5, min_data_in_leaf=2, max_bin=31, data_sample_strategy="goss",
        top_rate=0.21, other_rate=0.17, feature_fraction=fraction,
        random_state=master).fit(X, y, sample_weight=weights)
    payload = model.to_dict()
    assert len(payload["iterations"]) >= 3
    prior = weights[y == 1].sum() / weights.sum()
    raw = np.full(len(y), math.log(prior / (1 - prior)))
    seeds = np.random.SeedSequence(master).spawn(rounds)
    top_count = min(len(eligible) - 1, math.ceil(0.21 * len(eligible)))
    low_count = min(len(eligible) - top_count, math.ceil(0.17 * len(eligible)))
    correction = (len(eligible) - top_count) / low_count
    for index, record in enumerate(payload["iterations"]):
        seed = int(seeds[index].generate_state(1, dtype=np.uint64)[0])
        assert record["seed"] == seed
        streams = np.random.SeedSequence(seed).spawn(2)
        p = 1 / (1 + np.exp(-raw))
        gradient = weights * (p - y)
        ranked = np.asarray(sorted(eligible, key=lambda i: (-abs(gradient[i]), i)))
        top, pool = ranked[:top_count], np.sort(ranked[top_count:])
        expected_low = np.random.default_rng(streams[0]).choice(pool, size=low_count, replace=False)
        sampling = record["sampling"]
        assert sampling["strategy"] == "goss"
        assert set(sampling["top_indices"]) == set(top)
        rows = np.asarray(sampling["sampled_indices"])
        multipliers = np.asarray(sampling["multipliers"])
        assert len(rows) == len(set(rows)) == top_count + low_count
        assert set(rows) == set(top) | set(expected_low)
        np.testing.assert_array_equal(multipliers, [1.0 if i in top else correction for i in rows])
        # The feature stream is independent of the row-selection stream.
        expected_features = np.sort(np.random.default_rng(streams[1]).choice(X.shape[1],
            size=max(1, math.ceil(fraction * X.shape[1])), replace=False))
        np.testing.assert_array_equal(record["feature_indices"], expected_features)
        tree = record["tree"]
        assert tree["n_node_samples"][0] == len(rows)
        assert tree["sum_gradient"][0] == pytest.approx(float(np.sum(gradient[rows] * multipliers)), abs=1e-12)
        hessian = weights * np.maximum(p * (1 - p), 1e-12)
        assert tree["sum_hessian"][0] == pytest.approx(float(np.sum(hessian[rows] * multipliers)), rel=1e-12)
        raw += 0.2 * _raw_tree_predictions(tree, X)
        loss = np.sum(weights * (np.logaddexp(0, raw) - y * raw)) / weights.sum()
        assert payload["training_loss"][index + 1] == pytest.approx(float(loss), rel=1e-12)
    np.testing.assert_allclose(model.decision_function(X), raw, rtol=1e-12, atol=1e-12)


def test_prior_only_model_is_valid_for_zero_gain_xor_and_preserves_label_order():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array(["rest", "task", "task", "rest"])
    model = HandcraftedLightGBMClassifier(n_estimators=5, num_leaves=4,
        min_data_in_leaf=1, lambda_l2=0.0).fit(X, y)
    assert model.to_dict()["iterations"] == []
    np.testing.assert_array_equal(model.classes_, ["rest", "task"])
    np.testing.assert_array_equal(model.predict(X), ["rest"] * 4)
    np.testing.assert_array_equal(model.predict_proba(X), np.full((4, 2), 0.5))
    restored = HandcraftedLightGBMClassifier.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    assert restored.to_dict() == model.to_dict()


def test_zero_effective_rows_do_not_change_training_bins_bundles_or_predictions():
    X = np.zeros((24, 4))
    X[np.arange(24), np.arange(24) % 4] = np.tile([-3., -1., 2., 4., 6., 8.], 4)
    y = (np.arange(24) % 4 < 2).astype(int)
    params = dict(n_estimators=6, num_leaves=3, max_bin=7, min_data_in_leaf=1,
                  lambda_l2=0.5, random_state=82)
    original = HandcraftedLightGBMClassifier(**params).fit(X, y)
    # This excluded row would violate exclusivity and radically alter quantiles
    # if fit accidentally used all rows for bins or attempted to pack it.
    augmented = np.vstack((X, [1e100, -1e100, 1e100, -1e100]))
    extended = HandcraftedLightGBMClassifier(**params).fit(augmented, np.r_[y, 1],
        sample_weight=np.r_[np.ones(len(y)), 0])
    left, right = original.to_dict(), extended.to_dict()
    assert left["bin_mapper"] == right["bin_mapper"]
    assert left["bundle_layout"] == right["bundle_layout"]
    assert left["iterations"] == right["iterations"]
    np.testing.assert_array_equal(original.decision_function(augmented), extended.decision_function(augmented))


def test_handwritten_numerical_modules_import_only_stdlib_numpy_and_own_histograms():
    root = Path(__file__).resolve().parents[1] / "src/eeg_ma"
    import sys
    allowed = set(sys.stdlib_module_names) | {"numpy", "lightgbm_histogram"}
    for path in (root / "lightgbm.py", root / "lightgbm_histogram.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        assert set(imports) <= allowed, (path.name, set(imports) - allowed)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert not any(isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}
                       for node in calls)
