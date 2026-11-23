"""Bounded independent exhaustive mathematical oracles for handwritten trees.

These do not call the implementation's split search, histogram builder,
Newton-score helpers, or tree-growing routines. They verify a deliberately
narrow class of tiny fitted forests/boosters, not arbitrary production models.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np


class OracleNotApplicable(ValueError):
    """A requested independent proof lies outside the declared tiny scope."""


class OracleMismatch(AssertionError):
    """An observed tree contradicts an independently enumerated candidate."""


def _matrix(X, y, max_rows, max_features):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if (X.ndim != 2 or y.ndim != 1 or len(X) != len(y)
            or not 2 <= len(X) <= max_rows
            or not 1 <= X.shape[1] <= max_features
            or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all()):
        raise OracleNotApplicable("Tiny oracle requires bounded finite binary observations")
    return X, y


def _assert_close(actual, expected, name, *, rtol=1e-10, atol=1e-12):
    if not (math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)):
        raise OracleMismatch(f"Independent oracle disagrees on {name}: {actual} != {expected}")


def _rational_gini(y, weights, rows):
    mass = sum((Fraction(str(float(weights[i]))) for i in rows), Fraction(0))
    if mass <= 0:
        raise OracleNotApplicable("Tiny CART oracle needs positive weighted mass")
    by_class = [
        sum((Fraction(str(float(weights[i]))) for i in rows if int(y[i]) == c), Fraction(0))
        for c in (0, 1)
    ]
    return 1 - sum((value / mass) ** 2 for value in by_class)


def _cart_best(X, y, weights, rows, minimum):
    parent = _rational_gini(y, weights, rows)
    mass = sum((Fraction(str(float(weights[i]))) for i in rows), Fraction(0))
    candidates = []
    for feature in range(X.shape[1]):
        values = sorted(set(float(X[i, feature]) for i in rows))
        for a, b in zip(values[:-1], values[1:]):
            left = [i for i in rows if X[i, feature] <= a]
            right = [i for i in rows if X[i, feature] >= b]
            if len(left) < minimum or len(right) < minimum:
                continue
            left_mass = sum((Fraction(str(float(weights[i]))) for i in left), Fraction(0))
            right_mass = mass - left_mass
            gain = parent - (left_mass * _rational_gini(y, weights, left)
                             + right_mass * _rational_gini(y, weights, right)) / mass
            candidates.append((gain, feature, a, b, left, right))
    if not candidates:
        return None
    best = max(item[0] for item in candidates)
    # Exact rational ties resolve by feature then ascending boundary.
    return min((item for item in candidates if item[0] == best),
               key=lambda item: (item[1], item[2]))


def verify_cart_exhaustive_tiny(model, X, y, *, max_rows=24, max_features=5,
                               max_nodes=48, sample_weight=None):
    """Verify *every* split/stop on one non-bootstrap, all-feature CART tree.

    Use exact rational mass/Gini arithmetic on tiny integer/rational datasets.
    This is intentionally not an oracle for forests with random feature masks.
    """
    X, y = _matrix(X, y, max_rows, max_features)
    params = model.get_params()
    if (params["n_estimators"] != 1 or params["bootstrap"]
            or params["max_features"] is not None
            or params["class_weight"] is not None
            or params["min_impurity_decrease"] != 0):
        raise OracleNotApplicable("CART exhaustive oracle requires one full-feature, nonbootstrap, unweighted tree")
    weights = (np.ones(len(y), dtype=float) if sample_weight is None
               else np.asarray(sample_weight, dtype=float))
    if (weights.shape != (len(y),) or np.any(weights <= 0)
            or not np.isfinite(weights).all()):
        raise OracleNotApplicable("Tiny CART oracle only supports positive finite weights")
    tree = model.estimators_[0]
    if tree.node_count > max_nodes:
        raise OracleNotApplicable("CART node budget exceeded without partial success")
    visited = set()
    stack = [(0, list(range(len(y))))]
    checked = 0
    while stack:
        node, rows = stack.pop()
        if node in visited:
            raise OracleMismatch("CART node visited twice")
        visited.add(node)
        if tree.n_node_samples_[node] != len(rows):
            raise OracleMismatch("CART node samples disagree with independent routing")
        masses = [math.fsum(weights[i] for i in rows if int(y[i]) == c) for c in (0, 1)]
        for observed, wanted in zip(tree.class_counts_[node], masses):
            _assert_close(observed, wanted, "CART class mass")
        pure = len(set(y[i] for i in rows)) < 2
        can_split = (not pure
                     and len(rows) >= max(params["min_samples_split"],
                                          2 * params["min_samples_leaf"])
                     and (params["max_depth"] is None
                          or int(tree.depth_[node]) < params["max_depth"]))
        best = (_cart_best(X, y, weights, rows, params["min_samples_leaf"])
                if can_split else None)
        if best is None:
            if int(tree.feature_[node]) != -1:
                raise OracleMismatch("CART divided a mathematically terminal node")
            continue
        gain, feat, lower, upper, left, right = best
        if (int(tree.feature_[node]) != feat
                or int(tree.children_left_[node]) < 0
                or int(tree.children_right_[node]) < 0):
            raise OracleMismatch("CART selected a nonoptimal feature or dropped a feasible split")
        threshold = float(tree.threshold_[node])
        if not (lower <= threshold < upper):
            raise OracleMismatch("CART threshold does not match the optimal boundary")
        observed_left = [i for i in rows if X[i, feat] <= threshold]
        if observed_left != left:
            raise OracleMismatch("CART boundary produces a different partition")
        checked += 1
        stack.append((int(tree.children_right_[node]), right))
        stack.append((int(tree.children_left_[node]), left))
    if visited != set(range(tree.node_count)):
        raise OracleMismatch("CART has orphan nodes")
    return {
        "scope": "all_nodes_nonbootstrap_all_features_tiny_rational_CART",
        "nodes_checked": len(visited), "split_nodes_checked": checked,
        "reference_arithmetic": "fractions.Fraction",
    }


def _leaf_score(G, H, params):
    lam1 = float(params["lambda_l1"])
    lam2 = float(params["lambda_l2"])
    denominator = H + lam2
    if denominator <= 0:
        raise OracleNotApplicable("Newton objective has no positive denominator")
    softened = max(abs(G) - lam1, 0.0)
    magnitude = softened / denominator
    if params["max_delta_step"] > 0:
        magnitude = min(magnitude, float(params["max_delta_step"]))
    value = -math.copysign(magnitude, G) if softened else 0.0
    # Twice decrease of regularized quadratic surrogate at actual clipped step.
    score = 2 * softened * magnitude - denominator * magnitude ** 2
    return value, score


def _newton_best(Xbins, grad, hess, rows, feats, params, depth):
    if (len(rows) < 2 * params["min_data_in_leaf"]
            or params["max_depth"] is not None and depth >= params["max_depth"]):
        return None
    G = math.fsum(grad[i] for i in rows)
    H = math.fsum(hess[i] for i in rows)
    _, original = _leaf_score(G, H, params)
    candidates = []
    for feature in sorted(int(v) for v in feats):
        unique = sorted(set(int(Xbins[i, feature]) for i in rows))
        for boundary in unique[:-1]:
            left = [i for i in rows if int(Xbins[i, feature]) <= boundary]
            right = [i for i in rows if int(Xbins[i, feature]) > boundary]
            if (len(left) < params["min_data_in_leaf"]
                    or len(right) < params["min_data_in_leaf"]):
                continue
            GL, HL = math.fsum(grad[i] for i in left), math.fsum(hess[i] for i in left)
            GR, HR = math.fsum(grad[i] for i in right), math.fsum(hess[i] for i in right)
            if (HL < params["min_sum_hessian_in_leaf"]
                    or HR < params["min_sum_hessian_in_leaf"]):
                continue
            _, sl = _leaf_score(GL, HL, params)
            _, sr = _leaf_score(GR, HR, params)
            gain = math.fsum((sl, sr, -original))
            scale = math.fsum((abs(sl), abs(sr), abs(original)))
            if gain > params["min_gain_to_split"] and gain > 128 * np.finfo(float).eps * scale:
                candidates.append((gain, feature, boundary, left, right))
    if not candidates:
        return None
    maximum = max(item[0] for item in candidates)
    cutoff = maximum - 64 * np.finfo(float).eps * abs(maximum)
    return min((item for item in candidates if item[0] >= cutoff),
               key=lambda item: (item[1], item[2]))


def verify_booster_exhaustive_tiny(model, X, y, *, max_rows=24,
                                  max_features=5, max_nodes=64,
                                  max_rounds=8):
    """Independent exhaustive per-round leaf-wise split oracle on bounded data.

    Reconstructs Newton updates and all candidate feature/bin boundaries from
    raw bin assignments; GOSS multipliers and feature masks are consumed from
    already source-validated model evidence. It does *not* independently
    reproduce the PRNG or audit the quality of approximate training bins.
    """
    X, y = _matrix(X, y, max_rows, max_features)
    params = model.get_params()
    if (model.n_estimators_ > max_rounds or params["class_weight"] is not None
            or len(X) != model.n_samples_fit_):
        raise OracleNotApplicable("Booster tiny oracle requires bounded rounds and no class reweighting")
    bins = model.bin_mapper_.transform(X)
    p0 = 1 / (1 + math.exp(-float(model.base_score_)))
    raw = np.full(len(y), float(model.base_score_), dtype=float)
    checked_nodes, checked_splits = 0, 0
    for round_idx, tree in enumerate(model.estimators_):
        if checked_nodes + tree.node_count > max_nodes:
            raise OracleNotApplicable("Booster node budget exceeded without partial success")
        probs = np.array([1 / (1 + math.exp(-float(v))) for v in raw])
        gradients = probs - y.astype(float)
        hessians = probs * (1 - probs)
        record = model.sampling_records_[round_idx]
        rows = [int(v) for v in record["sampled_indices"]]
        multipliers = [float(v) for v in record["multipliers"]]
        grad = np.zeros(len(y), dtype=float)
        hess = np.zeros(len(y), dtype=float)
        for idx, scale in zip(rows, multipliers):
            grad[idx] = gradients[idx] * scale
            hess[idx] = hessians[idx] * scale
        if not len(rows) or len(rows) != len(set(rows)):
            raise OracleMismatch("Booster sampled row identity is invalid")
        features = list(model.feature_subsets_[round_idx])
        groups = {0: rows}
        expected_left, expected_right = {}, {}
        pending = {}
        next_node = 1
        while True:
            for node, group in list(groups.items()):
                if node not in pending and node not in expected_left:
                    depth = int(tree.depth_[node])
                    pending[node] = _newton_best(bins, grad, hess, group, features, params, depth)
            candidates = [(node, value) for node, value in pending.items() if value is not None]
            if not candidates or len(groups) >= tree.node_count:
                break
            maximum = max(item[1][0] for item in candidates)
            cutoff = maximum - 64 * np.finfo(float).eps * abs(maximum)
            node, (gain, feature, boundary, left, right) = min(
                (item for item in candidates if item[1][0] >= cutoff),
                key=lambda item: (item[1][1], item[1][2], item[0]),
            )
            pending.pop(node)
            if (int(tree.feature_[node]) != feature
                    or int(tree.threshold_bin_[node]) != boundary
                    or int(tree.children_left_[node]) != next_node
                    or int(tree.children_right_[node]) != next_node + 1):
                raise OracleMismatch("Booster diverged from exhaustive global leaf-wise optimum")
            _assert_close(tree.gain_[node], gain, "booster split gain", rtol=1e-9, atol=1e-11)
            expected_left[node], expected_right[node] = next_node, next_node + 1
            groups[next_node], groups[next_node + 1] = left, right
            next_node += 2
            checked_splits += 1
            # Children only become candidates on the next iteration.
        if len(groups) != tree.node_count:
            raise OracleMismatch("Booster contains more nodes than exhaustive growth")
        if len(groups) < 2 * params["num_leaves"] - 1:
            still_possible = any(candidate is not None for candidate in pending.values())
            if still_possible:
                raise OracleMismatch("Booster stopped before a feasible global leaf-wise split")
        for node, group in groups.items():
            if int(tree.n_node_samples_[node]) != len(group):
                raise OracleMismatch("Booster node count differs")
            G, H = math.fsum(grad[i] for i in group), math.fsum(hess[i] for i in group)
            value, _ = _leaf_score(G, H, params)
            _assert_close(tree.sum_gradient_[node], G, "booster node gradient")
            _assert_close(tree.sum_hessian_[node], H, "booster node Hessian")
            _assert_close(tree.value_[node], value, "booster clipped Newton leaf value")
        checked_nodes += len(groups)
        raw += params["learning_rate"] * tree.predict_binned(bins)
    return {
        "scope": "all_recorded_nodes_and_rounds_tiny_global_leafwise_exhaustion",
        "nodes_checked": checked_nodes, "split_nodes_checked": checked_splits,
        "rounds_checked": len(model.estimators_),
        "sampling_and_masks": "source_bound_fast_audit_plus_complete_model_refit",
    }


def run_tiny_oracle_canary():
    """Execute bounded independent checks; returns coverage, never EEG accuracy."""
    from .random_forest import HandcraftedRandomForestClassifier
    from .lightgbm import HandcraftedLightGBMClassifier

    cases = []
    datasets = [
        ("xor_zero_gain", np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2, dtype=float),
         np.array([0, 1, 1, 0] * 2, dtype=int), None),
        ("imbalanced_weighted", np.array([[i, i % 3] for i in range(10)], dtype=float),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1]), [1, 2, 1, 1, 3, 1, 2, 1, 1, 2]),
        ("constant_features", np.zeros((8, 2), dtype=float),
         np.array([0, 0, 1, 0, 1, 1, 0, 1]), None),
        ("adjacent_representable_floats",
         np.array([[np.nextafter(1., 0.), 0], [1., 0], [np.nextafter(1., 0.), 1],
                   [1., 1], [np.nextafter(1., 0.), 2], [1., 2]], dtype=float),
         np.array([0, 1, 0, 1, 0, 1]), None),
    ]
    for name, X, y, weights in datasets:
        model = HandcraftedRandomForestClassifier(
            n_estimators=1, bootstrap=False, max_features=None,
            max_depth=4, random_state=709, min_samples_leaf=1,
        ).fit(X, y, sample_weight=weights)
        check = verify_cart_exhaustive_tiny(model, X, y, sample_weight=weights)
        cases.append({"name": name, "algorithm": "handcrafted_CART",
                      "proof": check})
    X = np.array([
        [0, 0, 1], [0, 2, 0], [1, 0, 0], [2, 0, 1],
        [0, 1, 2], [1, 2, 0], [2, 1, 2], [1, 1, 1],
        [0, 0, 0], [2, 2, 0], [1, 0, 2], [0, 2, 2],
    ], dtype=float)
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=int)
    sparse = np.zeros((12, 4), dtype=float)
    for i in range(len(sparse)):
        sparse[i, i % 2] = 1 + i % 3
        sparse[i, 2 + (i // 2) % 2] = 1 + (i // 3) % 2
    sparse_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    for name, kwargs in [
        ("bagging_regularized", {"lambda_l1": 0, "lambda_l2": 1.0}),
        ("bounded_newton_l1", {"lambda_l1": .05, "lambda_l2": .1, "max_delta_step": .3}),
        ("goss_efb", {"data_sample_strategy": "goss", "top_rate": .4,
                       "other_rate": .4, "enable_bundle": True}),
    ]:
        xi, yi = (sparse, sparse_labels) if name == "goss_efb" else (X, y)
        params = {
            "n_estimators": 3, "num_leaves": 3, "max_depth": 3,
            "max_bin": 7, "min_data_in_leaf": 1,
            "learning_rate": .1, "random_state": 119, **kwargs,
        }
        model = HandcraftedLightGBMClassifier(**params).fit(xi, yi)
        if name == "goss_efb" and model.bundle_layout_.n_bundles_ >= xi.shape[1]:
            raise OracleMismatch("GOSS/EFB fixture did not actually compress any training features")
        check = verify_booster_exhaustive_tiny(model, xi, yi, max_rows=24,
                                               max_features=5, max_nodes=64)
        cases.append({"name": name, "algorithm": "handcrafted_histogram_Newton",
                      "proof": check, "n_feature_bundles": model.bundle_layout_.n_bundles_})
    return {
        "schema_version": "eeg-independent-tiny-oracles-1",
        "status": "verified_independent_tiny_oracles",
        "n_fixtures": len(cases),
        "coverage": cases,
        "limitations": [
            "tiny_exhaustive_math_not_a_proof_of_all_real_model_nodes",
            "oracle_uses_source_checked_sample_indices_and_feature_masks",
            "training_bins_are_fitted_by_the_model_and_replayed_by_the_fast_source_audit",
        ],
    }
