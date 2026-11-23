"""Auditable Random Forest implemented with Python and NumPy only.

This module owns CART split search, bootstrap sampling, node feature sampling,
probability voting, OOB accounting, and a non-executable JSON model format.  It
does not import a machine-learning library, including estimator base classes.

The chosen forest variant averages leaf class probabilities (soft voting).
Gini splits use sorted observations and cumulative class masses, with O(n)
scratch memory per feature rather than a samples-by-classes matrix.  The
minimum sample counts count positive-effective-weight bootstrap *draws*;
duplicated draws therefore count repeatedly.  Zero-effective-weight rows are
not sampled.  OOB accuracy is an unweighted diagnostic over eligible rows with
at least one vote, and is not an independent-participant performance estimate.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np


ALGORITHM_VERSION = "eeg-handcrafted-random-forest-1"
MODEL_SCHEMA_VERSION = "eeg-handcrafted-forest-json-1"
_TOL = 64 * np.finfo(np.float64).eps
_PARAMETERS = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
               "max_features", "bootstrap", "max_samples", "class_weight",
               "min_impurity_decrease", "random_state", "oob_score", "n_jobs")
_TREE_FIELDS = ("children_left", "children_right", "feature", "threshold",
                "n_node_samples", "weighted_n_node_samples", "class_counts",
                "impurity", "depth")


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _number(value: Any, name: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite allowed range")
    return result


def _readonly(value: Any, dtype=None) -> np.ndarray:
    """Back with immutable bytes, so setflags(write=True) cannot defeat it."""
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _public_array(array: np.ndarray) -> np.ndarray:
    """Fresh shape/stride metadata over immutable bytes, without a data copy."""
    buffer = array
    while isinstance(buffer, np.ndarray):
        buffer = buffer.base
    if buffer is None:
        # Constructor inputs may be read-only arrays which still own mutable
        # memory. Internal fitted arrays always reach an immutable bytes base.
        return _readonly(array)
    return np.frombuffer(buffer, dtype=array.dtype).reshape(array.shape)


def _matrix(X: Any, n_features: int | None = None, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0 or (not allow_empty and array.shape[0] == 0):
        raise ValueError("X must be a non-empty two-dimensional numeric feature matrix")
    if array.dtype.kind not in "biuf":
        raise ValueError("X must contain real numeric values, not strings, objects or complex values")
    if array.dtype.kind in "iu" and np.any((array > 2**53) | (array < -(2**53))):
        raise ValueError("Integer features beyond exact float64 range must be explicitly represented as float64")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("X contains NaN, infinity, or values outside float64 range")
    if n_features is not None and result.shape[1] != n_features:
        raise ValueError(f"X has {result.shape[1]} features; expected {n_features}")
    return result


def _labels(y: Any, n_samples: int) -> np.ndarray:
    if not isinstance(y, np.ndarray):
        # NumPy would silently coerce [1, "1"] into two identical strings,
        # or round mixed huge integers/floats before we can inspect them.
        source = np.asarray(y, dtype=object)
        if source.ndim == 1:
            kinds = set()
            for label in source:
                if isinstance(label, str):
                    kinds.add("string")
                elif isinstance(label, (bool, np.bool_)):
                    kinds.add("boolean")
                elif isinstance(label, (int, np.integer)):
                    kinds.add("integer")
                elif isinstance(label, (float, np.floating)):
                    kinds.add("float")
                else:
                    kinds.add("unsupported")
            if len(kinds) > 1 or "unsupported" in kinds:
                raise ValueError("Labels must have one homogeneous scalar type; implicit label coercion is forbidden")
    array = np.asarray(y)
    if array.ndim != 1 or len(array) != n_samples or not len(array):
        raise ValueError("y must be a one-dimensional label array matching X")
    if array.dtype.kind not in "biufU":
        raise ValueError("Labels must be homogeneous strings, booleans, integers or finite real numbers")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise ValueError("Floating class labels must use float64 precision or lower")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("Labels cannot contain NaN or infinity")
    if array.dtype.kind == "U" and any(not value for value in array.tolist()):
        raise ValueError("String labels cannot be empty")
    return array.copy()


def _weights(sample_weight: Any, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    raw = np.asarray(sample_weight)
    if raw.ndim != 1 or len(raw) != n_samples or raw.dtype.kind not in "iuf":
        raise ValueError("sample_weight must be a real one-dimensional array matching X")
    weights = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    try:
        total = math.fsum(weights.tolist())
    except OverflowError as exc:
        raise ValueError("sample_weight total mass exceeds float64 range") from exc
    if not math.isfinite(total) or total <= 0 or total > np.finfo(float).max / 4:
        raise ValueError("sample_weight must have positive, numerically representable total mass")
    return weights


def _max_features(value: Any, n_features: int) -> int:
    if value is None:
        return n_features
    if isinstance(value, str):
        return max(1, int(math.sqrt(n_features) if value == "sqrt" else math.log2(n_features)))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return max(1, int(math.floor(float(value) * n_features)))


def _draw_size(value: Any, n_samples: int) -> int:
    if value is None:
        return n_samples
    if isinstance(value, (int, np.integer)):
        return int(value)
    return max(1, int(math.ceil(float(value) * n_samples)))


def _tree_seed_sequence(seed: int, n_estimators: int) -> np.ndarray:
    return np.array([child.generate_state(1, dtype=np.uint64)[0]
                     for child in np.random.SeedSequence(seed).spawn(n_estimators)], dtype=np.uint64)


def _rngs(tree_seed: int):
    draw, split = np.random.SeedSequence(int(tree_seed)).spawn(2)
    return np.random.Generator(np.random.PCG64(draw)), np.random.Generator(np.random.PCG64(split))


def _bootstrap(eligible: np.ndarray, params: dict, seed: int) -> np.ndarray:
    if not params["bootstrap"]:
        return eligible.copy()
    draw_rng, _ = _rngs(seed)
    return eligible[draw_rng.integers(0, len(eligible), size=_draw_size(params["max_samples"], len(eligible)))]


def _impurity(counts: np.ndarray) -> float:
    mass = float(np.sum(counts))
    value = 1.0 - float(np.sum((counts / mass) ** 2))
    return max(0.0, min(1.0, value))


def _threshold(lo: float, hi: float) -> float:
    # No overflowing hi-lo or lo+hi; adjacent endpoints may have no interior
    # float.  The lower endpoint still reproduces the exact <= / > partition.
    mid = lo / 2.0 + hi / 2.0
    return float(mid if lo <= mid < hi else lo)


def _exact_fields(value: Any, keys, name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} fields differ from the versioned schema")


@dataclass(frozen=True, slots=True, eq=False)
class CARTTree:
    """Immutable fitted CART arrays; internal nodes send x <= threshold left."""

    children_left_: np.ndarray
    children_right_: np.ndarray
    feature_: np.ndarray
    threshold_: np.ndarray
    n_node_samples_: np.ndarray
    weighted_n_node_samples_: np.ndarray
    class_counts_: np.ndarray
    impurity_: np.ndarray
    depth_: np.ndarray
    n_features_in_: int

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name.endswith("_") and name[:-1] in _TREE_FIELDS:
            # Read-only storage alone does not protect NumPy shape metadata.
            # Never expose the array object stored inside the fitted tree.
            return _public_array(value) if isinstance(value, np.ndarray) and not value.flags.writeable else value
        return value

    def __post_init__(self):
        integer_names = ("children_left_", "children_right_", "feature_", "n_node_samples_", "depth_")
        for name in integer_names:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or value.dtype.kind not in "iu":
                raise ValueError(f"Tree {name} must be a one-dimensional integer array")
            if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
                raise ValueError(f"Tree {name} exceeds int64 range")
            object.__setattr__(self, name, _readonly(value, np.int64))
        for name in ("threshold_", "weighted_n_node_samples_", "class_counts_", "impurity_"):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"Tree {name} must contain finite real numbers")
            object.__setattr__(self, name, _readonly(value, np.float64))
        _integer(self.n_features_in_, "Tree n_features_in", 1)
        n = self.node_count
        if n == 0 or self.class_counts_.ndim != 2 or self.class_counts_.shape[0] != n or self.class_counts_.shape[1] == 0:
            raise ValueError("Tree class counts or node count are invalid")
        if any(getattr(self, field + "_").shape != (n,) for field in _TREE_FIELDS if field != "class_counts"):
            raise ValueError("Tree arrays have inconsistent node counts")
        if (np.any(self.n_node_samples_ <= 0) or np.any(self.weighted_n_node_samples_ <= 0)
                or np.any(self.class_counts_ < 0) or np.any(self.depth_ < 0)
                or self.depth_[0] != 0 or n > 2 * int(self.n_node_samples_[0]) - 1):
            raise ValueError("Tree has invalid sample counts, depths or size")
        if not np.allclose(self.class_counts_.sum(axis=1), self.weighted_n_node_samples_, rtol=2e-12, atol=0):
            raise ValueError("Tree class masses do not equal node masses")
        recalculated = np.array([_impurity(row) for row in self.class_counts_])
        if np.any(self.impurity_ < 0) or np.any(self.impurity_ > 1) or not np.allclose(recalculated, self.impurity_, atol=_TOL, rtol=_TOL):
            raise ValueError("Tree Gini impurities do not match its class masses")
        parents = np.zeros(n, dtype=int)
        for node in range(n):
            left, right, feature = int(self.children_left_[node]), int(self.children_right_[node]), int(self.feature_[node])
            if left == right == -1:
                if feature != -1 or self.threshold_[node] != 0:
                    raise ValueError("Leaf nodes require feature=-1 and threshold=0")
                continue
            if not (node < left < n and node < right < n and left != right and 0 <= feature < self.n_features_in_):
                raise ValueError("Tree contains invalid children, a cycle, or a feature outside the input")
            parents[[left, right]] += 1
            if (self.depth_[left] != self.depth_[node] + 1 or self.depth_[right] != self.depth_[node] + 1
                    or self.n_node_samples_[left] + self.n_node_samples_[right] != self.n_node_samples_[node]
                    or not np.allclose(self.class_counts_[left] + self.class_counts_[right], self.class_counts_[node], rtol=2e-12, atol=0)):
                raise ValueError("Tree child depths or sample/class conservation failed")
            weighted_child = ((self.weighted_n_node_samples_[left] / self.weighted_n_node_samples_[node]) * self.impurity_[left]
                              + (self.weighted_n_node_samples_[right] / self.weighted_n_node_samples_[node]) * self.impurity_[right])
            if self.impurity_[node] + 2e-12 < weighted_child:
                raise ValueError("Tree split increases weighted Gini impurity")
        if parents[0] != 0 or np.any(parents[1:] != 1):
            raise ValueError("Tree nodes must form one rooted tree without shared or unreachable children")

    @property
    def node_count(self) -> int:
        return len(self.feature_)

    @property
    def max_depth(self) -> int:
        return int(self.depth_.max())

    @property
    def feature_importances_(self) -> np.ndarray:
        scores = np.zeros(self.n_features_in_, dtype=float)
        for node in range(self.node_count):
            feature = int(self.feature_[node])
            if feature < 0:
                continue
            left, right = self.children_left_[node], self.children_right_[node]
            mass = self.weighted_n_node_samples_[node]
            gain = self.impurity_[node] - ((self.weighted_n_node_samples_[left] / mass) * self.impurity_[left]
                                         + (self.weighted_n_node_samples_[right] / mass) * self.impurity_[right])
            scores[feature] += (mass / self.weighted_n_node_samples_[0]) * max(0.0, gain)
        total = scores.sum()
        if total > 0:
            scores /= total
        return _readonly(scores)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        nodes = np.zeros(len(X), dtype=np.int64)
        active = self.feature_[nodes] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            current = nodes[rows]
            goes_left = X[rows, self.feature_[current]] <= self.threshold_[current]
            nodes[rows] = np.where(goes_left, self.children_left_[current], self.children_right_[current])
            active[rows] = self.feature_[nodes[rows]] >= 0
        counts = self.class_counts_[nodes]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        return self._predict(_matrix(X, self.n_features_in_, allow_empty=True))

    def to_dict(self) -> dict:
        return {name: getattr(self, name + "_").tolist() for name in _TREE_FIELDS}

    @classmethod
    def from_dict(cls, value: dict, *, n_features: int, n_classes: int):
        _exact_fields(value, _TREE_FIELDS, "Tree")
        arrays = {}
        integer_fields = {"children_left", "children_right", "feature", "n_node_samples", "depth"}
        for name in _TREE_FIELDS:
            raw = value[name]
            if not isinstance(raw, list):
                raise ValueError(f"Serialized tree {name} must be a list")
            if name in integer_fields:
                minimum = -1 if name in {"children_left", "children_right", "feature"} else 0
                arrays[name + "_"] = np.array([_integer(x, "Tree " + name, minimum, np.iinfo(np.int64).max)
                                                 for x in raw], dtype=np.int64)
            elif name == "class_counts":
                if any(not isinstance(row, list) or len(row) != n_classes for row in raw):
                    raise ValueError("Tree class mass rows must follow model classes")
                arrays[name + "_"] = np.array([[_number(x, "Tree class mass") for x in row] for row in raw], dtype=float)
            else:
                minimum = -np.finfo(float).max if name == "threshold" else 0
                arrays[name + "_"] = np.array([_number(x, "Tree " + name, minimum) for x in raw], dtype=float)
        result = cls(**arrays, n_features_in_=n_features)
        if result.class_counts_.shape[1] != n_classes:
            raise ValueError("Tree class order differs from forest classes")
        return result


def _feature_gains(X, y, weights, indices, feature, min_leaf, parent_gini, labels=None):
    """One sorted scan with O(n) scratch, including all valid boundary gains."""
    values = X[indices, feature]
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_y, sorted_w = values[order], y[indices[order]], weights[indices[order]]
    boundaries = np.arange(1, len(indices))
    valid = ((sorted_values[:-1] < sorted_values[1:])
             & (boundaries >= min_leaf) & ((len(indices) - boundaries) >= min_leaf))
    if not valid.any():
        return None
    observed_labels = np.unique(sorted_y) if labels is None else labels
    if len(observed_labels) == 2 and np.all(sorted_w == 1.0):
        # A common unweighted binary EEG fold has *integer-exact* class
        # counts. Avoid four weighted cumsums and repeated class-weight
        # buffers. For n <= 2**53 these divisions and ordered Gini
        # subtractions are identical to the generic float64 calculation.
        # Unbalanced or nonuniform weights always use the original path.
        left_mass = boundaries.astype(np.float64)
        right_mass = (len(indices) - boundaries).astype(np.float64)
        first = np.cumsum(sorted_y == observed_labels[0], dtype=np.int64)[:-1]
        left_first = first.astype(np.float64)
        right_first = (int(np.sum(sorted_y == observed_labels[0])) - first).astype(np.float64)
        left_second = left_mass - left_first
        right_second = right_mass - right_first
        left_gini = np.ones(len(indices) - 1)
        right_gini = np.ones(len(indices) - 1)
        left_gini -= (left_first / left_mass) ** 2
        right_gini -= (right_first / right_mass) ** 2
        left_gini -= (left_second / left_mass) ** 2
        right_gini -= (right_second / right_mass) ** 2
    else:
        left_mass = np.cumsum(sorted_w)[:-1]
        right_mass = np.cumsum(sorted_w[::-1])[::-1][1:]
        left_gini, right_gini = np.ones(len(indices) - 1), np.ones(len(indices) - 1)
        for label in observed_labels:
            class_w = np.where(sorted_y == label, sorted_w, 0.0)
            left_gini -= (np.cumsum(class_w)[:-1] / left_mass) ** 2
            right_gini -= (np.cumsum(class_w[::-1])[::-1][1:] / right_mass) ** 2
    np.clip(left_gini, 0.0, 1.0, out=left_gini)
    np.clip(right_gini, 0.0, 1.0, out=right_gini)
    total = left_mass + right_mass
    gains = parent_gini - (left_mass / total) * left_gini - (right_mass / total) * right_gini
    return order, sorted_values, valid, gains


def _best_split(X, y, weights, indices, feature_rng, feature_limit, min_leaf, parent_gini):
    """Use one global tolerance band, with O(n) scratch and O(m) scalars.

    Pairwise approximate equality is non-transitive. Keep actual maxima first,
    then rescan only the selected feature using the *global* cutoff; otherwise
    local boundary ties and feature ties can accidentally spend tolerance twice.
    """
    # One node has one class set. Reuse it for all sampled features instead
    # of sorting the same labels once per candidate. Retain only the
    # highest-maximum candidate's scan to avoid sorting it a second time.
    # Final feature/boundary selection still uses the *original* global
    # tolerance cutoff, never a local or chained approximate tie.
    labels = np.unique(y[indices])
    maxima = []
    cached_feature, cached_scan, highest = None, None, -np.inf
    for inspected, feature in enumerate(feature_rng.permutation(X.shape[1]), start=1):
        scanned = _feature_gains(X, y, weights, indices, feature, min_leaf, parent_gini, labels)
        if scanned is not None:
            _, _, valid, gains = scanned
            maximum = float(np.max(gains[valid]))
            maxima.append((int(feature), maximum))
            if maximum > highest:
                highest, cached_feature, cached_scan = maximum, int(feature), scanned
        # Constant/leaf-infeasible sampled features cannot force a dead end:
        # extend the permutation only until an admissible split has been seen.
        if inspected >= feature_limit and maxima:
            break
    if not maxima:
        return None
    cutoff = max(gain for _, gain in maxima) - _TOL
    feature = min(feature for feature, maximum in maxima if maximum >= cutoff)
    # A lower-index feature may win the global tie even though it does not
    # attain the largest raw gain; in that rare case perform the original
    # second scan. Common strict winners skip that redundant sort/scan.
    scanned = (cached_scan if feature == cached_feature else
               _feature_gains(X, y, weights, indices, feature,
                              min_leaf, parent_gini, labels))
    order, sorted_values, valid, gains = scanned
    boundary = int(np.flatnonzero(valid & (gains >= cutoff))[0])
    gain = float(gains[boundary])
    if gain < -_TOL:
        raise ValueError("Numerically unstable negative Gini decrease")
    threshold = _threshold(float(sorted_values[boundary]), float(sorted_values[boundary + 1]))
    return (max(0.0, gain), (feature, threshold),
            indices[order[:boundary + 1]], indices[order[boundary + 1:]])


def _fit_tree(X, y, weights, eligible, n_classes, params, seed):
    indices = _bootstrap(eligible, params, seed)
    # Repeated draws can overflow even when the original weight total fits.
    # Fail transactionally before bincount/cumulative arithmetic can produce
    # infinity; do not silently rescale or alter declared training weights.
    try:
        draw_mass = math.fsum(weights[indices].tolist())
    except OverflowError as exc:
        raise ValueError("Bootstrap draw weight mass exceeds float64 range") from exc
    if draw_mass > np.finfo(float).max / 4:
        raise ValueError("Bootstrap draw weight mass exceeds the numerical safety range")
    _, feature_rng = _rngs(seed)
    data = {name: [] for name in _TREE_FIELDS}

    def append_node(rows, depth):
        counts = np.bincount(y[rows], weights=weights[rows], minlength=n_classes)
        node = len(data["feature"])
        for name, value in (("children_left", -1), ("children_right", -1), ("feature", -1),
                            ("threshold", 0.0), ("n_node_samples", len(rows)),
                            ("weighted_n_node_samples", float(counts.sum())), ("class_counts", counts),
                            ("impurity", _impurity(counts)), ("depth", depth)):
            data[name].append(value)
        return node

    append_node(indices, 0)
    stack = [(0, indices)]
    root_mass = data["weighted_n_node_samples"][0]
    limit = _max_features(params["max_features"], X.shape[1])
    while stack:
        node, rows = stack.pop()
        depth = data["depth"][node]
        if (np.count_nonzero(data["class_counts"][node]) < 2
                or len(rows) < max(params["min_samples_split"], 2 * params["min_samples_leaf"])
                or (params["max_depth"] is not None and depth >= params["max_depth"])):
            continue
        best = _best_split(X, y, weights, rows, feature_rng, limit, params["min_samples_leaf"], data["impurity"][node])
        if best is None:
            continue
        gain, (feature, threshold), left_rows, right_rows = best
        weighted_gain = (data["weighted_n_node_samples"][node] / root_mass) * gain
        if weighted_gain < params["min_impurity_decrease"]:
            continue
        left, right = append_node(left_rows, depth + 1), append_node(right_rows, depth + 1)
        data["children_left"][node], data["children_right"][node] = left, right
        data["feature"][node], data["threshold"][node] = feature, threshold
        stack.extend(((right, right_rows), (left, left_rows)))
    tree = CARTTree(**{name + "_": np.asarray(values) for name, values in data.items()}, n_features_in_=X.shape[1])
    return tree, _readonly(indices, np.int64)


@dataclass(frozen=True, slots=True)
class _ForestState:
    params: dict
    classes: np.ndarray
    n_features: int
    n_samples: int
    eligible: np.ndarray
    tree_seeds: np.ndarray
    trees: tuple[CARTTree, ...]
    bootstrap_indices: tuple[np.ndarray, ...]
    importances: np.ndarray
    oob_counts: np.ndarray | None
    oob_decision: np.ndarray | None
    oob_score: float | None
    oob_coverage: float | None


def _forest_importances(trees):
    value = np.zeros(trees[0].n_features_in_)
    for tree in trees:
        value += tree.feature_importances_
    value /= len(trees)
    if value.sum() > 0:
        value /= value.sum()
    return _readonly(value)


class HandcraftedRandomForestClassifier:
    """Weighted-Gini CART forest with deterministic bootstrap and soft voting.

    ``random_state`` is a required deterministic integer (default 0), not None.
    ``max_features`` fractions use floor with minimum one. ``max_samples``
    fractions use ceil over positive-effective-weight eligible rows. Explicit
    class weights must cover exactly the observed labels; ``balanced`` uses
    positive-sample-weight row counts, then multiplies sample weights.

    Fitting is transactional: a failed fit preserves a previously fitted model.
    ``set_params`` validates first, then invalidates any fitted state. Returned
    fitted arrays cannot mutate the model. Tree-level work can use stdlib
    threads, but RNG streams and reduction order do not depend on scheduling.
    """

    def __init__(self, n_estimators=128, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 max_samples=None, class_weight=None, min_impurity_decrease=0.0,
                 random_state=0, oob_score=False, n_jobs=1):
        for key, value in locals().copy().items():
            if key in _PARAMETERS:
                setattr(self, key, deepcopy(value))
        self.validate_parameters()

    def get_params(self, deep=True) -> dict:
        return {name: deepcopy(getattr(self, name)) for name in _PARAMETERS}

    def set_params(self, **params):
        unknown = set(params) - set(_PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown random forest parameters: {sorted(unknown)}")
        candidate = type(self)(**(self.get_params() | params))
        for name in _PARAMETERS:
            setattr(self, name, getattr(candidate, name))
        self.__dict__.pop("_state", None)
        return self

    def validate_parameters(self, n_features=None, n_samples=None) -> None:
        _integer(self.n_estimators, "n_estimators", 1)
        if self.max_depth is not None:
            _integer(self.max_depth, "max_depth", 1)
        _integer(self.min_samples_split, "min_samples_split", 2)
        _integer(self.min_samples_leaf, "min_samples_leaf", 1)
        if n_features is not None:
            _integer(n_features, "n_features", 1)
        if n_samples is not None:
            _integer(n_samples, "n_samples", 1)
        value = self.max_features
        if value is None:
            pass
        elif isinstance(value, str):
            if value not in {"sqrt", "log2"}:
                raise ValueError("max_features must be sqrt, log2, None, a positive integer or fraction")
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            _integer(value, "max_features", 1, n_features)
        elif _number(value, "max_features", 0, 1) <= 0:
            raise ValueError("max_features fraction must be in (0,1]")
        if not isinstance(self.bootstrap, (bool, np.bool_)) or not isinstance(self.oob_score, (bool, np.bool_)):
            raise ValueError("bootstrap and oob_score must be booleans")
        if self.oob_score and not self.bootstrap:
            raise ValueError("OOB diagnostics require bootstrap=True")
        if self.max_samples is not None:
            if not self.bootstrap:
                raise ValueError("max_samples requires bootstrap=True")
            if isinstance(self.max_samples, (int, np.integer)) and not isinstance(self.max_samples, (bool, np.bool_)):
                _integer(self.max_samples, "max_samples", 1, n_samples)
            elif _number(self.max_samples, "max_samples", 0, 1) <= 0:
                raise ValueError("max_samples fraction must be in (0,1]")
        if self.class_weight is not None:
            if isinstance(self.class_weight, str):
                if self.class_weight != "balanced":
                    raise ValueError("class_weight supports only None, balanced, or an explicit label mapping")
            elif isinstance(self.class_weight, dict) and self.class_weight:
                for label, weight in self.class_weight.items():
                    if not isinstance(label, (str, bool, int, float, np.bool_, np.integer, np.floating)):
                        raise ValueError("Class-weight labels must be scalar labels")
                    if isinstance(label, (float, np.floating)) and not math.isfinite(float(label)):
                        raise ValueError("Class-weight labels must be finite")
                    _number(weight, "class_weight", 0)
            else:
                raise ValueError("class_weight must be None, balanced, or a nonempty label mapping")
        _number(self.min_impurity_decrease, "min_impurity_decrease", 0, 1)
        _integer(self.random_state, "random_state", 0, 2**64 - 1)
        if isinstance(self.n_jobs, (bool, np.bool_)) or not isinstance(self.n_jobs, (int, np.integer)) or (self.n_jobs != -1 and self.n_jobs < 1):
            raise ValueError("n_jobs must be -1 or a positive integer")

    def fit(self, X, y, sample_weight=None):
        self.validate_parameters()
        X = _matrix(X)
        y = _labels(y, len(X))
        classes, encoded = np.unique(y, return_inverse=True)
        weights = _weights(sample_weight, len(X))
        factors = np.ones(len(classes))
        if isinstance(self.class_weight, dict):
            if set(self.class_weight) != set(classes.tolist()):
                raise ValueError("Explicit class_weight must cover exactly all observed classes")
            factors = np.array([self.class_weight[label] for label in classes.tolist()], dtype=float)
        elif self.class_weight == "balanced":
            counts = np.bincount(encoded[weights > 0], minlength=len(classes))
            present = counts > 0
            factors[present] = counts.sum() / (present.sum() * counts[present])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            combined = weights * factors[encoded]
        if np.any((weights > 0) & (factors[encoded] > 0) & (combined == 0)):
            raise ValueError("Combined sample/class weights underflow float64")
        weights = _weights(combined, len(X))
        eligible = np.flatnonzero(weights > 0)
        self.validate_parameters(n_features=X.shape[1], n_samples=len(eligible))
        params = self.get_params()
        # Normalize scalar NumPy constructor values to their Python forms.
        for key in _PARAMETERS:
            if isinstance(params[key], np.bool_):
                params[key] = bool(params[key])
            elif isinstance(params[key], np.integer):
                params[key] = int(params[key])
            elif isinstance(params[key], np.floating):
                params[key] = float(params[key])
        seeds = _tree_seed_sequence(int(params["random_state"]), int(params["n_estimators"]))
        X.flags.writeable = encoded.flags.writeable = weights.flags.writeable = False

        def build(seed):
            return _fit_tree(X, encoded, weights, eligible, len(classes), params, int(seed))

        workers = min(len(seeds), (os.cpu_count() or 1) if self.n_jobs == -1 else int(self.n_jobs))
        if workers == 1:
            fitted = [build(seed) for seed in seeds]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="handcrafted-rf") as executor:
                fitted = list(executor.map(build, seeds))
        trees, bootstrap_indices = tuple(row[0] for row in fitted), tuple(row[1] for row in fitted)
        counts = decision = None
        oob_score = coverage = None
        if self.oob_score:
            counts, sums = np.zeros(len(X), dtype=np.int64), np.zeros((len(X), len(classes)))
            for tree, drawn in fitted:
                absent = np.ones(len(X), dtype=bool)
                absent[drawn] = False
                rows = np.flatnonzero(absent)
                if len(rows):
                    sums[rows] += tree._predict(X[rows])
                    counts[rows] += 1
            covered = counts > 0
            decision = np.full_like(sums, np.nan)
            decision[covered] = sums[covered] / counts[covered, None]
            usable = covered & (weights > 0)
            coverage = float(np.count_nonzero(usable) / len(eligible))
            oob_score = float(np.mean(np.argmax(decision[usable], axis=1) == encoded[usable])) if usable.any() else float("nan")
            counts, decision = _readonly(counts), _readonly(decision)
        self._state = _ForestState(params, _readonly(classes), X.shape[1], len(X), _readonly(eligible),
                                  _readonly(seeds), trees, bootstrap_indices, _forest_importances(trees),
                                  counts, decision, oob_score, coverage)
        return self

    def _fitted(self) -> _ForestState:
        if "_state" not in self.__dict__:
            raise ValueError("Random forest is not fitted")
        return self._state

    @property
    def classes_(self):
        return _public_array(self._fitted().classes)

    @property
    def n_classes_(self):
        return len(self._fitted().classes)

    @property
    def n_features_in_(self):
        return self._fitted().n_features

    @property
    def n_samples_fit_(self):
        return self._fitted().n_samples

    @property
    def estimators_(self):
        return self._fitted().trees

    @property
    def tree_seeds_(self):
        return _public_array(self._fitted().tree_seeds)

    @property
    def bootstrap_indices_(self):
        return tuple(_public_array(rows) for rows in self._fitted().bootstrap_indices)

    @property
    def feature_importances_(self):
        return _public_array(self._fitted().importances)

    def _oob(self):
        state = self._fitted()
        if state.oob_counts is None:
            raise AttributeError("OOB diagnostics were not enabled for this fitted forest")
        return state

    @property
    def oob_counts_(self):
        return _public_array(self._oob().oob_counts)

    @property
    def oob_decision_function_(self):
        return _public_array(self._oob().oob_decision)

    @property
    def oob_score_(self):
        return self._oob().oob_score

    @property
    def oob_coverage_(self):
        return self._oob().oob_coverage

    @property
    def oob_eligible_mask_(self):
        state = self._oob()
        mask = np.zeros(state.n_samples, dtype=bool)
        mask[state.eligible] = True
        return _readonly(mask)

    def predict_proba(self, X) -> np.ndarray:
        state = self._fitted()
        values = _matrix(X, state.n_features, allow_empty=True)
        result = np.zeros((len(values), len(state.classes)))
        for tree in state.trees:  # fixed tree order, independent of thread scheduling
            result += tree._predict(values)
        result /= len(state.trees)
        return result

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y, sample_weight=None) -> float:
        prediction = self.predict(X)
        truth = _labels(y, len(prediction))
        weights = _weights(sample_weight, len(prediction))
        return float(np.sum(weights * (prediction == truth)) / np.sum(weights))

    def to_dict(self) -> dict:
        """Export a detached JSON object; no executable pickle or input X."""
        state = self._fitted()
        params = deepcopy(state.params)
        if isinstance(params["class_weight"], dict):
            params["class_weight"] = {"labels": state.classes.tolist(),
                                      "weights": [float(params["class_weight"][label]) for label in state.classes.tolist()]}
        kind = {"b": "boolean", "i": "integer", "u": "unsigned_integer", "f": "float", "U": "string"}[state.classes.dtype.kind]
        oob = None
        if state.oob_counts is not None:
            oob = {"counts": state.oob_counts.tolist(),
                   "decision_function": [[float(x) if math.isfinite(float(x)) else None for x in row] for row in state.oob_decision],
                   "score": None if not math.isfinite(state.oob_score) else state.oob_score,
                   "coverage": state.oob_coverage}
        return {"schema_version": MODEL_SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
                "params": params, "n_features_in": state.n_features, "training_n_samples": state.n_samples,
                "classes": {"kind": kind, "values": state.classes.tolist()},
                "feature_importances": state.importances.tolist(),
                "training_eligible_indices": state.eligible.tolist(), "tree_seeds": [int(x) for x in state.tree_seeds],
                "bootstrap_indices": [rows.tolist() for rows in state.bootstrap_indices],
                "trees": [tree.to_dict() for tree in state.trees], "oob": oob}

    @classmethod
    def from_dict(cls, value: dict):
        """Validate and load predictive structure; not proof of data provenance.

        Tree topology, mass conservation, Gini values, bootstrap/RNG identity,
        and OOB coverage/probability structure are checked. Rechecking learned
        splits or OOB predictions against training data requires the separate
        experiment archive: the model deliberately does not contain input X.
        """
        keys = ("schema_version", "algorithm_version", "params", "n_features_in", "training_n_samples",
                "classes", "feature_importances", "training_eligible_indices", "tree_seeds", "bootstrap_indices", "trees", "oob")
        _exact_fields(value, keys, "Forest")
        if value["schema_version"] != MODEL_SCHEMA_VERSION or value["algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("Unsupported forest schema or algorithm version")
        n_features = _integer(value["n_features_in"], "n_features_in", 1)
        n_samples = _integer(value["training_n_samples"], "training_n_samples", 1)
        raw_importances = value["feature_importances"]
        if not isinstance(raw_importances, list) or len(raw_importances) != n_features:
            raise ValueError("Serialized importance vector must bind the declared feature count")
        importances = np.array([_number(x, "Feature importance", 0, 1) for x in raw_importances])
        _exact_fields(value["classes"], ("kind", "values"), "Classes")
        kind, labels = value["classes"]["kind"], value["classes"]["values"]
        if not isinstance(labels, list) or not labels:
            raise ValueError("Model classes must be a nonempty list")
        if kind == "string":
            if any(not isinstance(x, str) or not x for x in labels):
                raise ValueError("Invalid string classes")
            classes = np.array(labels, dtype=str)
        elif kind == "boolean":
            if any(type(x) is not bool for x in labels):
                raise ValueError("Invalid boolean classes")
            classes = np.array(labels, dtype=bool)
        elif kind in {"integer", "unsigned_integer"}:
            lower, upper = (0, 2**64 - 1) if kind == "unsigned_integer" else (-(2**63), 2**63 - 1)
            checked = [_integer(x, "class label", lower, upper) for x in labels]
            classes = np.array(checked, dtype=np.uint64 if kind == "unsigned_integer" else np.int64)
        elif kind == "float":
            if any(type(x) not in {int, float} or not math.isfinite(float(x)) for x in labels):
                raise ValueError("Invalid finite float classes")
            classes = np.array(labels, dtype=float)
        else:
            raise ValueError("Unsupported model class kind")
        if not np.array_equal(classes, np.unique(classes)):
            raise ValueError("Model classes must be unique and sorted")
        _exact_fields(value["params"], _PARAMETERS, "Forest parameters")
        params = deepcopy(value["params"])
        if isinstance(params["class_weight"], dict):
            mapping = params["class_weight"]
            _exact_fields(mapping, ("labels", "weights"), "Serialized class weights")
            mapped_labels = mapping["labels"]
            if (not isinstance(mapped_labels, list) or len(mapped_labels) != len(labels)
                    or any(type(a) is not type(b) or a != b for a, b in zip(mapped_labels, labels))
                    or not isinstance(mapping["weights"], list) or len(mapping["weights"]) != len(labels)):
                raise ValueError("Serialized class weights must follow model class order")
            params["class_weight"] = dict(zip(classes.tolist(), mapping["weights"]))
        model = cls(**params)
        eligible_raw = value["training_eligible_indices"]
        if not isinstance(eligible_raw, list) or not eligible_raw:
            raise ValueError("Model must declare eligible training rows")
        eligible = np.array([_integer(x, "eligible row", 0, n_samples - 1) for x in eligible_raw], dtype=np.int64)
        if not np.array_equal(eligible, np.unique(eligible)):
            raise ValueError("Eligible rows must be unique and sorted")
        model.validate_parameters(n_features=n_features, n_samples=len(eligible))
        # Bind allocation dimensions to materialized payload lengths before
        # creating seed schedules or training-row arrays from untrusted JSON.
        if (not isinstance(value["trees"], list) or not isinstance(value["bootstrap_indices"], list)
                or len(value["trees"]) != params["n_estimators"] or len(value["bootstrap_indices"]) != params["n_estimators"]):
            raise ValueError("Forest tree/bootstrap count differs from n_estimators")
        if not isinstance(value["tree_seeds"], list):
            raise ValueError("Tree seeds must be a list")
        seeds = np.array([_integer(x, "tree seed", 0, 2**64 - 1) for x in value["tree_seeds"]], dtype=np.uint64)
        expected_seeds = _tree_seed_sequence(params["random_state"], params["n_estimators"])
        if not np.array_equal(seeds, expected_seeds):
            raise ValueError("Tree RNG seeds differ from the configured deterministic schedule")
        trees, draws = [], []
        expected_counts = None
        if params["oob_score"]:
            _exact_fields(value["oob"], ("counts", "decision_function", "score", "coverage"), "OOB")
            if not isinstance(value["oob"]["counts"], list) or len(value["oob"]["counts"]) != n_samples:
                raise ValueError("OOB counts must match training rows")
            expected_counts = np.zeros(n_samples, dtype=np.int64)
        for seed, serialized, raw_draws in zip(seeds, value["trees"], value["bootstrap_indices"]):
            if not isinstance(raw_draws, list) or not raw_draws:
                raise ValueError("Bootstrap rows must be a nonempty list")
            sampled = np.array([_integer(x, "bootstrap row", 0, n_samples - 1) for x in raw_draws], dtype=np.int64)
            if not np.array_equal(sampled, _bootstrap(eligible, params, int(seed))):
                raise ValueError("Bootstrap samples differ from configured deterministic draws")
            tree = CARTTree.from_dict(serialized, n_features=n_features, n_classes=len(classes))
            if tree.n_node_samples_[0] != len(sampled):
                raise ValueError("Tree root count differs from its bootstrap sample count")
            if params["max_depth"] is not None and tree.max_depth > params["max_depth"]:
                raise ValueError("Tree exceeds configured max_depth")
            internal = tree.feature_ >= 0
            if np.any(tree.n_node_samples_[internal] < params["min_samples_split"]):
                raise ValueError("Tree split violates min_samples_split")
            # A whole tree may be a stump with fewer than min_samples_leaf;
            # only children created by a split must meet the leaf minimum.
            if np.any(tree.n_node_samples_[1:] < params["min_samples_leaf"]):
                raise ValueError("Tree children violate min_samples_leaf")
            for node in np.flatnonzero(internal):
                left, right = tree.children_left_[node], tree.children_right_[node]
                mass = tree.weighted_n_node_samples_[node]
                gain = tree.impurity_[node] - (tree.weighted_n_node_samples_[left] / mass) * tree.impurity_[left] - (tree.weighted_n_node_samples_[right] / mass) * tree.impurity_[right]
                if ((params["min_impurity_decrease"] > 0 and gain <= 0)
                        or (mass / tree.weighted_n_node_samples_[0]) * gain + 2e-12 < params["min_impurity_decrease"]):
                    raise ValueError("Tree split violates min_impurity_decrease")
            trees.append(tree)
            draws.append(_readonly(sampled))
            if expected_counts is not None:
                absent = np.ones(n_samples, dtype=bool)
                absent[sampled] = False
                expected_counts += absent
        counts = decision = None
        score = coverage = None
        if params["oob_score"]:
            oob = value["oob"]
            _exact_fields(oob, ("counts", "decision_function", "score", "coverage"), "OOB")
            if not isinstance(oob["counts"], list) or len(oob["counts"]) != n_samples:
                raise ValueError("OOB counts must match training rows")
            counts = np.array([_integer(x, "OOB count", 0, len(trees)) for x in oob["counts"]], dtype=np.int64)
            if not np.array_equal(counts, expected_counts):
                raise ValueError("OOB counts differ from bootstrap membership")
            raw = oob["decision_function"]
            if not isinstance(raw, list) or len(raw) != n_samples or any(not isinstance(row, list) or len(row) != len(classes) for row in raw):
                raise ValueError("OOB probabilities must match training rows and classes")
            decision = np.empty((n_samples, len(classes)))
            for index, row in enumerate(raw):
                if counts[index] == 0:
                    if any(x is not None for x in row):
                        raise ValueError("Uncovered OOB probabilities must be null, never imputed")
                    decision[index] = np.nan
                else:
                    decision[index] = [_number(x, "OOB probability", 0, 1) for x in row]
                    if not math.isclose(float(decision[index].sum()), 1.0, rel_tol=2e-12, abs_tol=_TOL):
                        raise ValueError("Covered OOB probabilities must sum to one")
            coverage = _number(oob["coverage"], "OOB coverage", 0, 1)
            wanted = float(np.count_nonzero(counts[eligible]) / len(eligible))
            if coverage != wanted:
                raise ValueError("OOB coverage does not match eligible covered rows")
            if wanted == 0:
                if oob["score"] is not None:
                    raise ValueError("OOB score must be null when no eligible row has votes")
                score = float("nan")
            else:
                score = _number(oob["score"], "OOB score", 0, 1)
            counts, decision = _readonly(counts), _readonly(decision)
        elif value["oob"] is not None:
            raise ValueError("OOB diagnostics must be absent when disabled")
        reconstructed_importances = _forest_importances(trees)
        if not np.allclose(importances, reconstructed_importances, rtol=2e-12, atol=_TOL):
            raise ValueError("Serialized importances disagree with fitted tree Gini decreases")
        model._state = _ForestState(deepcopy(params), _readonly(classes), n_features, n_samples, _readonly(eligible),
                                   _readonly(seeds), tuple(trees), tuple(draws), reconstructed_importances,
                                   counts, decision, score, coverage)
        return model
