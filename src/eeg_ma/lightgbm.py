"""Auditable, handwritten LightGBM-style binary gradient boosting.

Only Python, NumPy, and this project's histogram engine implement the learner.
The algorithm uses training-only quantile bins, lossless zero-conflict feature
bundling, Newton histograms, histogram subtraction, and globally best-first
leaf-wise growth. Optional GOSS keeps large weighted gradients and corrects
the sampled remainder. It is a deliberately specified numeric binary variant,
not a wrapper around, or a bit-compatible reimplementation of, LightGBM.

The objective is weighted logistic loss. The leaf score is twice the reduction
in its regularized quadratic objective, including the actual clipped Newton
step. Counts count retained rows, while GOSS corrections apply only to gradient
and Hessian sums. Zero-effective-weight rows never determine bins, bundles, or
sampling. Missing values, categorical features, and multiclass objectives are
outside this version's contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

from .lightgbm_histogram import (
    BinMapper,
    FeatureBundler,
    HistogramSubtractionError,
    build_histograms,
    subtract_histograms,
)


ALGORITHM_VERSION = "eeg-handcrafted-lightgbm-1"
MODEL_SCHEMA_VERSION = "eeg-handcrafted-lightgbm-json-1"
HESSIAN_FLOOR = 1e-12
_EPS = np.finfo(np.float64).eps
_TIE_TOLERANCE = 64 * _EPS
_GAIN_TOLERANCE = 128 * _EPS
_CHECK_TOLERANCE = 2e-12
_INT64_MAX = np.iinfo(np.int64).max
_PARAMETERS = (
    "n_estimators", "learning_rate", "num_leaves", "max_depth", "max_bin",
    "min_data_in_bin", "min_data_in_leaf", "min_sum_hessian_in_leaf",
    "lambda_l1", "lambda_l2", "min_gain_to_split", "max_delta_step",
    "feature_fraction", "bagging_fraction", "data_sample_strategy", "top_rate",
    "other_rate", "enable_bundle", "random_state", "n_jobs", "class_weight",
    "early_stopping_rounds", "early_stopping_min_delta",
)
_TREE_FIELDS = (
    "children_left", "children_right", "feature", "threshold_bin", "threshold",
    "n_node_samples", "sum_gradient", "sum_hessian", "value", "gain", "depth",
)
_TREE_INTEGERS = {
    "children_left", "children_right", "feature", "threshold_bin",
    "n_node_samples", "depth",
}
_MODEL_FIELDS = (
    "schema_version", "algorithm_version", "params", "classes", "n_features_in",
    "training_n_samples", "training_eligible_indices", "class_weight_factors",
    "training_class_mass", "base_score", "bin_mapper", "bundle_layout",
    "iterations", "best_iteration", "training_loss", "validation_loss",
    "stopping_reason", "feature_importances", "feature_split_counts",
)


def _integer(value: Any, name: str, minimum: int = 0,
             maximum: int | None = _INT64_MAX) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _number(value: Any, name: str, minimum: float = 0,
            maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is outside float64 range") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite allowed range")
    return result


def _json_integer(value: Any, name: str, minimum: int = 0,
                  maximum: int = _INT64_MAX) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    return _integer(value, name, minimum, maximum)


def _json_number(value: Any, name: str, minimum: float = -math.inf,
                 maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a JSON number")
    return _number(value, name, minimum, maximum)


def _exact_fields(value: Any, keys, name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} fields differ from the versioned schema")


def _readonly(value: Any, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _public_array(array: np.ndarray) -> np.ndarray:
    """Do not expose mutable shape metadata, even on read-only data buffers."""
    buffer = array
    while isinstance(buffer, np.ndarray):
        buffer = buffer.base
    if not isinstance(buffer, bytes):
        return _readonly(array)
    return np.frombuffer(buffer, dtype=array.dtype).reshape(array.shape)


def _matrix(X: Any, n_features: int | None = None, *, allow_empty=False) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0 or (not allow_empty and len(array) == 0):
        raise ValueError("X must be a non-empty two-dimensional numeric feature matrix")
    if array.dtype.kind not in "biuf":
        raise ValueError("X must contain real numeric values, not objects, strings, or complex values")
    if array.dtype.kind in "iu" and np.any((array > 2**53) | (array < -(2**53))):
        raise ValueError("Integer features beyond exact float64 range require explicit float64 representation")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("X contains NaN, infinity, or values outside float64 range")
    if n_features is not None and result.shape[1] != n_features:
        raise ValueError(f"X has {result.shape[1]} features; expected {n_features}")
    return result


def _labels(y: Any, n_samples: int) -> np.ndarray:
    if not isinstance(y, np.ndarray):
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
                raise ValueError("Labels require one homogeneous scalar type; implicit coercion is forbidden")
    array = np.asarray(y)
    if array.ndim != 1 or len(array) != n_samples or not len(array):
        raise ValueError("y must be a non-empty one-dimensional label array matching X")
    if array.dtype.kind not in "biufU":
        raise ValueError("Labels must be homogeneous strings, booleans, integers, or finite numbers")
    if array.dtype.kind == "f" and (array.dtype.itemsize > 8 or not np.isfinite(array).all()):
        raise ValueError("Floating labels must be finite and use at most float64 precision")
    if array.dtype.kind == "U" and any(not value for value in array.tolist()):
        raise ValueError("String labels cannot be empty")
    return array.copy()


def _encoded_labels(y: Any, n_samples: int, classes: np.ndarray) -> np.ndarray:
    labels = _labels(y, n_samples)
    # Integers of different widths can represent the same labels exactly; never
    # cross between float/int, string/numeric, or boolean/integer namespaces.
    kind = labels.dtype.kind
    expected = classes.dtype.kind
    if kind != expected and not (kind in "iu" and expected in "iu"):
        raise ValueError("Evaluation labels must have the same scalar type as training labels")
    # NumPy compares signed int64 with uint64 through float64 in some mixed
    # cases, which can alias distinct labels above 2**53. Python scalar keys
    # preserve the full integer value in every supported label namespace.
    lookup = {label: index for index, label in enumerate(classes.tolist())}
    try:
        return np.array([lookup[label] for label in labels.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("Evaluation labels contain a class absent from training") from exc


def _label_kind(label):
    if isinstance(label, (bool, np.bool_)):
        return "b"
    if isinstance(label, (int, np.integer)):
        return "i"
    if isinstance(label, (float, np.floating)):
        return "f"
    if isinstance(label, str):
        return "U"
    return None


def _sum(values, name: str) -> float:
    try:
        result = math.fsum(float(x) for x in values)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds numerical range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _weights(sample_weight: Any, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    raw = np.asarray(sample_weight)
    if raw.ndim != 1 or len(raw) != n_samples or raw.dtype.kind not in "iuf":
        raise ValueError("sample_weight must be a real one-dimensional array matching X")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and nonnegative")
    total = _sum(weights, "sample_weight total")
    if total <= 0 or total > np.finfo(float).max / 4:
        raise ValueError("sample_weight requires positive numerically representable total mass")
    return weights


def _combine_weights(weights, encoded, factors):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        combined = weights * factors[encoded]
    if np.any((weights > 0) & (factors[encoded] > 0) & (combined == 0)):
        raise ValueError("Combined sample/class weights underflow float64")
    return _weights(combined, len(weights))


def _sigmoid(raw: np.ndarray) -> np.ndarray:
    result = np.empty_like(raw, dtype=np.float64)
    positive = raw >= 0
    with np.errstate(under="ignore"):
        result[positive] = 1 / (1 + np.exp(-raw[positive]))
        exponential = np.exp(raw[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def logistic_gradients(raw, y01, weights) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted logistic G/H, preserving saturated negative gradients.

    The Hessian floor is applied to p(1-p) BEFORE multiplying by effective
    sample weight. Positive weights that cannot retain a positive float64
    Hessian are rejected rather than silently dropped.
    """
    raw = np.asarray(raw, dtype=np.float64)
    encoded = np.asarray(y01)
    weights = np.asarray(weights, dtype=np.float64)
    if (raw.ndim != 1 or encoded.shape != raw.shape or weights.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isin(encoded, [0, 1]).all()):
        raise ValueError("Logistic derivatives require aligned finite scores, binary labels, and nonnegative weights")
    p, q = _sigmoid(raw), _sigmoid(-raw)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gradient = weights * np.where(encoded == 1, -q, p)
        hessian = weights * np.maximum(p * q, HESSIAN_FLOOR)
    if (not np.isfinite(gradient).all() or not np.isfinite(hessian).all()
            or np.any((weights > 0) & (hessian <= 0))):
        raise ValueError("Weighted logistic derivatives exceed float64 dynamic range")
    return gradient, hessian


def _logloss(raw, encoded, weights) -> float:
    # logaddexp avoids log(0); normalize weights before multiplying by losses.
    loss = np.logaddexp(0.0, np.where(encoded == 1, -raw, raw))
    normalized = weights / _sum(weights, "Loss weight total")
    return _sum(loss * normalized, "Weighted logistic loss")


def _newton(gradient: float, hessian: float, params: dict) -> tuple[float, float]:
    """Clipped Newton value and twice its regularized quadratic improvement."""
    denominator = _sum((hessian, params["lambda_l2"]), "Newton denominator")
    if denominator <= 0:
        raise ValueError("Newton denominator must be positive")
    softened = max(abs(gradient) - params["lambda_l1"], 0.0)
    if softened == 0:
        return 0.0, 0.0
    # An overflowing ratio is harmless when bounded by a finite step limit.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        magnitude = float(np.float64(softened) / denominator)
    if params["max_delta_step"] > 0:
        magnitude = min(magnitude, params["max_delta_step"])
    if not math.isfinite(magnitude):
        raise ValueError("Unbounded Newton step exceeds float64 range; use a positive max_delta_step")
    value = -math.copysign(magnitude, gradient)
    # Algebraically -2*G*v-(H+L2)*v*v-2*L1*abs(v), without G**2
    # and without cancellation between a large G and its L1 threshold.
    residual = softened - denominator * magnitude
    score = _sum((magnitude * softened, magnitude * residual), "Newton leaf score")
    if score < 0:
        raise ValueError("Newton leaf score became negative outside numerical tolerance")
    return value, score


def _gain(left_gradient, left_hessian, right_gradient, right_hessian,
          parent_gradient, parent_hessian, params):
    left_value, left_score = _newton(left_gradient, left_hessian, params)
    right_value, right_score = _newton(right_gradient, right_hessian, params)
    _, parent_score = _newton(parent_gradient, parent_hessian, params)
    gain = _sum((left_score, right_score, -parent_score), "Split gain")
    scale = _sum((abs(left_score), abs(right_score), abs(parent_score)), "Split score scale")
    return gain, _GAIN_TOLERANCE * scale, left_value, right_value


def iteration_seed_sequence(master_seed, n) -> np.ndarray:
    """Iteration seed schedule with identical prefixes for larger ensembles."""
    master_seed = _integer(master_seed, "random_state", 0, 2**64 - 1)
    n = _integer(n, "Iteration count", 0)
    return np.array([child.generate_state(1, dtype=np.uint64)[0]
                     for child in np.random.SeedSequence(master_seed).spawn(n)], dtype=np.uint64)


def _rngs(seed):
    seed = _integer(seed, "Iteration seed", 0, 2**64 - 1)
    rows, features = np.random.SeedSequence(seed).spawn(2)
    return np.random.Generator(np.random.PCG64(rows)), np.random.Generator(np.random.PCG64(features))


def sampled_features(n_features, fraction, iteration_seed) -> np.ndarray:
    """Sorted, without-replacement ceil(fraction * n_features) feature draw."""
    n_features = _integer(n_features, "n_features", 1)
    fraction = _number(fraction, "feature_fraction", 0, 1)
    if fraction <= 0:
        raise ValueError("feature_fraction must be positive")
    count = max(1, int(math.ceil(fraction * n_features)))
    if count == n_features:
        return np.arange(n_features, dtype=np.int64)
    _, rng = _rngs(iteration_seed)
    return np.sort(rng.choice(n_features, size=count, replace=False)).astype(np.int64)


def _sample_from_top(eligible: np.ndarray, params: dict, iteration_seed: int,
                     top: np.ndarray | None = None) -> dict:
    rng, _ = _rngs(iteration_seed)
    n = len(eligible)
    strategy = params["data_sample_strategy"]
    if strategy == "bagging":
        count = max(1, int(math.ceil(params["bagging_fraction"] * n)))
        selected = eligible.copy() if count == n else np.sort(rng.choice(eligible, size=count, replace=False))
        return {"strategy": strategy, "sampled_indices": selected.tolist(),
                "multipliers": [1.0] * len(selected), "top_indices": []}
    n_top = min(n - 1, int(math.ceil(params["top_rate"] * n)))
    n_low = min(n - n_top, int(math.ceil(params["other_rate"] * n)))
    if top is None or len(top) != n_top:
        raise ValueError("GOSS top-gradient row count is invalid")
    lower_pool = eligible[~np.isin(eligible, top)]
    low = np.sort(rng.choice(lower_pool, size=n_low, replace=False))
    selected = np.sort(np.concatenate((top, low)))
    multiplier = np.where(np.isin(selected, top), 1.0, len(lower_pool) / n_low)
    return {"strategy": strategy, "sampled_indices": selected.tolist(),
            "multipliers": multiplier.tolist(), "top_indices": np.sort(top).tolist()}


def sampling_plan(eligible_original_indices, full_gradients, params: dict,
                  iteration_seed: int) -> dict:
    """Reproducible original-row sample and finite-population GOSS factors.

    GOSS has no warmup. Top and low sample sizes use ceil(rate * eligible_n),
    leaving at least one low-pool row. The correction is the ACTUAL low-pool
    size divided by its draw count, not an asymptotic rate approximation.
    """
    eligible = np.asarray(eligible_original_indices)
    gradient = np.asarray(full_gradients)
    if (eligible.ndim != 1 or eligible.dtype.kind not in "iu" or len(eligible) < 2
            or gradient.ndim != 1 or gradient.dtype.kind not in "iuf"
            or not np.isfinite(gradient).all() or np.any(eligible < 0)
            or np.any(eligible >= len(gradient)) or not np.array_equal(eligible, np.unique(eligible))):
        raise ValueError("Sampling requires sorted unique eligible rows and aligned finite gradients")
    eligible = eligible.astype(np.int64, copy=False)
    top = None
    if params["data_sample_strategy"] == "goss":
        count = min(len(eligible) - 1, int(math.ceil(params["top_rate"] * len(eligible))))
        rank = np.lexsort((eligible, -np.abs(gradient[eligible])))
        top = np.sort(eligible[rank[:count]])
    elif params["data_sample_strategy"] != "bagging":
        raise ValueError("Unknown data_sample_strategy")
    return _sample_from_top(eligible, params, iteration_seed, top)


def _close(a: float, b: float, *, scale: float = 0.0) -> bool:
    return abs(a - b) <= _CHECK_TOLERANCE * max(abs(a), abs(b), scale)


def _conserved(parent, left, right) -> bool:
    return _close(parent, _sum((left, right), "Child sum"),
                  scale=max(abs(parent), abs(left), abs(right)))


@dataclass(frozen=True, slots=True, eq=False)
class HistogramNewtonTree:
    """Immutable node arrays; leaf values exclude the ensemble learning rate."""

    children_left_: np.ndarray
    children_right_: np.ndarray
    feature_: np.ndarray
    threshold_bin_: np.ndarray
    threshold_: np.ndarray
    n_node_samples_: np.ndarray
    sum_gradient_: np.ndarray
    sum_hessian_: np.ndarray
    value_: np.ndarray
    gain_: np.ndarray
    depth_: np.ndarray
    n_features_in_: int

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name.endswith("_") and name[:-1] in _TREE_FIELDS and isinstance(value, np.ndarray):
            return _public_array(value)
        return value

    def __post_init__(self):
        for field in _TREE_FIELDS:
            name = field + "_"
            value = np.asarray(getattr(self, name))
            if value.ndim != 1:
                raise ValueError(f"Tree {field} must be a one-dimensional array")
            if field in _TREE_INTEGERS:
                if value.dtype.kind not in "iu" or (value.dtype.kind == "u" and np.any(value > _INT64_MAX)):
                    raise ValueError(f"Tree {field} requires int64-representable integers")
                value = _readonly(value, np.int64)
            else:
                if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                    raise ValueError(f"Tree {field} requires finite real numbers")
                value = _readonly(value, np.float64)
            object.__setattr__(self, name, value)
        _integer(self.n_features_in_, "Tree feature count", 1)
        n = self.node_count
        if n == 0 or any(len(getattr(self, field + "_")) != n for field in _TREE_FIELDS):
            raise ValueError("Tree arrays have inconsistent or empty node counts")
        if (np.any(self.n_node_samples_ <= 0) or np.any(self.sum_hessian_ <= 0)
                or np.any(self.gain_ < 0) or np.any(self.depth_ < 0) or self.depth_[0] != 0
                or n > 2 * int(self.n_node_samples_[0]) - 1):
            raise ValueError("Tree has invalid counts, Hessians, gains, depths, or size")
        parents = np.zeros(n, dtype=np.int64)
        for node in range(n):
            left, right = int(self.children_left_[node]), int(self.children_right_[node])
            feature = int(self.feature_[node])
            if left == right == -1:
                if feature != -1 or self.threshold_bin_[node] != -1 or self.threshold_[node] != 0 or self.gain_[node] != 0:
                    raise ValueError("Leaf nodes require canonical feature, threshold, and gain markers")
                continue
            if not (node < left < n and node < right < n and left != right
                    and 0 <= feature < self.n_features_in_ and self.threshold_bin_[node] >= 0
                    and self.gain_[node] > 0):
                raise ValueError("Tree has invalid children, cycle, split feature, boundary, or gain")
            parents[left] += 1
            parents[right] += 1
            if (self.depth_[left] != self.depth_[node] + 1 or self.depth_[right] != self.depth_[node] + 1
                    or self.n_node_samples_[left] + self.n_node_samples_[right] != self.n_node_samples_[node]
                    or not _conserved(self.sum_gradient_[node], self.sum_gradient_[left], self.sum_gradient_[right])
                    or not _conserved(self.sum_hessian_[node], self.sum_hessian_[left], self.sum_hessian_[right])):
                raise ValueError("Tree child depths, sample counts, or derivative conservation failed")
        if parents[0] != 0 or np.any(parents[1:] != 1):
            raise ValueError("Tree must have one root and exactly one parent for every other node")

    @property
    def node_count(self):
        return len(self.feature_)

    @property
    def max_depth(self):
        return int(np.max(self.depth_))

    def _predict(self, values, *, binned):
        result = np.empty(len(values), dtype=np.float64)
        stack = [(0, np.arange(len(values), dtype=np.int64))]
        left, right, feature = self.children_left_, self.children_right_, self.feature_
        thresholds = self.threshold_bin_ if binned else self.threshold_
        leaf_values = self.value_
        while stack:
            node, rows = stack.pop()
            if not len(rows):
                continue
            if left[node] < 0:
                result[rows] = leaf_values[node]
            else:
                mask = values[rows, feature[node]] <= thresholds[node]
                stack.append((int(right[node]), rows[~mask]))
                stack.append((int(left[node]), rows[mask]))
        return result

    def predict(self, X):
        return self._predict(_matrix(X, self.n_features_in_, allow_empty=True), binned=False)

    def predict_binned(self, bins):
        values = np.asarray(bins)
        if (values.ndim != 2 or values.shape[1] != self.n_features_in_
                or values.dtype.kind not in "iu" or np.any(values < 0)):
            raise ValueError("Binned prediction requires nonnegative integer bins with the fitted feature count")
        return self._predict(values, binned=True)

    def to_dict(self):
        return {"n_features_in": int(self.n_features_in_),
                **{field: getattr(self, field + "_").tolist() for field in _TREE_FIELDS}}

    @classmethod
    def from_dict(cls, value):
        _exact_fields(value, ("n_features_in",) + _TREE_FIELDS, "Histogram tree")
        n_features = _json_integer(value["n_features_in"], "Tree feature count", 1)
        arrays = {}
        for field in _TREE_FIELDS:
            raw = value[field]
            if not isinstance(raw, list):
                raise ValueError("Serialized tree arrays must be JSON lists")
            if field in _TREE_INTEGERS:
                checked = [_json_integer(x, field, -1) for x in raw]
                arrays[field + "_"] = np.array(checked, dtype=np.int64)
            else:
                arrays[field + "_"] = np.array([_json_number(x, field) for x in raw], dtype=np.float64)
        return cls(**arrays, n_features_in_=n_features)


@dataclass(slots=True)
class _Split:
    feature: int
    boundary: int
    gain: float
    left_rows: np.ndarray
    right_rows: np.ndarray
    left_gradient: float
    right_gradient: float
    left_hessian: float
    right_hessian: float


def _accurate_prefix(values):
    """Expansion-summed prefixes, retaining small terms through cancellation.

    A plain cumsum can turn [1e16, 1, -1e16] into a zero gradient and thereby
    discard an otherwise valid positive split. Non-overlapping partials retain
    every rounding residual, with work proportional to the (usually tiny)
    expansion length rather than recomputing each complete prefix.
    """
    partials = []
    result = np.empty(len(values), dtype=np.float64)
    for index, number in enumerate(values):
        x = float(number)
        end = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            high = x + y
            low = y - (high - x)
            if not math.isfinite(high):
                raise ValueError("Histogram prefix exceeds float64 range")
            if low:
                partials[end] = low
                end += 1
            x = high
        partials[end:] = [x]
        result[index] = _sum(partials, "Histogram prefix")
    return result


def _score_error_bound(gradient, hessian, error_gradient, error_hessian, params):
    """Lipschitz bound for a leaf's score over histogram rounding intervals.

    For the optimized quadratic score, |dS/dG| <= 2|v| and
    |dS/dH| <= v**2. Bound |v| over the entire derivative interval,
    including L1 and the configured step cap. Infinity only causes a direct
    candidate refinement; it never becomes a learned score or widened tie.
    """
    lower_denominator = max(0.0, hessian - error_hessian) + params["lambda_l2"]
    upper_gradient = max(0.0, abs(gradient) + error_gradient - params["lambda_l1"])
    if lower_denominator <= 0:
        magnitude = math.inf
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = float(np.float64(upper_gradient) / lower_denominator)
    if params["max_delta_step"] > 0:
        magnitude = min(magnitude, params["max_delta_step"])
    if not math.isfinite(magnitude):
        return math.inf
    bound = 2 * (magnitude * error_gradient) + magnitude * (magnitude * error_hessian)
    return math.nextafter(bound, math.inf)


def _best_split(rows, histograms, bins, gradients, hessians, parent_gradient,
                parent_hessian, depth, params, numerical_error):
    if (len(rows) < 2 * params["min_data_in_leaf"]
            or (params["max_depth"] is not None and depth >= params["max_depth"])):
        return None
    candidates = []
    error_gradient, error_hessian = numerical_error
    for feature in sorted(histograms):
        histogram = histograms[feature]
        if len(histogram.count) < 2:
            continue
        counts = np.cumsum(histogram.count, dtype=np.int64)
        left_g = _accurate_prefix(histogram.gradient)[:-1]
        left_h = _accurate_prefix(histogram.hessian)[:-1]
        # Independent reverse prefixes avoid subtracting nearly equal totals.
        right_g = _accurate_prefix(histogram.gradient[::-1])[::-1][1:]
        right_h = _accurate_prefix(histogram.hessian[::-1])[::-1][1:]
        left_absolute = _accurate_prefix(np.abs(histogram.gradient))[:-1]
        right_absolute = _accurate_prefix(np.abs(histogram.gradient[::-1]))[::-1][1:]
        for boundary in range(len(left_g)):
            if (counts[boundary] < params["min_data_in_leaf"]
                    or counts[-1] - counts[boundary] < params["min_data_in_leaf"]):
                continue
            lg, rg = float(left_g[boundary]), float(right_g[boundary])
            lh, rh = float(left_h[boundary]), float(right_h[boundary])
            minimum_h = params["min_sum_hessian_in_leaf"]
            # Bin totals are rounded floats: summing [1e16+1] in one bin
            # followed by [-1e16] in another loses information even with an
            # accurate prefix. Refine only cancellation-sensitive candidates
            # from their actual row derivatives before screening gains.
            error_factor = _TIE_TOLERANCE * len(rows)
            sensitive_g = ((left_absolute[boundary] > 0 and abs(lg) <= error_factor * left_absolute[boundary])
                           or (right_absolute[boundary] > 0 and abs(rg) <= error_factor * right_absolute[boundary]))
            sensitive_h = (abs(lh - minimum_h) <= error_factor * max(lh, minimum_h)
                           or abs(rh - minimum_h) <= error_factor * max(rh, minimum_h))
            if sensitive_g or sensitive_h:
                mask = bins[rows, feature] <= boundary
                if sensitive_g:
                    lg = _sum(gradients[rows[mask]], "Refined left gradient")
                    rg = _sum(gradients[rows[~mask]], "Refined right gradient")
                if sensitive_h:
                    lh = _sum(hessians[rows[mask]], "Refined left Hessian")
                    rh = _sum(hessians[rows[~mask]], "Refined right Hessian")
            if (lh < params["min_sum_hessian_in_leaf"]
                    or rh < params["min_sum_hessian_in_leaf"] or lh <= 0 or rh <= 0):
                continue
            gain, tolerance, _, _ = _gain(
                lg, lh, rg, rh,
                parent_gradient, parent_hessian, params)
            uncertainty = (_score_error_bound(lg, lh, error_gradient, error_hessian, params)
                           + _score_error_bound(rg, rh, error_gradient, error_hessian, params)
                           + tolerance)
            upper_gain = math.nextafter(gain + uncertainty, math.inf)
            if upper_gain > params["min_gain_to_split"]:
                candidates.append((-upper_gain, int(feature), boundary))
    # Histogram sums are rounded within bins. Refine every candidate whose
    # conservative upper gain can still beat or tie the best exact row-derived
    # gain. Refining only an approximate winner can select the wrong feature,
    # even when two candidates partition precisely the same training rows.
    heapq.heapify(candidates)
    confirmed = []
    maximum = -math.inf
    while candidates:
        upper_gain, feature, boundary = heapq.heappop(candidates)
        if confirmed and -upper_gain < maximum - _TIE_TOLERANCE * abs(maximum):
            break
        mask = bins[rows, feature] <= boundary
        left, right = rows[mask], rows[~mask]
        lg, rg = _sum(gradients[left], "Left gradient"), _sum(gradients[right], "Right gradient")
        lh, rh = _sum(hessians[left], "Left Hessian"), _sum(hessians[right], "Right Hessian")
        gain, tolerance, _, _ = _gain(lg, lh, rg, rh, parent_gradient, parent_hessian, params)
        if (len(left) >= params["min_data_in_leaf"] and len(right) >= params["min_data_in_leaf"]
                and lh >= params["min_sum_hessian_in_leaf"] and rh >= params["min_sum_hessian_in_leaf"]
                and gain > params["min_gain_to_split"] and gain > tolerance):
            confirmed.append((gain, feature, boundary))
            maximum = max(maximum, gain)
    if not confirmed:
        return None
    # The final tie band is still EXACTLY 64 eps relative to one global maximum.
    # Numerical screening bounds above never widen the selection policy.
    cutoff = maximum - _TIE_TOLERANCE * abs(maximum)
    gain, feature, boundary = min((item for item in confirmed if item[0] >= cutoff),
                                 key=lambda item: (item[1], item[2]))
    mask = bins[rows, feature] <= boundary
    left, right = rows[mask], rows[~mask]
    return _Split(feature, boundary, gain, left, right,
                  _sum(gradients[left], "Left gradient"), _sum(gradients[right], "Right gradient"),
                  _sum(hessians[left], "Left Hessian"), _sum(hessians[right], "Right Hessian"))


def _fit_tree(bins, packed, mapper, layout, gradients, hessians, rows, features, params):
    fields = {field: [] for field in _TREE_FIELDS}

    def append_node(node_rows, gradient, hessian, depth):
        node = len(fields["feature"])
        value, _ = _newton(gradient, hessian, params)
        for name, item in (
            ("children_left", -1), ("children_right", -1), ("feature", -1),
            ("threshold_bin", -1), ("threshold", 0.0), ("n_node_samples", len(node_rows)),
            ("sum_gradient", gradient), ("sum_hessian", hessian), ("value", value),
            ("gain", 0.0), ("depth", depth),
        ):
            fields[name].append(item)
        return node

    def histogram(node_rows):
        return build_histograms(layout, packed, gradients, hessians, rows=node_rows,
                                features=features, n_jobs=params["n_jobs"])

    root_g = _sum(gradients[rows], "Root gradient")
    root_h = _sum(hessians[rows], "Root Hessian")
    # Standard gamma_k = k*eps/(1-k*eps) forward-error envelope. The budget
    # conservatively allows four full-row reductions (bin, unpack, prefix,
    # and at most n-1 ancestor subtractions) plus 16 scalar score operations.
    # Root absolute derivative masses bound every descendant's inherited error.
    operations = 4 * len(rows) + 16
    gamma = operations * _EPS / (1 - operations * _EPS)
    numerical_error = (gamma * _sum(np.abs(gradients[rows]), "Absolute root gradient"), gamma * root_h)
    append_node(rows, root_g, root_h, 0)
    root_histogram = histogram(rows)
    candidate = _best_split(rows, root_histogram, bins, gradients, hessians, root_g, root_h, 0, params, numerical_error)
    leaves = {0: (root_histogram, candidate)} if candidate is not None else {}
    leaf_count = 1
    while leaves and leaf_count < params["num_leaves"]:
        maximum = max(item[1].gain for item in leaves.values())
        cutoff = maximum - _TIE_TOLERANCE * abs(maximum)
        node = min((index for index, item in leaves.items() if item[1].gain >= cutoff),
                   key=lambda index: (leaves[index][1].feature, leaves[index][1].boundary, index))
        parent_histogram, split = leaves.pop(node)
        child_depth = fields["depth"][node] + 1
        left = append_node(split.left_rows, split.left_gradient, split.left_hessian, child_depth)
        right = append_node(split.right_rows, split.right_gradient, split.right_hessian, child_depth)
        fields["children_left"][node], fields["children_right"][node] = left, right
        fields["feature"][node], fields["threshold_bin"][node] = split.feature, split.boundary
        fields["threshold"][node] = float(mapper.thresholds_[split.feature][split.boundary])
        fields["gain"][node] = split.gain
        if len(split.left_rows) <= len(split.right_rows):
            small_node, small_rows, large_node, large_rows = left, split.left_rows, right, split.right_rows
        else:
            small_node, small_rows, large_node, large_rows = right, split.right_rows, left, split.left_rows
        small_histogram = histogram(small_rows)
        try:
            large_histogram = subtract_histograms(parent_histogram, small_histogram)
        except HistogramSubtractionError:
            large_histogram = histogram(large_rows)
        for child, child_rows, child_histogram in (
            (small_node, small_rows, small_histogram), (large_node, large_rows, large_histogram),
        ):
            candidate = _best_split(child_rows, child_histogram, bins, gradients, hessians,
                                    fields["sum_gradient"][child], fields["sum_hessian"][child],
                                    child_depth, params, numerical_error)
            if candidate is not None:
                leaves[child] = (child_histogram, candidate)
        leaf_count += 1
    return HistogramNewtonTree(
        **{field + "_": np.array(values, dtype=np.int64 if field in _TREE_INTEGERS else np.float64)
           for field, values in fields.items()}, n_features_in_=bins.shape[1])


def _importances(trees, active, n_features):
    gains = [(int(feature), float(gain)) for tree in trees[:active]
             for feature, gain in zip(tree.feature_, tree.gain_) if feature >= 0]
    counts = np.zeros(n_features, dtype=np.int64)
    importance = np.zeros(n_features, dtype=np.float64)
    if gains:
        scale = max(gain for _, gain in gains)
        buckets = [[] for _ in range(n_features)]
        for feature, gain in gains:
            counts[feature] += 1
            buckets[feature].append(gain / scale)
        importance = np.array([math.fsum(bucket) for bucket in buckets])
        importance /= math.fsum(importance.tolist())
    return _readonly(importance), _readonly(counts)


@dataclass(frozen=True, slots=True)
class _State:
    params: dict
    classes: np.ndarray
    n_features: int
    n_samples: int
    eligible: np.ndarray
    class_factors: np.ndarray
    class_mass: np.ndarray
    base_score: float
    mapper: BinMapper
    layout: FeatureBundler
    trees: tuple
    seeds: np.ndarray
    sampling: tuple
    features: tuple
    best_iteration: int
    training_loss: np.ndarray
    validation_loss: np.ndarray | None
    stopping_reason: str
    importances: np.ndarray
    split_counts: np.ndarray


class HandcraftedLightGBMClassifier:
    """Deterministic binary histogram Newton boosting with leaf-wise growth.

    Exactly two homogeneous labels and positive effective mass for both are
    required. ``balanced`` class weights use positive-input-weight row counts;
    explicit class weights must cover both observed labels. All training
    weights remain on their original scale because regularization depends on
    their absolute mass. Per-round feature and row draws use independent RNGs.
    ``max_delta_step=0`` disables clipping; unrepresentable Newton steps or
    objective scores still fail explicitly instead of producing a NaN model.

    Explicit ``eval_set=(X_valid, y_valid)`` enables standalone validation.
    Early stopping compares weighted logistic loss against the INITIAL model,
    so ``best_iteration_ == 0`` is valid. All trained split-bearing trees remain
    inspectable; prediction uses only the best prefix. Validation weights use
    the class factors learned on training rows. The integrated experiment
    runner instead disables early stopping and tunes iteration counts inside
    inner CV. Fit is transactional; successful set_params invalidates state.
    """

    def __init__(self, n_estimators=64, learning_rate=0.05, num_leaves=7,
                 max_depth=None, max_bin=63, min_data_in_bin=1, min_data_in_leaf=5,
                 min_sum_hessian_in_leaf=1e-3, lambda_l1=0.0, lambda_l2=1.0,
                 min_gain_to_split=0.0, max_delta_step=10.0, feature_fraction=1.0,
                 bagging_fraction=1.0, data_sample_strategy="bagging", top_rate=0.2,
                 other_rate=0.1, enable_bundle=True, random_state=0, n_jobs=1,
                 class_weight=None, early_stopping_rounds=None, early_stopping_min_delta=0.0):
        for name, value in locals().copy().items():
            if name in _PARAMETERS:
                setattr(self, name, deepcopy(value))
        self.validate_parameters()

    def get_params(self, deep=True):
        return {name: deepcopy(getattr(self, name)) for name in _PARAMETERS}

    def set_params(self, **params):
        unknown = set(params) - set(_PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown LightGBM-style parameters: {sorted(unknown)}")
        candidate = type(self)(**(self.get_params() | params))
        for name in _PARAMETERS:
            setattr(self, name, getattr(candidate, name))
        self.__dict__.pop("_state", None)
        return self

    def validate_parameters(self, n_features=None, n_samples=None):
        _integer(self.n_estimators, "n_estimators", 1)
        _integer(self.num_leaves, "num_leaves", 2)
        if self.max_depth is not None:
            _integer(self.max_depth, "max_depth", 1)
        _integer(self.max_bin, "max_bin", 2, 65535)
        _integer(self.min_data_in_bin, "min_data_in_bin", 1)
        _integer(self.min_data_in_leaf, "min_data_in_leaf", 1)
        if n_features is not None:
            _integer(n_features, "n_features", 1)
        if n_samples is not None:
            _integer(n_samples, "n_samples", 2)
        for name in ("learning_rate",):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be strictly positive")
        for name in ("max_delta_step", "min_sum_hessian_in_leaf", "lambda_l1", "lambda_l2",
                     "min_gain_to_split", "early_stopping_min_delta"):
            _number(getattr(self, name), name)
        for name in ("feature_fraction", "bagging_fraction", "top_rate", "other_rate"):
            if _number(getattr(self, name), name, 0, 1) <= 0:
                raise ValueError(f"{name} must be in (0,1]")
        if self.top_rate >= 1 or self.top_rate + self.other_rate > 1:
            raise ValueError("GOSS rates require top_rate < 1 and top_rate + other_rate <= 1")
        if self.data_sample_strategy not in ("bagging", "goss"):
            raise ValueError("data_sample_strategy must be bagging or goss")
        if self.data_sample_strategy == "goss" and self.bagging_fraction != 1:
            raise ValueError("GOSS requires bagging_fraction=1; two row samplers cannot be combined")
        if not isinstance(self.enable_bundle, (bool, np.bool_)):
            raise ValueError("enable_bundle must be a boolean")
        _integer(self.random_state, "random_state", 0, 2**64 - 1)
        if isinstance(self.n_jobs, (bool, np.bool_)) or not isinstance(self.n_jobs, (int, np.integer)) or (self.n_jobs != -1 and self.n_jobs < 1):
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.early_stopping_rounds is not None:
            _integer(self.early_stopping_rounds, "early_stopping_rounds", 1)
        if self.class_weight is not None:
            if isinstance(self.class_weight, str):
                if self.class_weight != "balanced":
                    raise ValueError("class_weight supports only None, balanced, or an explicit label mapping")
            elif isinstance(self.class_weight, dict) and self.class_weight:
                for label, weight in self.class_weight.items():
                    if not isinstance(label, (str, bool, int, float, np.bool_, np.integer, np.floating)):
                        raise ValueError("Class-weight labels must be scalar labels")
                    if isinstance(label, str) and not label:
                        raise ValueError("Class-weight labels cannot be empty")
                    if isinstance(label, (float, np.floating)) and not math.isfinite(float(label)):
                        raise ValueError("Class-weight labels must be finite")
                    _number(weight, "class_weight")
            else:
                raise ValueError("class_weight must be None, balanced, or a nonempty label mapping")

    def fit(self, X, y, sample_weight=None, *, eval_set=None, eval_sample_weight=None):
        self.validate_parameters()
        values = _matrix(X)
        labels = _labels(y, len(values))
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("This binary classifier requires exactly two observed classes")
        weights = _weights(sample_weight, len(values))
        factors = np.ones(2, dtype=np.float64)
        if isinstance(self.class_weight, dict):
            expected_kind = "i" if classes.dtype.kind in "iu" else classes.dtype.kind
            if (set(self.class_weight) != set(classes.tolist())
                    or any(_label_kind(label) != expected_kind for label in self.class_weight)):
                raise ValueError("Explicit class_weight must cover exactly the two observed classes")
            factors = np.array([self.class_weight[label] for label in classes.tolist()], dtype=np.float64)
        elif self.class_weight == "balanced":
            counts = np.bincount(encoded[weights > 0], minlength=2)
            if np.any(counts == 0):
                raise ValueError("Both classes require positive effective training mass")
            factors = counts.sum() / (2.0 * counts)
        weights = _combine_weights(weights, encoded, factors)
        masses = np.array([_sum(weights[encoded == cls], "Training class mass") for cls in range(2)])
        if np.any(masses <= 0):
            raise ValueError("Both classes require positive effective training mass")
        eligible = np.flatnonzero(weights > 0)
        self.validate_parameters(n_features=values.shape[1], n_samples=len(eligible))
        params = self.get_params()
        for name, value in params.items():
            if isinstance(value, np.generic):
                params[name] = value.item()
        base_score = math.log(float(masses[1])) - math.log(float(masses[0]))
        valid_values = valid_encoded = valid_weights = None
        if eval_set is not None:
            if not isinstance(eval_set, tuple) or len(eval_set) != 2:
                raise ValueError("eval_set must be one explicit (X_valid, y_valid) tuple")
            valid_values = _matrix(eval_set[0], values.shape[1])
            valid_encoded = _encoded_labels(eval_set[1], len(valid_values), classes)
            valid_weights = _combine_weights(_weights(eval_sample_weight, len(valid_values)), valid_encoded, factors)
        elif eval_sample_weight is not None or params["early_stopping_rounds"] is not None:
            raise ValueError("eval_sample_weight and early stopping require an explicit eval_set")
        mapper = BinMapper.fit(values[eligible], max_bin=params["max_bin"], min_data_in_bin=params["min_data_in_bin"])
        full_bins = mapper.transform(values)
        bins = full_bins[eligible]
        layout = FeatureBundler.fit(bins, mapper, enable_bundle=params["enable_bundle"])
        # Excluded rows may conflict with a valid training bundle. Never pack
        # them; prediction uses normal original-feature bins exclusively.
        packed = layout.transform(bins)
        valid_bins = None if valid_values is None else mapper.transform(valid_values)
        raw = np.full(len(values), base_score, dtype=np.float64)
        valid_raw = None if valid_values is None else np.full(len(valid_values), base_score, dtype=np.float64)
        training_loss = [_logloss(raw, encoded, weights)]
        validation_loss = None if valid_raw is None else [_logloss(valid_raw, valid_encoded, valid_weights)]
        trees, seeds, records, feature_sets = [], [], [], []
        best_iteration, stale = 0, 0
        best_loss = None if validation_loss is None else validation_loss[0]
        reason = "n_estimators"
        seed_stream = np.random.SeedSequence(params["random_state"])
        for _ in range(params["n_estimators"]):
            seed = int(seed_stream.spawn(1)[0].generate_state(1, dtype=np.uint64)[0])
            gradients, hessians = logistic_gradients(raw, encoded, weights)
            record = sampling_plan(eligible, gradients, params, seed)
            original_rows = np.array(record["sampled_indices"], dtype=np.int64)
            rows = np.searchsorted(eligible, original_rows).astype(np.int64)
            multipliers = np.array(record["multipliers"], dtype=np.float64)
            local_g, local_h = gradients[eligible].copy(), hessians[eligible].copy()
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                local_g[rows] *= multipliers
                local_h[rows] *= multipliers
            if not np.isfinite(local_g).all() or not np.isfinite(local_h).all():
                raise ValueError("GOSS-corrected derivative sums exceed float64 range")
            features = sampled_features(values.shape[1], params["feature_fraction"], seed)
            tree = _fit_tree(bins, packed, mapper, layout, local_g, local_h, rows, features, params)
            if tree.node_count == 1:
                reason = "no_positive_split_gain"
                break
            with np.errstate(over="ignore", invalid="ignore"):
                raw += params["learning_rate"] * tree.predict_binned(full_bins)
                if valid_raw is not None:
                    valid_raw += params["learning_rate"] * tree.predict_binned(valid_bins)
            if not np.isfinite(raw).all() or (valid_raw is not None and not np.isfinite(valid_raw).all()):
                raise ValueError("Boosted raw scores exceed float64 range")
            trees.append(tree)
            seeds.append(seed)
            records.append(record)
            feature_sets.append(_readonly(features, np.int64))
            training_loss.append(_logloss(raw, encoded, weights))
            if valid_raw is not None:
                validation_loss.append(_logloss(valid_raw, valid_encoded, valid_weights))
            if params["early_stopping_rounds"] is None:
                best_iteration = len(trees)
            elif validation_loss[-1] < best_loss - params["early_stopping_min_delta"]:
                best_loss, best_iteration, stale = validation_loss[-1], len(trees), 0
            else:
                stale += 1
                if stale >= params["early_stopping_rounds"]:
                    reason = "early_stopping"
                    break
        trees = tuple(trees)
        importance, split_counts = _importances(trees, best_iteration, values.shape[1])
        state = _State(
            params=deepcopy(params), classes=_readonly(classes), n_features=values.shape[1], n_samples=len(values),
            eligible=_readonly(eligible, np.int64), class_factors=_readonly(factors), class_mass=_readonly(masses),
            base_score=base_score, mapper=mapper, layout=layout, trees=trees,
            seeds=_readonly(seeds, np.uint64), sampling=tuple(deepcopy(records)), features=tuple(feature_sets),
            best_iteration=best_iteration, training_loss=_readonly(training_loss),
            validation_loss=None if validation_loss is None else _readonly(validation_loss),
            stopping_reason=reason, importances=importance, split_counts=split_counts,
        )
        self._state = state  # the only fitted-state mutation, after all work succeeds
        return self

    def _fitted(self):
        if "_state" not in self.__dict__:
            raise ValueError("HandcraftedLightGBMClassifier is not fitted")
        return self._state

    @property
    def classes_(self):
        return _public_array(self._fitted().classes)

    @property
    def n_classes_(self):
        self._fitted()
        return 2

    @property
    def n_features_in_(self):
        return self._fitted().n_features

    @property
    def n_samples_fit_(self):
        return self._fitted().n_samples

    @property
    def training_eligible_indices_(self):
        return _public_array(self._fitted().eligible)

    @property
    def class_weight_factors_(self):
        return _public_array(self._fitted().class_factors)

    @property
    def training_class_mass_(self):
        return _public_array(self._fitted().class_mass)

    @property
    def bin_mapper_(self):
        return self._fitted().mapper

    @property
    def bundle_layout_(self):
        return self._fitted().layout

    @property
    def base_score_(self):
        return self._fitted().base_score

    @property
    def estimators_(self):
        return self._fitted().trees

    @property
    def iteration_seeds_(self):
        return _public_array(self._fitted().seeds)

    @property
    def sampling_records_(self):
        return tuple(deepcopy(record) for record in self._fitted().sampling)

    @property
    def feature_subsets_(self):
        return tuple(_public_array(values) for values in self._fitted().features)

    @property
    def n_estimators_(self):
        return len(self._fitted().trees)

    @property
    def best_iteration_(self):
        return self._fitted().best_iteration

    @property
    def training_loss_(self):
        return _public_array(self._fitted().training_loss)

    @property
    def validation_loss_(self):
        value = self._fitted().validation_loss
        return None if value is None else _public_array(value)

    @property
    def feature_importances_(self):
        return _public_array(self._fitted().importances)

    @property
    def feature_split_counts_(self):
        return _public_array(self._fitted().split_counts)

    @property
    def stopping_reason_(self):
        return self._fitted().stopping_reason

    def decision_function(self, X):
        state = self._fitted()
        values = _matrix(X, state.n_features, allow_empty=True)
        bins = state.mapper.transform(values)
        result = np.full(len(values), state.base_score, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for tree in state.trees[:state.best_iteration]:
                result += state.params["learning_rate"] * tree.predict_binned(bins)
        if not np.isfinite(result).all():
            raise ValueError("Prediction raw scores exceed float64 range")
        return result

    def predict_proba(self, X):
        raw = self.decision_function(X)
        return np.column_stack((_sigmoid(-raw), _sigmoid(raw)))

    def predict(self, X):
        # Use the actual reported probabilities: even a tiny positive raw score
        # may round both columns to .5, in which case sorted class order wins.
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        state = self._fitted()
        encoded = _encoded_labels(y, len(prediction), state.classes)
        weights = _weights(sample_weight, len(prediction))
        accuracy = _sum((weights / _sum(weights, "Score weight total")) *
                        (prediction == state.classes[encoded]), "Weighted accuracy")
        return min(1.0, max(0.0, accuracy))

    def to_dict(self):
        """Detached non-executable model JSON, with fit-time parameters."""
        state = self._fitted()
        params = deepcopy(state.params)
        if isinstance(params["class_weight"], dict):
            params["class_weight"] = {"labels": state.classes.tolist(),
                                      "weights": [float(params["class_weight"][label]) for label in state.classes.tolist()]}
        kind = {"b": "boolean", "i": "integer", "u": "unsigned_integer", "f": "float", "U": "string"}[state.classes.dtype.kind]
        return {
            "schema_version": MODEL_SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
            "params": params, "classes": {"kind": kind, "values": state.classes.tolist()},
            "n_features_in": state.n_features, "training_n_samples": state.n_samples,
            "training_eligible_indices": state.eligible.tolist(),
            "class_weight_factors": state.class_factors.tolist(), "training_class_mass": state.class_mass.tolist(),
            "base_score": state.base_score, "bin_mapper": state.mapper.to_dict(), "bundle_layout": state.layout.to_dict(),
            "iterations": [{"seed": int(seed), "sampling": deepcopy(record), "feature_indices": features.tolist(),
                            "tree": tree.to_dict()}
                           for seed, record, features, tree in zip(state.seeds, state.sampling, state.features, state.trees)],
            "best_iteration": state.best_iteration, "training_loss": state.training_loss.tolist(),
            "validation_loss": None if state.validation_loss is None else state.validation_loss.tolist(),
            "stopping_reason": state.stopping_reason, "feature_importances": state.importances.tolist(),
            "feature_split_counts": state.split_counts.tolist(),
        }

    @classmethod
    def from_dict(cls, value):
        """Validate predictive structure and numerical invariants, without code execution.

        This validates types, materialized dimension bounds, bins and bundles,
        RNG/sample evidence, topology, derivative conservation, Newton values,
        gains, and active-prefix history. The JSON deliberately omits training
        X/y: proving optimal splits, GOSS gradient ranks, or genuine provenance
        requires the experiment runner's separate source-derived replay.
        """
        return _load_model(cls, value)


def _load_classes(value):
    _exact_fields(value, ("kind", "values"), "Classes")
    kind, labels = value["kind"], value["values"]
    if not isinstance(labels, list) or len(labels) != 2:
        raise ValueError("Binary models require exactly two class labels")
    if kind == "string":
        if any(type(label) is not str or not label for label in labels):
            raise ValueError("Invalid string class labels")
        classes = np.array(labels, dtype=str)
    elif kind == "boolean":
        if any(type(label) is not bool for label in labels):
            raise ValueError("Invalid boolean class labels")
        classes = np.array(labels, dtype=bool)
    elif kind in ("integer", "unsigned_integer"):
        lower, upper = (0, 2**64 - 1) if kind == "unsigned_integer" else (-(2**63), 2**63 - 1)
        classes = np.array([_json_integer(label, "Class label", lower, upper) for label in labels],
                           dtype=np.uint64 if kind == "unsigned_integer" else np.int64)
    elif kind == "float":
        classes = np.array([_json_number(label, "Class label") for label in labels], dtype=np.float64)
    else:
        raise ValueError("Unknown class-label kind")
    if not np.array_equal(classes, np.unique(classes)):
        raise ValueError("Class labels must be sorted and distinct")
    return classes


def _load_params(value, classes):
    _exact_fields(value, _PARAMETERS, "Model parameters")
    params = deepcopy(value)
    integers = {"n_estimators", "num_leaves", "max_depth", "max_bin", "min_data_in_bin", "min_data_in_leaf",
                "random_state", "n_jobs", "early_stopping_rounds"}
    for name, item in params.items():
        if name in integers:
            if item is None and name in ("max_depth", "early_stopping_rounds"):
                continue
            if type(item) is not int:
                raise ValueError(f"Serialized {name} must be an integer")
        elif name == "enable_bundle":
            if type(item) is not bool:
                raise ValueError("Serialized enable_bundle must be a boolean")
        elif name == "data_sample_strategy":
            if type(item) is not str:
                raise ValueError("Serialized data_sample_strategy must be a string")
        elif name != "class_weight":
            _json_number(item, name, 0)
    if isinstance(params["class_weight"], dict):
        mapping = params["class_weight"]
        _exact_fields(mapping, ("labels", "weights"), "Serialized class weights")
        labels, weights = mapping["labels"], mapping["weights"]
        expected = classes.tolist()
        if (not isinstance(labels, list) or len(labels) != 2
                or any(type(a) is not type(b) or a != b for a, b in zip(labels, expected))
                or not isinstance(weights, list) or len(weights) != 2):
            raise ValueError("Serialized class weights must follow the model class order")
        params["class_weight"] = dict(zip(expected, [_json_number(weight, "Class weight", 0) for weight in weights]))
    elif params["class_weight"] is not None and params["class_weight"] != "balanced":
        raise ValueError("Invalid serialized class weights")
    return params


def _load_vector(raw, name, length=None, *, integer=False, minimum=0, maximum=None):
    if not isinstance(raw, list) or (length is not None and len(raw) != length):
        raise ValueError(f"{name} must be a JSON list with its declared length")
    if integer:
        return np.array([_json_integer(x, name, minimum, _INT64_MAX if maximum is None else maximum) for x in raw], dtype=np.int64)
    return np.array([_json_number(x, name, minimum, maximum) for x in raw], dtype=np.float64)


def _validate_tree(tree, mapper, features, sample_count, params):
    if (tree.n_features_in_ != mapper.n_features_in_ or tree.n_node_samples_[0] != sample_count
            or tree.node_count < 3 or tree.node_count > 2 * min(sample_count, params["num_leaves"]) - 1
            or (params["max_depth"] is not None and tree.max_depth > params["max_depth"])):
        raise ValueError("Tree feature/sample counts, size, or depth violate fitted parameters")
    for node in range(tree.node_count):
        expected_value, _ = _newton(float(tree.sum_gradient_[node]), float(tree.sum_hessian_[node]), params)
        if not _close(float(tree.value_[node]), expected_value):
            raise ValueError("Tree node value does not equal its regularized clipped Newton step")
        if node and (tree.n_node_samples_[node] < params["min_data_in_leaf"]
                     or tree.sum_hessian_[node] < params["min_sum_hessian_in_leaf"]):
            raise ValueError("Tree child violates minimum count or Hessian constraints")
        feature = int(tree.feature_[node])
        if feature < 0:
            continue
        boundary = int(tree.threshold_bin_[node])
        if (feature not in features or boundary >= len(mapper.thresholds_[feature])
                or float(tree.threshold_[node]) != float(mapper.thresholds_[feature][boundary])):
            raise ValueError("Tree split differs from its feature draw or learned bin boundary")
        left, right = int(tree.children_left_[node]), int(tree.children_right_[node])
        gain, tolerance, _, _ = _gain(
            float(tree.sum_gradient_[left]), float(tree.sum_hessian_[left]),
            float(tree.sum_gradient_[right]), float(tree.sum_hessian_[right]),
            float(tree.sum_gradient_[node]), float(tree.sum_hessian_[node]), params)
        if gain <= params["min_gain_to_split"] or gain <= tolerance or not _close(float(tree.gain_[node]), gain):
            raise ValueError("Tree gain differs from its objective or lacks a strictly positive improvement")


def _validate_history(params, training, validation, best, reason, n_trees, masses, base_score):
    expected_initial = _logloss(np.array([base_score, base_score]), np.array([0, 1]), masses)
    if not _close(float(training[0]), expected_initial):
        raise ValueError("Initial training loss differs from the weighted base model")
    if reason not in ("n_estimators", "no_positive_split_gain", "early_stopping"):
        raise ValueError("Unknown stopping reason")
    if reason == "n_estimators" and n_trees != params["n_estimators"]:
        raise ValueError("n_estimators stopping requires all configured iterations")
    if reason == "no_positive_split_gain" and n_trees >= params["n_estimators"]:
        raise ValueError("No-gain stopping requires a remaining attempted iteration")
    if params["early_stopping_rounds"] is None:
        if best != n_trees or reason == "early_stopping":
            raise ValueError("A model without early stopping must use every stored tree")
        return
    if validation is None:
        raise ValueError("Early stopping requires a validation history")
    best_loss, expected_best, stale, triggered = float(validation[0]), 0, 0, False
    for index, loss in enumerate(validation[1:], start=1):
        if triggered:
            raise ValueError("History continues after early-stopping patience was exhausted")
        if loss < best_loss - params["early_stopping_min_delta"]:
            best_loss, expected_best, stale = float(loss), index, 0
        else:
            stale += 1
            triggered = stale >= params["early_stopping_rounds"]
    if best != expected_best or triggered != (reason == "early_stopping"):
        raise ValueError("Early-stopping prefix or stopping reason differs from its validation curve")


def _load_model(cls, value):
    _exact_fields(value, _MODEL_FIELDS, "LightGBM-style model")
    if value["schema_version"] != MODEL_SCHEMA_VERSION or value["algorithm_version"] != ALGORITHM_VERSION:
        raise ValueError("Unsupported LightGBM-style model schema or algorithm version")
    n_features = _json_integer(value["n_features_in"], "Feature count", 1)
    n_samples = _json_integer(value["training_n_samples"], "Training sample count", 2)
    # Bind dimensional allocations to actual JSON vectors before creating any
    # n_features- or training_n_samples-sized array from untrusted declarations.
    importance = _load_vector(value["feature_importances"], "Feature importances", n_features, maximum=1)
    split_counts = _load_vector(value["feature_split_counts"], "Feature split counts", n_features, integer=True)
    classes = _load_classes(value["classes"])
    params = _load_params(value["params"], classes)
    model = cls(**params)
    eligible = _load_vector(value["training_eligible_indices"], "Eligible rows", integer=True, maximum=n_samples - 1)
    if len(eligible) < 2 or not np.array_equal(eligible, np.unique(eligible)):
        raise ValueError("Eligible training rows must contain at least two sorted unique indices")
    model.validate_parameters(n_features=n_features, n_samples=len(eligible))
    factors = _load_vector(value["class_weight_factors"], "Class factors", 2)
    masses = _load_vector(value["training_class_mass"], "Training class mass", 2)
    if np.any(factors <= 0) or np.any(masses <= 0) or _sum(masses, "Training mass") > np.finfo(float).max / 4:
        raise ValueError("Both classes require positive numerically representable factors and mass")
    if params["class_weight"] is None and not np.array_equal(factors, np.ones(2)):
        raise ValueError("Unweighted class factors must both equal one")
    if isinstance(params["class_weight"], dict):
        expected = np.array([params["class_weight"][label] for label in classes.tolist()])
        if not np.array_equal(factors, expected):
            raise ValueError("Explicit class factors differ from class_weight")
    elif params["class_weight"] == "balanced":
        counts = len(eligible) / (2 * factors)
        rounded = np.rint(counts)
        if np.any(rounded < 1) or not np.allclose(counts, rounded, rtol=_CHECK_TOLERANCE, atol=0) or _sum(rounded, "Balanced class counts") != len(eligible):
            raise ValueError("Balanced class factors cannot arise from eligible class counts")
    base_score = _json_number(value["base_score"], "Base score")
    expected_base = math.log(float(masses[1])) - math.log(float(masses[0]))
    if not _close(base_score, expected_base):
        raise ValueError("Base score differs from positive weighted class odds")
    mapper = BinMapper.from_dict(value["bin_mapper"])
    layout = FeatureBundler.from_dict(value["bundle_layout"])
    mapper_payload = value["bin_mapper"]
    if (mapper.n_features_in_ != n_features or mapper.n_samples_fit_ != len(eligible)
            or mapper_payload["max_bin"] != params["max_bin"]
            or mapper_payload["min_data_in_bin"] != params["min_data_in_bin"]
            or layout.n_features_in_ != n_features or layout.n_samples_fit_ != len(eligible)
            or value["bundle_layout"]["enable_bundle"] != params["enable_bundle"]
            or not np.array_equal(mapper.n_bins_, layout.n_bins_)
            or not np.array_equal(mapper.zero_bins_, layout.zero_bins_)
            or any(not np.array_equal(a, b) for a, b in zip(mapper.bin_counts_, layout.bin_counts_))):
        raise ValueError("Binner and bundle layout do not match training dimensions or fitted parameters")
    iterations = value["iterations"]
    if not isinstance(iterations, list) or len(iterations) > params["n_estimators"]:
        raise ValueError("Stored iteration count exceeds the configured ensemble")
    expected_seeds = iteration_seed_sequence(params["random_state"], len(iterations))
    trees, records, feature_sets = [], [], []
    for index, item in enumerate(iterations):
        _exact_fields(item, ("seed", "sampling", "feature_indices", "tree"), "Iteration")
        seed = _json_integer(item["seed"], "Iteration seed", 0, 2**64 - 1)
        if seed != int(expected_seeds[index]):
            raise ValueError("Iteration seed differs from the configured deterministic schedule")
        record = item["sampling"]
        _exact_fields(record, ("strategy", "sampled_indices", "multipliers", "top_indices"), "Sampling record")
        if record["strategy"] != params["data_sample_strategy"]:
            raise ValueError("Sampling strategy differs from fitted parameters")
        rows = _load_vector(record["sampled_indices"], "Sampled rows", integer=True, maximum=n_samples - 1)
        multipliers = _load_vector(record["multipliers"], "Sampling multipliers", len(rows), minimum=1)
        top = _load_vector(record["top_indices"], "GOSS top rows", integer=True, maximum=n_samples - 1)
        if (not len(rows) or not np.array_equal(rows, np.unique(rows)) or not np.isin(rows, eligible).all()
                or not np.array_equal(top, np.unique(top)) or not np.isin(top, eligible).all()
                or (record["strategy"] == "bagging" and len(top))):
            raise ValueError("Sampling rows must be sorted unique eligible indices")
        expected_record = _sample_from_top(eligible, params, seed, top)
        if (record["sampled_indices"] != expected_record["sampled_indices"]
                or not np.array_equal(multipliers, expected_record["multipliers"])
                or record["top_indices"] != expected_record["top_indices"]):
            raise ValueError("Sampling rows or corrections differ from the deterministic draw")
        features = _load_vector(item["feature_indices"], "Sampled features", integer=True, maximum=n_features - 1)
        expected_features = sampled_features(n_features, params["feature_fraction"], seed)
        if not np.array_equal(features, expected_features):
            raise ValueError("Feature draw differs from the deterministic RNG schedule")
        tree = HistogramNewtonTree.from_dict(item["tree"])
        _validate_tree(tree, mapper, features, len(rows), params)
        trees.append(tree)
        records.append(deepcopy(record))
        feature_sets.append(_readonly(features, np.int64))
    best = _json_integer(value["best_iteration"], "Best iteration", 0, len(trees))
    training = _load_vector(value["training_loss"], "Training loss", len(trees) + 1)
    validation = None if value["validation_loss"] is None else _load_vector(value["validation_loss"], "Validation loss", len(trees) + 1)
    reason = value["stopping_reason"]
    if type(reason) is not str:
        raise ValueError("Stopping reason must be a string")
    _validate_history(params, training, validation, best, reason, len(trees), masses, base_score)
    expected_importance, expected_counts = _importances(tuple(trees), best, n_features)
    if (not np.allclose(importance, expected_importance, rtol=_CHECK_TOLERANCE, atol=0)
            or not np.array_equal(split_counts, expected_counts)):
        raise ValueError("Feature importance or split counts differ from the active tree prefix")
    model._state = _State(
        params=deepcopy(params), classes=_readonly(classes), n_features=n_features, n_samples=n_samples,
        eligible=_readonly(eligible, np.int64), class_factors=_readonly(factors), class_mass=_readonly(masses),
        base_score=base_score, mapper=mapper, layout=layout, trees=tuple(trees), seeds=_readonly(expected_seeds, np.uint64),
        sampling=tuple(records), features=tuple(feature_sets), best_iteration=best, training_loss=_readonly(training),
        validation_loss=None if validation is None else _readonly(validation), stopping_reason=reason,
        importances=_readonly(importance), split_counts=_readonly(split_counts, np.int64),
    )
    return model
