"""Training-only numeric bins, lossless EFB and Newton histograms.

Only Python and NumPy are used.  The explicit variant uses deterministic rank
targets for bins and *zero-conflict* feature bundles.  Bundling compresses the
training representation; inference always uses the original feature bins, so
new observations may activate several features which were exclusive in train.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Mapping

import numpy as np


BIN_SCHEMA_VERSION = "eeg-lightgbm-binner-1"
EFB_SCHEMA_VERSION = "eeg-lightgbm-efb-1"
MAX_BIN = 65535
_EPS = np.finfo(np.float64).eps
_INT_MAX = np.iinfo(np.int64).max


def _integer(value, name, minimum=0, maximum=_INT_MAX):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _json_integer(value, name, minimum=0, maximum=_INT_MAX):
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    return _integer(value, name, minimum, maximum)


def _fields(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} fields differ from the versioned schema")


def _immutable(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _view(array):
    storage = array
    while isinstance(storage, np.ndarray):
        storage = storage.base
    # Every internal array is backed by immutable bytes.  Reconstructing the
    # public view prevents mutable NumPy shape metadata from leaking back in.
    return np.frombuffer(storage, dtype=array.dtype).reshape(array.shape)


def _real_vector(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric vector")
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
        for v in value
    ):
        raise ValueError(f"{name} cannot coerce booleans or strings into numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float64 values")
    return result


def _counts(value, name, *, allow_zero=True):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{name} must be a nonempty integer vector")
    minimum = 0 if allow_zero else 1
    return np.array([_integer(v, name, minimum) for v in value], dtype=np.int64)


def _features(X, n_features=None, *, allow_empty=False):
    raw = np.asarray(X)
    if raw.ndim != 2 or not raw.shape[1] or (not allow_empty and not raw.shape[0]):
        raise ValueError("Features must be a nonempty two-dimensional matrix")
    if raw.dtype.kind not in "biuf":
        raise ValueError("Features must be real numeric values")
    if raw.dtype.kind in "iu" and np.any((raw > 2**53) | (raw < -(2**53))):
        raise ValueError("Integer features outside exact float64 range must be explicitly represented")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite; missing and categorical values are unsupported")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError("Feature count differs from the fitted bin mapper")
    return values


def _dtype(maximum):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Packed bin IDs exceed the supported integer range")


def _midpoint(lower, upper):
    middle = lower / 2.0 + upper / 2.0
    return float(middle if lower <= middle < upper else lower)


def _fit_column(column, max_bin, min_data_in_bin):
    values, counts = np.unique(column, return_counts=True)
    n = len(column)
    if len(values) <= max_bin and np.all(counts >= min_data_in_bin):
        boundaries = list(range(len(values) - 1))
    else:
        targets = min(max_bin, len(values), max(1, n // min_data_in_bin))
        cumulative = np.cumsum(counts, dtype=np.int64)[:-1]
        boundaries, previous = [], 0
        for k in range(1, targets):
            # Duplicate values never straddle bins.  Some rank targets may be
            # impossible; skipping them preserves all minimum-count bounds.
            lower = previous + min_data_in_bin
            upper = n - (targets - k) * min_data_in_bin
            start = int(np.searchsorted(cumulative, lower, side="left"))
            end = int(np.searchsorted(cumulative, upper, side="right"))
            if start >= end:
                continue
            # Looking on both sides of the *ceiling* catches the closest
            # integer rank even when the fractional target is above 1/2.
            target_ceiling = (k * n + targets - 1) // targets
            nearest = int(np.searchsorted(cumulative, target_ceiling, side="left"))
            options = {max(start, min(end - 1, nearest)), max(start, min(end - 1, nearest - 1))}
            chosen = min(options, key=lambda i: (abs(int(cumulative[i]) * targets - k * n), int(cumulative[i])))
            boundaries.append(chosen)
            previous = int(cumulative[chosen])
    cuts = np.array([_midpoint(float(values[i]), float(values[i + 1])) for i in boundaries])
    original_cumulative = np.cumsum(counts, dtype=np.int64)
    bin_counts = np.diff([0, *(int(original_cumulative[i]) for i in boundaries), n]).astype(np.int64)
    return cuts, bin_counts


@dataclass(frozen=True, slots=True, eq=False)
class BinMapper:
    """Immutable ordered bins fitted from the caller's training rows only."""

    _thresholds: tuple
    _counts: tuple
    max_bin: int
    min_data_in_bin: int
    n_samples_fit_: int

    def __post_init__(self):
        maximum = _integer(self.max_bin, "max_bin", 2, MAX_BIN)
        minimum = _integer(self.min_data_in_bin, "min_data_in_bin", 1)
        n = _integer(self.n_samples_fit_, "n_samples_fit", 1)
        if not isinstance(self._thresholds, tuple) or not self._thresholds or len(self._thresholds) != len(self._counts):
            raise ValueError("Bin mapper feature arrays are incomplete")
        cuts, counts = [], []
        for raw_cut, raw_count in zip(self._thresholds, self._counts):
            cut = _real_vector(raw_cut, "Bin thresholds")
            count = _counts(raw_count, "Training bin counts", allow_zero=False)
            if (len(cut) + 1 != len(count) or len(count) > maximum
                    or np.any(cut[1:] <= cut[:-1]) or sum(map(int, count)) != n
                    or (len(count) > 1 and np.any(count < minimum))):
                raise ValueError("Bin thresholds/counts violate the fitted bin contract")
            cuts.append(_immutable(cut)); counts.append(_immutable(count))
        object.__setattr__(self, "_thresholds", tuple(cuts))
        object.__setattr__(self, "_counts", tuple(counts))
        object.__setattr__(self, "max_bin", maximum)
        object.__setattr__(self, "min_data_in_bin", minimum)
        object.__setattr__(self, "n_samples_fit_", n)

    @classmethod
    def fit(cls, X, *, max_bin=63, min_data_in_bin=1):
        maximum = _integer(max_bin, "max_bin", 2, MAX_BIN)
        minimum = _integer(min_data_in_bin, "min_data_in_bin", 1)
        values = _features(X)
        columns = [_fit_column(values[:, j], maximum, minimum) for j in range(values.shape[1])]
        return cls(tuple(x[0] for x in columns), tuple(x[1] for x in columns), maximum, minimum, len(values))

    @property
    def thresholds_(self):
        return tuple(_view(x) for x in self._thresholds)

    @property
    def bin_counts_(self):
        return tuple(_view(x) for x in self._counts)

    @property
    def n_bins_(self):
        return _immutable([len(x) for x in self._counts], dtype=np.int64)

    @property
    def zero_bins_(self):
        return _immutable([np.searchsorted(x, 0.0, side="left") for x in self._thresholds], dtype=np.int64)

    @property
    def n_features_in_(self):
        return len(self._thresholds)

    def transform(self, X):
        values = _features(X, self.n_features_in_, allow_empty=True)
        result = np.empty(values.shape, dtype=_dtype(max(map(len, self._thresholds))))
        for j, cuts in enumerate(self._thresholds):
            result[:, j] = np.searchsorted(cuts, values[:, j], side="left")
        return result

    def to_dict(self):
        return {"schema_version": BIN_SCHEMA_VERSION, "max_bin": self.max_bin,
                "min_data_in_bin": self.min_data_in_bin, "n_samples_fit": self.n_samples_fit_,
                "features": [{"thresholds": cut.tolist(), "counts": count.tolist(),
                              "zero_bin": int(np.searchsorted(cut, 0.0, side="left"))}
                             for cut, count in zip(self._thresholds, self._counts)]}

    @classmethod
    def from_dict(cls, value):
        _fields(value, ("schema_version", "max_bin", "min_data_in_bin", "n_samples_fit", "features"), "Bin mapper")
        if type(value["schema_version"]) is not str or value["schema_version"] != BIN_SCHEMA_VERSION:
            raise ValueError("Unsupported bin mapper schema")
        _json_integer(value["max_bin"], "max_bin", 2, MAX_BIN)
        _json_integer(value["min_data_in_bin"], "min_data_in_bin", 1)
        _json_integer(value["n_samples_fit"], "n_samples_fit", 1)
        features = value["features"]
        if not isinstance(features, list) or not features:
            raise ValueError("Bin mapper needs materialized feature records")
        cuts, counts, zero_bins = [], [], []
        for feature in features:
            _fields(feature, ("thresholds", "counts", "zero_bin"), "Bin feature")
            if not isinstance(feature["thresholds"], list) or not isinstance(feature["counts"], list):
                raise ValueError("Serialized bins must be materialized lists")
            if any(type(cut) not in (int, float) for cut in feature["thresholds"]):
                raise ValueError("Serialized thresholds must be JSON numbers")
            for count in feature["counts"]:
                _json_integer(count, "Training bin count", 1)
            cuts.append(feature["thresholds"]); counts.append(feature["counts"])
            zero_bins.append(_json_integer(feature["zero_bin"], "zero_bin"))
        mapper = cls(tuple(cuts), tuple(counts), value["max_bin"], value["min_data_in_bin"], value["n_samples_fit"])
        if mapper.zero_bins_.tolist() != zero_bins:
            raise ValueError("Serialized zero bins do not match thresholds")
        return mapper


def _normal_bins(value, n_bins, *, allow_empty=True):
    bins = np.asarray(value)
    if (bins.ndim != 2 or bins.shape[1] != len(n_bins) or bins.dtype.kind not in "iu"
            or (not allow_empty and not len(bins)) or np.any(bins < 0)
            or np.any(bins >= np.asarray(n_bins)[None, :])):
        raise ValueError("Normal bin matrix has invalid shape, type or bin IDs")
    return bins


@dataclass(frozen=True, slots=True, eq=False)
class FeatureBundler:
    """Lossless zero-conflict packing of nondefault *bin* activations."""

    _counts: tuple
    _zero_bins: np.ndarray
    _bundles: tuple
    enable_bundle: bool
    n_samples_fit_: int

    def __post_init__(self):
        enabled = _boolean(self.enable_bundle, "enable_bundle")
        n = _integer(self.n_samples_fit_, "n_samples_fit", 1)
        if not isinstance(self._counts, tuple) or not self._counts:
            raise ValueError("EFB feature bin counts are missing")
        counts = tuple(_counts(row, "EFB training bin counts", allow_zero=False) for row in self._counts)
        if any(len(row) > MAX_BIN or sum(map(int, row)) != n for row in counts):
            raise ValueError("EFB dimensions must be bound to materialized training bin counts")
        zeros = _counts(self._zero_bins, "EFB zero bins")
        if len(zeros) != len(counts) or any(z >= len(row) for z, row in zip(zeros, counts)):
            raise ValueError("EFB default bin IDs differ from feature bin dimensions")
        if not isinstance(self._bundles, tuple) or not self._bundles:
            raise ValueError("EFB bundles must be a nonempty tuple")
        groups = []
        for group in self._bundles:
            if not isinstance(group, tuple) or not group:
                raise ValueError("An EFB group must be a nonempty tuple")
            parsed = tuple(_integer(f, "EFB feature", 0, len(counts) - 1) for f in group)
            if tuple(sorted(parsed)) != parsed or len(set(parsed)) != len(parsed):
                raise ValueError("EFB features must be ordered and unique within each group")
            groups.append(parsed)
        groups = tuple(groups)
        if groups != tuple(sorted(groups)) or sorted(f for group in groups for f in group) != list(range(len(counts))):
            raise ValueError("EFB groups must uniquely and canonically partition every feature")
        if not enabled and groups != tuple((i,) for i in range(len(counts))):
            raise ValueError("Disabled EFB must retain one group per feature")
        object.__setattr__(self, "_counts", tuple(_immutable(row) for row in counts))
        object.__setattr__(self, "_zero_bins", _immutable(zeros))
        object.__setattr__(self, "_bundles", groups)
        object.__setattr__(self, "enable_bundle", enabled)
        object.__setattr__(self, "n_samples_fit_", n)

    @classmethod
    def fit(cls, bins, mapper: BinMapper, *, enable_bundle=True):
        enabled = _boolean(enable_bundle, "enable_bundle")
        values = _normal_bins(bins, mapper.n_bins_, allow_empty=False)
        if len(values) != mapper.n_samples_fit_:
            raise ValueError("EFB training rows must match the fitted bin mapper")
        counts, zeros = mapper.bin_counts_, mapper.zero_bins_
        for j, count in enumerate(counts):
            observed = np.bincount(values[:, j].astype(np.int64), minlength=len(count))
            if not np.array_equal(observed, count):
                raise ValueError("EFB training bins differ from the mapper's training counts")
        if not enabled:
            groups = [(j,) for j in range(values.shape[1])]
        else:
            active_counts = np.count_nonzero(values != zeros[None, :], axis=0)
            ordered = sorted(range(values.shape[1]), key=lambda j: (-int(active_counts[j]), j))
            groups, occupied = [], []
            for j in ordered:
                active = values[:, j] != zeros[j]
                for group, used in zip(groups, occupied):
                    if not np.any(active & used):
                        group.append(j); used |= active
                        break
                else:
                    groups.append([j]); occupied.append(active.copy())
            groups = sorted(tuple(sorted(group)) for group in groups)
        return cls(tuple(counts), zeros, tuple(groups), enabled, len(values))

    @property
    def n_features_in_(self):
        return len(self._counts)

    @property
    def n_bundles_(self):
        return len(self._bundles)

    @property
    def bundles_(self):
        return self._bundles

    @property
    def bin_counts_(self):
        return tuple(_view(x) for x in self._counts)

    @property
    def n_bins_(self):
        return _immutable([len(x) for x in self._counts], dtype=np.int64)

    @property
    def zero_bins_(self):
        return _view(self._zero_bins)

    @property
    def packed_bin_counts_(self):
        return _immutable([1 + sum(len(self._counts[j]) - 1 for j in group) for group in self._bundles], dtype=np.int64)

    @property
    def training_active_counts_(self):
        return _immutable([self.n_samples_fit_ - int(count[zero])
                           for count, zero in zip(self._counts, self._zero_bins)], dtype=np.int64)

    def transform(self, bins):
        values = _normal_bins(bins, self.n_bins_)
        result = np.zeros((len(values), self.n_bundles_), dtype=_dtype(int(self.packed_bin_counts_.max()) - 1))
        for b, group in enumerate(self._bundles):
            occupied, offset = np.zeros(len(values), dtype=bool), 0
            for j in group:
                zero = int(self._zero_bins[j])
                active = values[:, j] != zero
                if np.any(occupied & active):
                    raise ValueError("EFB cannot encode overlapping nondefault feature bins")
                local = values[active, j].astype(np.int64)
                result[active, b] = offset + 1 + local - (local > zero)
                occupied |= active
                offset += len(self._counts[j]) - 1
        return result

    def to_dict(self):
        return {"schema_version": EFB_SCHEMA_VERSION, "enable_bundle": self.enable_bundle,
                "n_samples_fit": self.n_samples_fit_, "feature_bin_counts": [x.tolist() for x in self._counts],
                "zero_bins": self._zero_bins.tolist(), "bundles": [list(group) for group in self._bundles]}

    @classmethod
    def from_dict(cls, value):
        _fields(value, ("schema_version", "enable_bundle", "n_samples_fit", "feature_bin_counts", "zero_bins", "bundles"), "EFB")
        if type(value["schema_version"]) is not str or value["schema_version"] != EFB_SCHEMA_VERSION:
            raise ValueError("Unsupported EFB schema")
        if (not isinstance(value["feature_bin_counts"], list) or not value["feature_bin_counts"]
                or any(not isinstance(row, list) for row in value["feature_bin_counts"])
                or not isinstance(value["zero_bins"], list) or not isinstance(value["bundles"], list)
                or any(not isinstance(group, list) for group in value["bundles"])):
            raise ValueError("EFB payload needs materialized feature/group lists")
        if type(value["enable_bundle"]) is not bool:
            raise ValueError("Serialized enable_bundle must be a JSON boolean")
        _json_integer(value["n_samples_fit"], "n_samples_fit", 1)
        for row in value["feature_bin_counts"]:
            for count in row:
                _json_integer(count, "EFB training bin count", 1)
        for zero in value["zero_bins"]:
            _json_integer(zero, "EFB zero bin")
        for group in value["bundles"]:
            for feature in group:
                _json_integer(feature, "EFB feature")
        return cls(tuple(value["feature_bin_counts"]), value["zero_bins"],
                   tuple(tuple(group) for group in value["bundles"]), value["enable_bundle"], value["n_samples_fit"])


@dataclass(frozen=True, slots=True, eq=False)
class Histogram:
    """Counts and first/second derivatives in original monotonic bin order."""

    _count: np.ndarray
    _gradient: np.ndarray
    _hessian: np.ndarray

    def __post_init__(self):
        count = _counts(self._count, "Histogram counts")
        gradient = _real_vector(self._gradient, "Histogram gradients")
        hessian = _real_vector(self._hessian, "Histogram Hessians")
        if (sum(map(int, count)) > _INT_MAX or count.shape != gradient.shape
                or count.shape != hessian.shape or np.any(hessian < 0)
                or np.any((count == 0) & ((gradient != 0) | (hessian != 0)))):
            raise ValueError("Histogram count/derivative arrays are inconsistent")
        object.__setattr__(self, "_count", _immutable(count))
        object.__setattr__(self, "_gradient", _immutable(gradient))
        object.__setattr__(self, "_hessian", _immutable(hessian))

    @property
    def count(self):
        return _view(self._count)

    @property
    def gradient(self):
        return _view(self._gradient)

    @property
    def hessian(self):
        return _view(self._hessian)


def _sum(values, name):
    try:
        result = math.fsum(values)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds float64 mass range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not a finite mass")
    return result


def _indices(values, limit, name, *, allow_empty=True):
    raw = np.asarray(values)
    if raw.ndim != 1 or (len(raw) and raw.dtype.kind not in "iu") or (not allow_empty and not len(raw)):
        raise ValueError(f"{name} must be integer indices")
    # Preserve scalar types from a list: np.asarray([0, True]) silently turns
    # the bool into an integer and would otherwise bypass strict validation.
    result = np.array([_integer(i, name, 0, limit - 1) for i in values], dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def build_histograms(layout: FeatureBundler, packed, gradients, hessians, rows=None, features=None, n_jobs=1):
    """Aggregate packed codes once per bundle, then recover feature defaults.

    Selected row indices refer to the complete packed matrix and aligned g/h
    vectors.  Counts are actual retained rows, including under GOSS weighting.
    A stable grouping pass and ``math.fsum`` avoid order-dependent derivative
    accumulation drift within bins.  This costs O(r log r) per needed bundle,
    where r is the selected row count; temporary storage is O(r + packed_bins).
    Severe floating cancellation uses an accurate direct default-bin sum.
    """
    codes = _normal_bins(packed, layout.packed_bin_counts_)
    g, h = _real_vector(gradients, "Gradients"), _real_vector(hessians, "Hessians")
    if len(g) != len(codes) or len(h) != len(codes) or np.any(h < 0):
        raise ValueError("Gradient/Hessian vectors must match packed rows and Hessians must be nonnegative")
    selected = np.arange(len(codes), dtype=np.int64) if rows is None else _indices(rows, len(codes), "Histogram rows")
    chosen = np.arange(layout.n_features_in_, dtype=np.int64) if features is None else _indices(features, layout.n_features_in_, "Histogram features")
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)) or (n_jobs != -1 and n_jobs < 1):
        raise ValueError("n_jobs must be -1 or a positive integer")
    if not len(chosen):
        return {}
    wanted = set(map(int, chosen))
    bundles = [(b, group) for b, group in enumerate(layout._bundles) if wanted.intersection(group)]
    selected_g, selected_h = g[selected], h[selected]
    total_g, total_h = _sum(selected_g.tolist(), "Gradient total"), _sum(selected_h.tolist(), "Hessian total")
    abs_g = _sum(np.abs(selected_g).tolist(), "Absolute gradient total")
    roundoff = 64 * _EPS * max(1, len(selected))

    def build(item):
        b, group = item
        code = codes[selected, b].astype(np.int64)
        size = int(layout.packed_bin_counts_[b])
        count = np.bincount(code, minlength=size)
        gradient, hessian = np.zeros(size), np.zeros(size)
        if len(code):
            order = np.argsort(code, kind="stable")
            ordered_codes = code[order]
            ordered_g, ordered_h = selected_g[order], selected_h[order]
            stops = np.r_[0, np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1, len(code)]
            for start, stop in zip(stops[:-1], stops[1:]):
                index = int(ordered_codes[start])
                gradient[index] = _sum(ordered_g[start:stop], "Bin gradient")
                hessian[index] = _sum(ordered_h[start:stop], "Bin Hessian")
        out, offset = {}, 0
        for feature in group:
            n_bins, zero = len(layout._counts[feature]), int(layout._zero_bins[feature])
            if feature in wanted:
                normal = np.arange(n_bins) != zero
                part = slice(offset + 1, offset + n_bins)
                c, a, d = np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins), np.zeros(n_bins)
                c[normal], a[normal], d[normal] = count[part], gradient[part], hessian[part]
                c[zero] = len(selected) - sum(map(int, c[normal]))
                a[zero] = total_g - _sum(a[normal].tolist(), "Nondefault gradient total")
                d[zero] = total_h - _sum(d[normal].tolist(), "Nondefault Hessian total")
                if c[zero] == 0:
                    # There are exactly no default observations; numerical
                    # subtraction residuals cannot be a learned statistic.
                    a[zero] = d[zero] = 0.0
                else:
                    default = (code <= offset) | (code >= offset + n_bins)
                    if abs(a[zero]) <= roundoff * abs_g or not math.isfinite(a[zero]):
                        a[zero] = _sum(selected_g[default].tolist(), "Default-bin gradient")
                    if d[zero] <= roundoff * total_h or not math.isfinite(d[zero]):
                        d[zero] = _sum(selected_h[default].tolist(), "Default-bin Hessian")
                out[feature] = Histogram(c, a, d)
            offset += n_bins - 1
        return out

    workers = min(len(bundles), (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs))
    if workers <= 1:
        parts = [build(item) for item in bundles]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="handcrafted-hist") as pool:
            parts = list(pool.map(build, bundles))
    merged = {feature: histogram for part in parts for feature, histogram in part.items()}
    return {feature: merged[feature] for feature in sorted(wanted)}


class HistogramSubtractionError(ValueError):
    """Subtraction lost a trustworthy statistic; build the other child directly."""


def subtract_histograms(parent: Mapping[int, Histogram], child: Mapping[int, Histogram]):
    """Subtract a directly built smaller child, with safe direct-fallback signals."""
    if set(parent) != set(child):
        raise ValueError("Histogram subtraction needs identical feature keys")
    result = {}
    for feature in sorted(parent):
        p, c = parent[feature], child[feature]
        if p._count.shape != c._count.shape or np.any(c._count > p._count):
            raise ValueError("Child histogram counts cannot exceed the parent's")
        count = p._count - c._count
        with np.errstate(over="ignore", invalid="ignore"):
            gradient, hessian = p._gradient - c._gradient, p._hessian - c._hessian
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise HistogramSubtractionError("Histogram subtraction overflowed")
        factor = 64 * _EPS * max(1, sum(map(int, p._count)))
        if factor >= 1:
            raise HistogramSubtractionError("Histogram row mass is too large to bound subtraction error")
        g_tolerance = factor * np.maximum(np.abs(p._gradient), np.abs(c._gradient))
        h_tolerance = factor * np.maximum(p._hessian, c._hessian)
        empty = count == 0
        if np.any(empty & ((np.abs(gradient) > g_tolerance) | (np.abs(hessian) > h_tolerance))):
            raise HistogramSubtractionError("Empty child bins have inconsistent derivative mass")
        positive = ~empty
        unsafe_gradient = positive & (g_tolerance > 0) & (np.abs(gradient) <= g_tolerance)
        unsafe_hessian = positive & ((hessian < 0) | ((h_tolerance > 0) & (hessian <= h_tolerance)))
        if np.any(unsafe_gradient | unsafe_hessian):
            raise HistogramSubtractionError("Histogram subtraction has unsafe gradient/Hessian cancellation")
        gradient[empty] = hessian[empty] = 0.0
        result[feature] = Histogram(count, gradient, hessian)
    return result
