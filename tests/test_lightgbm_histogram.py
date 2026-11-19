"""Independent bin/packing/mass checks for the handwritten boosting backend."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math

import numpy as np
import pytest

from eeg_ma.lightgbm_histogram import (
    BinMapper, FeatureBundler, Histogram, HistogramSubtractionError,
    build_histograms, subtract_histograms,
)


def _sparse_matrix():
    X = np.zeros((12, 5))
    X[[0, 1], 0] = [-3, 3]
    X[[2, 3], 1] = [-2, 2]
    X[[4, 5], 2] = [-1, 1]
    X[[0, 6, 7], 3] = [-1, 2, 3]
    X[:, 4] = 4  # A nonzero constant still has one default bin.
    return X


def _layout(X=None, *, enable_bundle=True):
    X = _sparse_matrix() if X is None else X
    mapper = BinMapper.fit(X)
    bins = mapper.transform(X)
    layout = FeatureBundler.fit(bins, mapper, enable_bundle=enable_bundle)
    return mapper, bins, layout, layout.transform(bins)


def _direct_histograms(bins, n_bins, gradients, hessians, rows):
    """Separate masks and accurate sums; no bundled/default reconstruction."""
    result = {}
    for feature, count in enumerate(n_bins):
        counts, grad, hess = [], [], []
        for bin_id in range(count):
            members = rows[bins[rows, feature] == bin_id]
            counts.append(len(members))
            grad.append(math.fsum(float(gradients[row]) for row in members))
            hess.append(math.fsum(float(hessians[row]) for row in members))
        result[feature] = (counts, grad, hess)
    return result


def _assert_histograms_equal(first, second):
    assert list(first) == list(second)
    for feature in first:
        for attribute in ("count", "gradient", "hessian"):
            np.testing.assert_array_equal(getattr(first[feature], attribute), getattr(second[feature], attribute))


def test_rank_targets_choose_nearest_admissible_counts_and_lower_ties():
    mapper = BinMapper.fit(np.arange(8.0)[:, None], max_bin=3)
    # Rank targets 8/3 and 16/3 are nearest to 3 and 5, respectively.
    np.testing.assert_array_equal(mapper.thresholds_[0], [2.5, 4.5])
    np.testing.assert_array_equal(mapper.bin_counts_[0], [3, 2, 3])
    tied = BinMapper.fit(np.arange(5.0)[:, None], max_bin=2)
    np.testing.assert_array_equal(tied.thresholds_[0], [1.5])
    np.testing.assert_array_equal(tied.bin_counts_[0], [2, 3])


def test_unique_values_are_preserved_when_bin_budget_and_counts_allow():
    X = np.array([[9, 0], [-3, 0], [1, 0], [-3, 0], [9, 0]], dtype=float)
    mapper = BinMapper.fit(X, max_bin=3)
    np.testing.assert_array_equal(mapper.transform(X), [[2, 0], [0, 0], [1, 0], [0, 0], [2, 0]])
    np.testing.assert_array_equal(mapper.bin_counts_[0], [2, 1, 2])
    assert mapper.n_bins_.tolist() == [3, 1]


@pytest.mark.parametrize("minimum", [1, 2, 3, 5, 11, 100])
def test_duplicate_aware_bins_respect_minimum_and_never_split_equal_values(minimum):
    X = np.array([0] * 9 + [1] + [2] * 2 + [3] * 7 + [4] + [5] * 3, dtype=float)[:, None]
    mapper = BinMapper.fit(X, max_bin=5, min_data_in_bin=minimum)
    bins = mapper.transform(X)[:, 0]
    counts = np.bincount(bins)
    np.testing.assert_array_equal(counts, mapper.bin_counts_[0])
    assert len(counts) <= 5 and counts.sum() == len(X)
    assert len(counts) == 1 or np.all(counts >= minimum)
    assert np.all(np.diff(bins.astype(int)) >= 0)
    for value in np.unique(X):
        assert len(np.unique(bins[X[:, 0] == value])) == 1


@pytest.mark.parametrize("lower, upper", [
    (1.0, np.nextafter(1.0, np.inf)),
    (np.nextafter(np.finfo(float).max, 0), np.finfo(float).max),
    (-np.finfo(float).max, np.nextafter(-np.finfo(float).max, 0)),
    (-np.finfo(float).max, np.finfo(float).max),
    (0.0, np.nextafter(0.0, 1.0)),
    (-np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
])
def test_float_boundaries_are_finite_and_preserve_exact_partition(lower, upper):
    X = np.array([[lower], [lower], [upper], [upper]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mapper = BinMapper.fit(X)
        threshold = mapper.thresholds_[0][0]
        assert math.isfinite(threshold) and lower <= threshold < upper
        np.testing.assert_array_equal(mapper.transform(X)[:, 0], [0, 0, 1, 1])
        assert mapper.transform([[threshold]])[0, 0] == 0


def test_training_only_bins_are_row_order_invariant_and_validation_never_refits():
    rng = np.random.default_rng(822)
    X = rng.integers(-4, 9, size=(43, 6)).astype(float)
    mapper = BinMapper.fit(X, max_bin=5, min_data_in_bin=4)
    expected = mapper.to_dict()
    assert BinMapper.fit(X[rng.permutation(len(X))], max_bin=5, min_data_in_bin=4).to_dict() == expected
    shifted = mapper.transform(np.full((4, 6), 1e100))
    np.testing.assert_array_equal(shifted, np.tile(mapper.n_bins_ - 1, (4, 1)))
    assert mapper.to_dict() == expected
    assert mapper.transform(np.empty((0, 6))).shape == (0, 6)


def test_zero_conflict_bundles_compress_negative_positive_and_constant_features():
    mapper, bins, layout, packed = _layout()
    assert mapper.zero_bins_.tolist() == [1, 1, 1, 1, 0]
    assert layout.n_bundles_ == 2 < mapper.n_features_in_
    assert packed.shape == (len(bins), 2)
    assert packed.dtype == np.uint8
    assert layout.training_active_counts_.tolist() == [2, 2, 2, 3, 0]
    for group in layout.bundles_:
        assert np.all(np.sum(bins[:, group] != mapper.zero_bins_[list(group)], axis=1) <= 1)
    shuffled = np.array([11, 0, 9, 3, 5, 1, 2, 4, 6, 7, 8, 10])
    reordered = FeatureBundler.fit(bins[shuffled], mapper)
    assert reordered.to_dict() == layout.to_dict()
    np.testing.assert_array_equal(reordered.transform(bins[shuffled]), packed[shuffled])


def test_overlap_is_never_discarded_and_unseen_overlap_remains_valid_for_inference():
    X = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=float)
    mapper, _, layout, _ = _layout(X)
    assert layout.bundles_ == ((0, 1),)
    future = mapper.transform([[1, 1], [-1, -1]])
    np.testing.assert_array_equal(future, [[2, 2], [0, 0]])
    with pytest.raises(ValueError, match="overlapping"):
        layout.transform(future)
    overlapping_train = np.array([[-1, 0], [1, 1], [0, -1], [0, 1], [0, 0]], dtype=float)
    _, _, independent, _ = _layout(overlapping_train)
    assert independent.n_bundles_ == 2


@pytest.mark.parametrize("seed", range(5))
def test_histograms_equal_independent_bin_masks_and_thread_counts(seed):
    mapper, bins, layout, packed = _layout()
    rng = np.random.default_rng(seed)
    # Dyadic masses make all oracle sums exact, including negative gradients.
    g = rng.integers(-12, 13, len(bins)).astype(float) / 8
    h = rng.integers(0, 13, len(bins)).astype(float) / 16
    rows = rng.choice(len(bins), size=9, replace=False)
    expected = _direct_histograms(bins, mapper.n_bins_, g, h, rows)
    actual = build_histograms(layout, packed, g, h, rows=rows, n_jobs=1)
    for feature, (count, gradient, hessian) in expected.items():
        np.testing.assert_array_equal(actual[feature].count, count)
        np.testing.assert_array_equal(actual[feature].gradient, gradient)
        np.testing.assert_array_equal(actual[feature].hessian, hessian)
    for n_jobs in (2, -1):
        _assert_histograms_equal(actual, build_histograms(layout, packed, g, h, rows=rows, n_jobs=n_jobs))


def test_bundled_and_plain_histograms_are_bit_identical_with_feature_subsampling():
    mapper, bins, layout, packed = _layout()
    plain = FeatureBundler.fit(bins, mapper, enable_bundle=False)
    rng = np.random.default_rng(91)
    g, h = rng.normal(size=len(bins)), rng.uniform(0.01, 1, len(bins))
    rows = np.array([11, 8, 6, 4, 2, 0])
    kwargs = dict(rows=rows, features=[4, 0, 3], n_jobs=2)
    bundled = build_histograms(layout, packed, g, h, **kwargs)
    assert list(bundled) == [0, 3, 4]
    _assert_histograms_equal(bundled, build_histograms(plain, plain.transform(bins), g, h, **kwargs))
    assert build_histograms(layout, packed, g, h, features=[]) == {}
    empty = build_histograms(layout, packed, g, h, rows=[])
    assert all(np.all(hist.count == 0) and np.all(hist.gradient == 0) and np.all(hist.hessian == 0)
               for hist in empty.values())


def test_accurate_recovery_preserves_small_default_mass_and_signed_cancellation():
    # Default rows have gradient/Hessian 1, below the ulp of the active mass.
    X = np.array([[0], [1], [1], [1]], dtype=float)
    _, _, layout, packed = _layout(X)
    hist = build_histograms(layout, packed, [1, 1e16, 1, -1e16], [1, 1e16, 1, 1])[0]
    np.testing.assert_array_equal(hist.gradient, [1, 1])
    assert hist.hessian[0] == 1
    assert hist.count.tolist() == [1, 3]


def test_positive_bin_masses_do_not_lose_repeated_small_terms_after_a_large_one():
    # Ordinary weighted bincount loses every trailing 1 after 1e16.  This
    # would incorrectly reject a child which exactly meets a Hessian minimum.
    n = 201
    X = np.repeat([0., 1.], n)[:, None]
    _, _, layout, packed = _layout(X)
    g = np.tile(np.r_[2e16, np.full(n - 1, 2.)], 2)
    h = np.tile(np.r_[1e16, np.ones(n - 1)], 2)
    histogram = build_histograms(layout, packed, g, h)[0]
    np.testing.assert_array_equal(histogram.gradient, [2e16 + 400, 2e16 + 400])
    np.testing.assert_array_equal(histogram.hessian, [1e16 + 200, 1e16 + 200])
    rows = np.arange(len(X) - 1, -1, -1)
    _assert_histograms_equal({0: histogram}, build_histograms(layout, packed, g, h, rows=rows))


def test_subtraction_matches_direct_sibling_and_detects_lost_small_mass():
    mapper, bins, layout, packed = _layout()
    g, h = np.arange(1., 13.), np.arange(1., 13.) / 4
    parent = build_histograms(layout, packed, g, h)
    child_rows = np.array([0, 3, 5])
    child = build_histograms(layout, packed, g, h, rows=child_rows)
    sibling = build_histograms(layout, packed, g, h, rows=np.array([1, 2, 4, 6, 7, 8, 9, 10, 11]))
    _assert_histograms_equal(subtract_histograms(parent, child), sibling)
    p = {0: Histogram([2], [1e16], [1e16])}
    c = {0: Histogram([1], [1e16], [1e16])}
    with pytest.raises(HistogramSubtractionError, match="cancellation"):
        subtract_histograms(p, c)
    # Empty bins are known exactly, so tiny residuals must become exact zeros.
    empty = subtract_histograms({0: Histogram([1], [1.], [1.])},
                                {0: Histogram([1], [np.nextafter(1., 0)], [1.])})
    assert empty[0].gradient.tolist() == [0.]


def test_subtraction_overflow_and_inconsistent_empty_bins_signal_fallback():
    huge = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(HistogramSubtractionError, match="overflow"):
            subtract_histograms({0: Histogram([2], [huge], [1])},
                                {0: Histogram([1], [-huge], [0])})
    with pytest.raises(HistogramSubtractionError, match="inconsistent"):
        subtract_histograms({0: Histogram([1], [2], [1])}, {0: Histogram([1], [1], [1])})
    with pytest.raises(ValueError, match="counts"):
        subtract_histograms({0: Histogram([1], [1], [1])}, {0: Histogram([2], [1], [1])})


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning"
)
def test_json_roundtrips_are_exact_and_public_state_is_immutable():
    mapper, bins, layout, packed = _layout()
    for original, loader in ((mapper, BinMapper.from_dict), (layout, FeatureBundler.from_dict)):
        restored = loader(json.loads(json.dumps(original.to_dict(), allow_nan=False)))
        assert restored.to_dict() == original.to_dict()
    np.testing.assert_array_equal(BinMapper.from_dict(mapper.to_dict()).transform(_sparse_matrix()), bins)
    np.testing.assert_array_equal(FeatureBundler.from_dict(layout.to_dict()).transform(bins), packed)
    with pytest.raises(FrozenInstanceError):
        mapper.max_bin = 12
    for array in [mapper.thresholds_[0], mapper.bin_counts_[0], mapper.n_bins_, mapper.zero_bins_,
                  layout.bin_counts_[0], layout.zero_bins_, layout.packed_bin_counts_, layout.training_active_counts_]:
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 7
        array.shape = (array.size, 1)
    assert mapper.thresholds_[0].ndim == 1 and mapper.bin_counts_[0].ndim == 1
    assert layout.zero_bins_.ndim == 1 and layout.packed_bin_counts_.ndim == 1
    hist = Histogram([1, 2], [1., -1.], [0.5, 0.75])
    view = hist.gradient
    view.shape = (2, 1)
    assert hist.gradient.shape == (2,)
    with pytest.raises(ValueError):
        hist.gradient.setflags(write=True)


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version="future"),
    lambda d: d.update(extra=1),
    lambda d: d.update(max_bin=True),
    lambda d: d.update(max_bin=np.int64(63)),
    lambda d: d.update(max_bin=65536),
    lambda d: d.update(min_data_in_bin=0),
    lambda d: d.update(n_samples_fit=10**100),
    lambda d: d["features"][0].update(counts=[10**100]),
    lambda d: d["features"][0].update(counts=[True, 1, 10]),
    lambda d: d["features"][0].update(thresholds=[0., 0.]),
    lambda d: d["features"][0].update(thresholds=[-1., float("nan")]),
    lambda d: d["features"][0].update(thresholds=["-1", 1.]),
    lambda d: d["features"][0].update(thresholds=[np.float64(-1), 1.]),
    lambda d: d["features"][0].update(zero_bin=0),
])
def test_bin_mapper_rejects_invalid_json_and_unbounded_scalar_claims(change):
    payload = _layout()[0].to_dict()
    change(payload)
    with pytest.raises(ValueError):
        BinMapper.from_dict(payload)


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version="future"),
    lambda d: d.update(enable_bundle="true"),
    lambda d: d.update(enable_bundle=np.bool_(True)),
    lambda d: d.update(n_samples_fit=10**100),
    lambda d: d["feature_bin_counts"].__setitem__(0, [10**100]),
    lambda d: d["feature_bin_counts"].__setitem__(0, [1, 1, 9]),
    lambda d: d["zero_bins"].__setitem__(0, True),
    lambda d: d["zero_bins"].__setitem__(0, 10**100),
    lambda d: d.update(bundles=[[0, 0], [1, 2, 3, 4]]),
    lambda d: d.update(bundles=[[0, 1, 2, 3]]),
    lambda d: d.update(bundles=[[0, 1, 2, 3, 10**100]]),
    lambda d: d.update(bundles=[[1, 0], [2, 3, 4]]),
    lambda d: d.update(enable_bundle=False),
])
def test_efb_rejects_invalid_json_and_unbounded_scalar_claims(change):
    payload = _layout()[2].to_dict()
    change(payload)
    with pytest.raises(ValueError):
        FeatureBundler.from_dict(payload)


@pytest.mark.parametrize("X", [
    [[float("nan")]], [[float("inf")]], [["1"]], [[1j]], [[2**54]],
    np.empty((0, 1)), np.empty((1, 0)), [1., 2.],
])
def test_fit_rejects_invalid_features_without_coercing_categories(X):
    with pytest.raises(ValueError):
        BinMapper.fit(X)


@pytest.mark.parametrize("kwargs", [
    {"rows": [0, True]}, {"rows": [0, 0]}, {"rows": [-1]}, {"rows": [12]},
    {"rows": [1.]}, {"features": [0, False]}, {"features": [5]},
    {"features": [0, 0]}, {"n_jobs": True}, {"n_jobs": 0}, {"n_jobs": -2},
])
def test_histogram_rejects_ambiguous_indices_and_parallel_parameters(kwargs):
    _, _, layout, packed = _layout()
    with pytest.raises(ValueError):
        build_histograms(layout, packed, np.ones(12), np.ones(12), **kwargs)


def test_bins_and_derivatives_validation_rejects_invalid_shapes_and_masses():
    mapper, bins, layout, packed = _layout()
    wrong = bins.copy()
    wrong[0, 0] = mapper.n_bins_[0]
    with pytest.raises(ValueError):
        layout.transform(wrong)
    with pytest.raises(ValueError):
        FeatureBundler.fit(bins[:-1], mapper)
    changed = bins.copy()
    changed[0, 0] = 1
    with pytest.raises(ValueError, match="training counts"):
        FeatureBundler.fit(changed, mapper)
    for g, h in ((np.ones(11), np.ones(12)), (np.ones(12), -np.ones(12)),
                 (np.full(12, np.nan), np.ones(12)), (np.ones(12), np.full(12, np.inf))):
        with pytest.raises(ValueError):
            build_histograms(layout, packed, g, h)
    with pytest.raises(ValueError):
        Histogram([0], [1], [0])
    with pytest.raises(ValueError):
        Histogram([1], [0], [-1])
