import json

import numpy as np
import pytest

from eeg_ma.phase_metrics import PhaseMetricError, phase_metrics_from_phases


@pytest.mark.parametrize("lag", [0.0, np.pi, -np.pi, 2 * np.pi])
def test_real_unit_mean_has_explicit_undefined_ciplv(lag):
    result = phase_metrics_from_phases(np.full(128, lag), np.zeros(128))
    assert result.plv == pytest.approx(1.0, abs=2e-15)
    assert result.iplv == pytest.approx(0.0, abs=1e-14)
    assert result.ciplv is None
    assert result.ciplv_status == "undefined"
    assert result.ciplv_reason == "real_unit_mean_zero_or_pi_at_float64_precision"
    assert result.denominator <= result.ciplv_min_denominator
    # Strict JSON carries null, not an invented zero, NaN or Infinity.
    assert json.loads(json.dumps(result.as_dict(), allow_nan=False))["ciplv"] is None


@pytest.mark.parametrize("lag", [np.pi / 2, -np.pi / 2, np.pi / 6, -np.pi / 6, 0.001, 1e-5])
def test_constant_nonzero_lag_normalizes_ciplv_without_epsilon(lag):
    result = phase_metrics_from_phases(np.full(128, lag), np.zeros(128))
    assert result.plv == pytest.approx(1.0, abs=2e-15)
    assert result.iplv == pytest.approx(abs(np.sin(lag)), rel=2e-14)
    assert result.ciplv == pytest.approx(1.0, abs=2e-15)
    assert result.ciplv_status == "defined"
    assert result.ciplv_reason is None


def test_known_two_angle_mean_and_signed_imaginary_cancellation():
    result = phase_metrics_from_phases([0.0, np.pi / 2], [0.0, 0.0])
    assert result.mean_real == pytest.approx(0.5)
    assert result.mean_imag == pytest.approx(0.5)
    assert result.plv == pytest.approx(np.sqrt(0.5))
    assert result.iplv == pytest.approx(0.5)
    assert result.denominator == pytest.approx(np.sqrt(0.75))
    assert result.ciplv == pytest.approx(1 / np.sqrt(3))
    # iPLV is abs(mean(sin(delta))), not mean(abs(sin(delta))).
    cancelled = phase_metrics_from_phases([np.pi / 6, -np.pi / 6], [0.0, 0.0])
    assert cancelled.iplv == 0.0
    assert cancelled.ciplv == 0.0
    assert cancelled.ciplv_status == "defined"


def test_zero_pi_mixture_is_defined_despite_each_individual_real_phase():
    result = phase_metrics_from_phases([0.0, np.pi], [0.0, 0.0])
    assert result.plv == pytest.approx(0.0, abs=1e-15)
    assert result.denominator == pytest.approx(1.0)
    assert result.ciplv == pytest.approx(0.0, abs=1e-15)
    assert result.ciplv_status == "defined"


@pytest.mark.parametrize("lag", [1e-7, -1e-7, np.pi - 1e-7, -np.pi + 1e-7])
def test_near_real_unit_mean_is_not_silently_repaired(lag):
    result = phase_metrics_from_phases(np.full(128, lag), np.zeros(128))
    assert result.ciplv is None
    assert result.ciplv_status == "undefined"
    assert result.ciplv_reason == "denominator_at_or_below_prespecified_minimum"
    assert 0 < result.denominator <= 1e-6


def test_threshold_boundary_and_explicit_metric_policy():
    result = phase_metrics_from_phases([np.arcsin(1e-6)], [0.0])
    assert result.denominator == 1e-6
    assert result.ciplv is None
    # The low-level numerical function records the supplied validity policy.
    relaxed = phase_metrics_from_phases([1e-7], [0.0], ciplv_min_denominator=1e-8)
    assert relaxed.ciplv == pytest.approx(1.0)
    assert relaxed.ciplv_min_denominator == 1e-8


def test_random_finite_values_match_direct_definition_and_are_symmetric():
    rng = np.random.default_rng(731)
    for length in (2, 31, 500, 2500):
        a, b = rng.uniform(-10, 10, (2, length))
        z = np.mean(np.exp(1j * (a - b)))
        result = phase_metrics_from_phases(a, b)
        reversed_result = phase_metrics_from_phases(b, a)
        assert result.plv == pytest.approx(abs(z), abs=2e-15)
        assert result.iplv == pytest.approx(abs(z.imag), abs=2e-15)
        assert result.ciplv == pytest.approx(abs(z.imag) / np.sqrt(1 - z.real**2), abs=2e-15)
        for name in ("plv", "iplv", "ciplv"):
            assert 0 <= getattr(result, name) <= 1
            assert getattr(result, name) == getattr(reversed_result, name)


@pytest.mark.parametrize("a,b", [
    ([], []), ([0], [0, 1]), ([[0, 1]], [[0, 1]]),
    ([0, np.nan], [0, 0]), ([0, 0], [0, np.inf]),
    ([0, -np.inf], [0, 0]), ([0j, 1j], [0, 0]),
    (["0", "1"], [0, 0]), ([True, False], [0, 0]),
    ([np.finfo(float).max], [-np.finfo(float).max]),
])
def test_invalid_phase_input_rejected_without_sample_deletion(a, b):
    with pytest.raises(PhaseMetricError):
        phase_metrics_from_phases(a, b)


@pytest.mark.parametrize("threshold", [0, -1, 1, np.nan, np.inf, True, "1e-6"])
def test_invalid_validity_threshold_rejected(threshold):
    with pytest.raises(PhaseMetricError):
        phase_metrics_from_phases([0], [1], ciplv_min_denominator=threshold)
