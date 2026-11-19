"""Auditable phase metrics for the prospective, matched sensitivity controls.

This module does not replace the historical PLV/iPLV implementations.  For
``z = mean(exp(1j * (phase_a - phase_b)))`` the undirected magnitudes are
``PLV = abs(z)``, ``iPLV = abs(imag(z))`` and
``ciPLV = abs(imag(z)) / sqrt(1 - real(z)**2)``.  The correction follows
Bruña, Maestú & Pereda (2018), J. Neural Eng. 15 056011, equations 13–14,
https://doi.org/10.1088/1741-2552/aacfe4 (author manuscript:
https://arxiv.org/abs/1710.08037).  Taking the absolute value is this project's
explicit undirected, nonnegative convention.

The denominator is evaluated with the algebraically equivalent identity
``1 - mean(cos(delta))**2 = mean(sin(delta)**2) + var(cos(delta))``.
This avoids subtractive cancellation near a real unit mean.  The threshold
below is an explicit numerical validity policy, *not* an added denominator.
Undefined ciPLV values are ``None`` with an auditable reason, never zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


PHASE_METRICS_VERSION = "eeg-phase-metrics-1"
DEFAULT_CIPLV_MIN_DENOMINATOR = 1e-6
DENOMINATOR_EVALUATION = "sqrt(mean(sin(delta)^2)+mean((cos(delta)-mean_cos)^2))"
_BOUNDS_TOLERANCE = 64 * np.finfo(np.float64).eps


class PhaseMetricError(ValueError):
    """Phase inputs or numerical bounds are invalid."""


@dataclass(frozen=True)
class PhaseMetrics:
    """One pair's values and the evidence needed to interpret ciPLV."""

    plv: float
    iplv: float
    ciplv: float | None
    ciplv_status: str
    ciplv_reason: str | None
    mean_real: float
    mean_imag: float
    denominator: float
    ciplv_min_denominator: float
    n_samples: int
    roundoff_adjustments: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        result = asdict(self)
        result["roundoff_adjustments"] = list(self.roundoff_adjustments)
        result["schema_version"] = PHASE_METRICS_VERSION
        result["denominator_evaluation"] = DENOMINATOR_EVALUATION
        return result


def validate_denominator_minimum(value: float) -> float:
    """Require an explicit positive validity threshold, bounded by one."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PhaseMetricError("ciplv_min_denominator must be a finite real number in (0, 1)")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise PhaseMetricError("ciplv_min_denominator must be a finite real number in (0, 1)")
    return threshold


def _phase_array(value, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != 1 or not array.size:
        raise PhaseMetricError(f"{label} must be a nonempty one-dimensional real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise PhaseMetricError(f"{label} must contain only finite float64 phases")
    return array


def _bounded(value: float, name: str, adjustments: list[str]) -> float:
    if not math.isfinite(value) or value < -_BOUNDS_TOLERANCE or value > 1.0 + _BOUNDS_TOLERANCE:
        raise PhaseMetricError(f"{name} violated the mathematical [0, 1] bound")
    if value < 0.0 or value > 1.0:
        adjustments.append(name + "_roundoff_clipped_to_unit_interval")
        return min(1.0, max(0.0, value))
    return value


def phase_metrics_from_phases(
    phase_a,
    phase_b,
    *,
    ciplv_min_denominator: float = DEFAULT_CIPLV_MIN_DENOMINATOR,
) -> PhaseMetrics:
    """Compute all three metrics from exactly the same finite sample vectors.

    ``ciplv_status == 'undefined'`` means a zero or insufficient denominator;
    the accompanying PLV and iPLV remain mathematically defined.  A caller
    comparing representations must apply its prospectively specified common
    exclusion policy.  This function does not impute, filter, fit, score, infer
    a neural cause, or authorize the use of any scientific dataset.
    """
    threshold = validate_denominator_minimum(ciplv_min_denominator)
    a = _phase_array(phase_a, "phase_a")
    b = _phase_array(phase_b, "phase_b")
    if a.shape != b.shape:
        raise PhaseMetricError("phase_a and phase_b must have the same shape")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = a - b
    if not np.isfinite(delta).all():
        raise PhaseMetricError("phase difference overflows float64")

    cosine, sine = np.cos(delta), np.sin(delta)
    n_samples = len(delta)
    real = math.fsum(cosine) / n_samples
    imag = math.fsum(sine) / n_samples
    # Compensated sums and the variance identity retain precision near 0/pi.
    radicand = (math.fsum(sine * sine)
                 + math.fsum((cosine - real) * (cosine - real))) / n_samples
    adjustments: list[str] = []
    denominator = _bounded(math.sqrt(radicand), "denominator", adjustments)
    plv = _bounded(math.hypot(real, imag), "plv", adjustments)
    iplv = _bounded(abs(imag), "iplv", adjustments)

    if denominator <= threshold:
        # Exact pi has a tiny sine residual in float64, so distinguish the
        # real-unit numerical degeneracy from a larger, merely small divisor.
        reason = ("real_unit_mean_zero_or_pi_at_float64_precision"
                  if abs(real) == 1.0 and denominator <= _BOUNDS_TOLERANCE
                  else "denominator_at_or_below_prespecified_minimum")
        return PhaseMetrics(plv, iplv, None, "undefined", reason, real, imag,
                            denominator, threshold, n_samples, tuple(adjustments))

    ciplv = _bounded(iplv / denominator, "ciplv", adjustments)
    return PhaseMetrics(plv, iplv, ciplv, "defined", None, real, imag,
                        denominator, threshold, n_samples, tuple(adjustments))
