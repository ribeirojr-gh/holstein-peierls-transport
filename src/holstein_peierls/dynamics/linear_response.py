"""Paired-field linear-response estimators for TP2.

TP2 separates field-driven drift from finite-time stochastic bias by pairing
trajectories at ``+E`` and ``-E`` with common random-number seeds.  The primary
odd/even decomposition is

    v_odd  = (v(+E) - v(-E)) / 2
    v_even = (v(+E) + v(-E)) / 2.

The validated field convention is electron-like, so a positive electron
mobility satisfies ``v = -mu E``.  This module performs only statistical
analysis; it does not propagate trajectories and it does not claim that a
particular field interval is linear or converged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]

# (angstrom/fs)/(V/angstrom) = angstrom^2/(V fs).
# 1 angstrom^2/fs = 0.1 cm^2/s.
ANGSTROM2_PER_FS_TO_CM2_PER_S = 0.1


@dataclass(frozen=True, slots=True)
class MeanConfidenceInterval:
    """Arithmetic mean, sample standard deviation/SE, and two-sided CI."""

    mean: float
    sample_std: float
    standard_error: float
    lower: float
    upper: float
    confidence: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class SeedLinearResponse:
    """Through-origin odd-response fit for one independent stochastic seed."""

    slope_A2_per_V_fs: float
    electron_mobility_cm2_per_V_s: float
    r_squared_origin: float


@dataclass(frozen=True, slots=True)
class EnsembleLinearResponse:
    """Paired-field response summary preserving seed-level independence."""

    field_magnitudes_mV_per_A: FloatArray
    odd_velocity_A_per_fs: FloatArray
    even_velocity_A_per_fs: FloatArray
    field_mobility_cm2_per_V_s: FloatArray
    field_odd_velocity_ci: tuple[MeanConfidenceInterval, ...]
    field_even_velocity_ci: tuple[MeanConfidenceInterval, ...]
    field_mobility_ci: tuple[MeanConfidenceInterval, ...]
    seed_responses: tuple[SeedLinearResponse, ...]
    mobility_ci: MeanConfidenceInterval
    ensemble_mean_slope_A2_per_V_fs: float
    ensemble_mean_mobility_cm2_per_V_s: float
    ensemble_mean_r_squared_origin: float
    maximum_fractional_linearity_residual: float | None
    maximum_even_to_odd_mean_ratio: float | None


def paired_velocity_components(
    velocity_plus_A_per_fs: ArrayLike,
    velocity_minus_A_per_fs: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return odd/even velocity components for paired ``+E``/``-E`` samples."""
    plus = np.asarray(velocity_plus_A_per_fs, dtype=np.float64)
    minus = np.asarray(velocity_minus_A_per_fs, dtype=np.float64)
    if plus.shape != minus.shape:
        raise ValueError("paired +E and -E velocity arrays must have identical shapes")
    if plus.size == 0 or not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("paired velocities must be non-empty and finite")
    return (
        np.asarray(0.5 * (plus - minus), dtype=np.float64),
        np.asarray(0.5 * (plus + minus), dtype=np.float64),
    )


def electron_mobility_from_odd_velocity(
    odd_velocity_A_per_fs: ArrayLike,
    field_magnitude_mV_per_A: ArrayLike,
) -> FloatArray:
    """Return electron-like mobility in ``cm^2/(V s)``.

    The field magnitudes must be strictly positive.  With the validated
    electron-like Peierls convention ``v_odd = -mu E``.
    """
    velocity = np.asarray(odd_velocity_A_per_fs, dtype=np.float64)
    field_mV = np.asarray(field_magnitude_mV_per_A, dtype=np.float64)
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(field_mV)):
        raise ValueError("velocity and field values must be finite")
    if np.any(field_mV <= 0.0):
        raise ValueError("field magnitudes must be strictly positive")
    try:
        velocity_b, field_b = np.broadcast_arrays(velocity, field_mV)
    except ValueError as exc:
        raise ValueError("velocity and field arrays are not broadcast-compatible") from exc
    field_V_per_A = 1.0e-3 * field_b
    mobility = -ANGSTROM2_PER_FS_TO_CM2_PER_S * velocity_b / field_V_per_A
    return np.asarray(mobility, dtype=np.float64)


def mean_confidence_interval(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
) -> MeanConfidenceInterval:
    """Return a two-sided Student-t confidence interval for independent samples."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    conf = float(confidence)
    if data.size < 2:
        raise ValueError("at least two independent samples are required for a t interval")
    if not np.all(np.isfinite(data)):
        raise ValueError("confidence-interval samples must be finite")
    if not np.isfinite(conf) or not 0.0 < conf < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    mean = float(np.mean(data))
    sample_std = float(np.std(data, ddof=1))
    standard_error = float(sample_std / np.sqrt(data.size))
    critical = float(student_t.ppf(0.5 * (1.0 + conf), data.size - 1))
    half_width = critical * standard_error
    return MeanConfidenceInterval(
        mean=mean,
        sample_std=sample_std,
        standard_error=standard_error,
        lower=float(mean - half_width),
        upper=float(mean + half_width),
        confidence=conf,
        sample_count=int(data.size),
    )


def through_origin_slope(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[float, float]:
    """Return least-squares slope through the origin and origin-based R^2."""
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if xx.size == 0 or xx.shape != yy.shape:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal size")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("fit arrays must be finite")
    denominator = float(np.dot(xx, xx))
    if denominator <= 0.0:
        raise ValueError("through-origin fit requires at least one non-zero x value")
    slope = float(np.dot(xx, yy) / denominator)
    residual = yy - slope * xx
    sse = float(np.dot(residual, residual))
    sst_origin = float(np.dot(yy, yy))
    if sst_origin == 0.0:
        r_squared = 1.0 if sse == 0.0 else 0.0
    else:
        r_squared = float(1.0 - sse / sst_origin)
    return slope, r_squared


def seed_linear_response(
    field_magnitudes_mV_per_A: ArrayLike,
    odd_velocity_A_per_fs: ArrayLike,
) -> SeedLinearResponse:
    """Fit one seed's odd velocity to field through the origin."""
    fields = np.asarray(field_magnitudes_mV_per_A, dtype=np.float64).reshape(-1)
    velocity = np.asarray(odd_velocity_A_per_fs, dtype=np.float64).reshape(-1)
    if fields.shape != velocity.shape or fields.size == 0:
        raise ValueError("field and odd-velocity arrays must have equal non-zero size")
    if not np.all(np.isfinite(fields)) or np.any(fields <= 0.0):
        raise ValueError("field magnitudes must be finite and positive")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("odd velocities must be finite")
    fields_V_per_A = 1.0e-3 * fields
    slope, r_squared = through_origin_slope(fields_V_per_A, velocity)
    mobility = -ANGSTROM2_PER_FS_TO_CM2_PER_S * slope
    return SeedLinearResponse(
        slope_A2_per_V_fs=float(slope),
        electron_mobility_cm2_per_V_s=float(mobility),
        r_squared_origin=float(r_squared),
    )


def analyze_paired_field_ensemble(
    field_magnitudes_mV_per_A: ArrayLike,
    velocity_plus_A_per_fs: ArrayLike,
    velocity_minus_A_per_fs: ArrayLike,
    *,
    confidence: float = 0.95,
) -> EnsembleLinearResponse:
    """Analyze a ``(n_seed, n_field)`` paired-field velocity ensemble.

    Statistical uncertainty is formed across independent seeds.  Repeated field
    values within a seed are used to estimate that seed's through-origin slope
    and are not incorrectly counted as independent samples.
    """
    fields = np.asarray(field_magnitudes_mV_per_A, dtype=np.float64).reshape(-1)
    plus = np.asarray(velocity_plus_A_per_fs, dtype=np.float64)
    minus = np.asarray(velocity_minus_A_per_fs, dtype=np.float64)
    if fields.size == 0 or not np.all(np.isfinite(fields)) or np.any(fields <= 0.0):
        raise ValueError("field magnitudes must be a non-empty finite positive array")
    if np.unique(fields).size != fields.size:
        raise ValueError("field magnitudes must be unique")
    if plus.ndim != 2 or minus.shape != plus.shape:
        raise ValueError("paired velocity arrays must both have shape (n_seed, n_field)")
    if plus.shape[1] != fields.size:
        raise ValueError("velocity field dimension does not match field magnitudes")
    if plus.shape[0] < 2:
        raise ValueError("at least two independent seeds are required")

    odd, even = paired_velocity_components(plus, minus)
    field_mobility = electron_mobility_from_odd_velocity(
        odd,
        fields[np.newaxis, :],
    )

    odd_ci = tuple(
        mean_confidence_interval(odd[:, index], confidence=confidence)
        for index in range(fields.size)
    )
    even_ci = tuple(
        mean_confidence_interval(even[:, index], confidence=confidence)
        for index in range(fields.size)
    )
    mobility_field_ci = tuple(
        mean_confidence_interval(field_mobility[:, index], confidence=confidence)
        for index in range(fields.size)
    )

    seed_responses = tuple(
        seed_linear_response(fields, odd[index, :])
        for index in range(plus.shape[0])
    )
    seed_mobility = np.asarray(
        [response.electron_mobility_cm2_per_V_s for response in seed_responses],
        dtype=np.float64,
    )
    mobility_ci = mean_confidence_interval(seed_mobility, confidence=confidence)

    fields_V_per_A = 1.0e-3 * fields
    mean_odd = np.mean(odd, axis=0)
    mean_even = np.mean(even, axis=0)
    slope, r_squared = through_origin_slope(fields_V_per_A, mean_odd)
    ensemble_mobility = -ANGSTROM2_PER_FS_TO_CM2_PER_S * slope
    fitted = slope * fields_V_per_A
    signal_scale = float(np.max(np.abs(fitted)))
    if signal_scale <= 1.0e-15:
        fractional_residual = None
    else:
        fractional_residual = float(np.max(np.abs(mean_odd - fitted)) / signal_scale)
    odd_scale = float(np.max(np.abs(mean_odd)))
    if odd_scale <= 1.0e-15:
        even_to_odd = None
    else:
        even_to_odd = float(np.max(np.abs(mean_even)) / odd_scale)

    return EnsembleLinearResponse(
        field_magnitudes_mV_per_A=fields.copy(),
        odd_velocity_A_per_fs=np.asarray(odd, dtype=np.float64),
        even_velocity_A_per_fs=np.asarray(even, dtype=np.float64),
        field_mobility_cm2_per_V_s=np.asarray(field_mobility, dtype=np.float64),
        field_odd_velocity_ci=odd_ci,
        field_even_velocity_ci=even_ci,
        field_mobility_ci=mobility_field_ci,
        seed_responses=seed_responses,
        mobility_ci=mobility_ci,
        ensemble_mean_slope_A2_per_V_fs=float(slope),
        ensemble_mean_mobility_cm2_per_V_s=float(ensemble_mobility),
        ensemble_mean_r_squared_origin=float(r_squared),
        maximum_fractional_linearity_residual=fractional_residual,
        maximum_even_to_odd_mean_ratio=even_to_odd,
    )
