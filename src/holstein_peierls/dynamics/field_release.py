"""Gauge-continuous field-release and trailing-pattern helpers for IP1n."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .field import UniformElectricField2D
from .wavepacket_flux import outward_boundary_flux_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HeldPeierlsPhase2D:
    """Constant vector-potential continuation after electric-field release.

    ``ex_v_per_angstrom`` and ``ey_v_per_angstrom`` retain the axis metadata of
    the driving field so the phased Hamiltonian builder keeps the same cell
    validation.  They are *not* post-release physical electric fields.  The
    physical field is zero because ``phase_rates_per_fs`` returns zero.
    """

    ex_v_per_angstrom: float
    ey_v_per_angstrom: float
    ax_angstrom: float
    ay_angstrom: float
    time_origin_fs: float
    held_phi_x: float
    held_phi_y: float

    @classmethod
    def from_driving_field(
        cls,
        field: UniformElectricField2D,
        switch_time_fs: float,
    ) -> "HeldPeierlsPhase2D":
        time = float(switch_time_fs)
        if not np.isfinite(time):
            raise ValueError("switch_time_fs must be finite")
        phi_x, phi_y = field.phases(time)
        return cls(
            ex_v_per_angstrom=float(field.ex_v_per_angstrom),
            ey_v_per_angstrom=float(field.ey_v_per_angstrom),
            ax_angstrom=float(field.ax_angstrom),
            ay_angstrom=float(field.ay_angstrom),
            time_origin_fs=time,
            held_phi_x=float(phi_x),
            held_phi_y=float(phi_y),
        )

    @property
    def is_zero(self) -> bool:
        """Return whether the held vector potential has exactly zero phase.

        This property is a Hamiltonian shortcut used by the existing builders;
        it is not the physical electric-field status.  The physical field is
        always zero for this class because the phase rates vanish.
        """
        return self.held_phi_x == 0.0 and self.held_phi_y == 0.0

    @property
    def physical_field_is_zero(self) -> bool:
        return True

    def phases(self, time_fs: float) -> tuple[float, float]:
        time = float(time_fs)
        if not np.isfinite(time):
            raise ValueError("time_fs must be finite")
        return float(self.held_phi_x), float(self.held_phi_y)

    def phase_rates_per_fs(self) -> tuple[float, float]:
        return 0.0, 0.0


@dataclass(frozen=True, slots=True)
class AmplitudeWindowMetrics:
    start_fs: float
    end_fs: float
    center_fs: float
    rms_amplitude_eV_per_fs: float
    peak_amplitude_eV_per_fs: float


def trailing_outward_matrix(
    longitudinal_flux_eV_per_fs: ArrayLike,
    s_axis: ArrayLike,
    *,
    distances_sites: tuple[int, ...] = (1, 2, 3, 4),
) -> FloatArray:
    """Return time x distance trailing outward-current matrix."""
    columns = [
        outward_boundary_flux_series(
            longitudinal_flux_eV_per_fs,
            s_axis,
            int(distance),
            side="backward",
        )
        for distance in distances_sites
    ]
    if not columns:
        raise ValueError("at least one distance is required")
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("trailing currents must be finite")
    return matrix


def trailing_amplitude_series(outward_matrix: ArrayLike) -> FloatArray:
    """RMS current amplitude across the selected trailing boundaries."""
    matrix = np.asarray(outward_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("outward_matrix must be finite with shape (time, distance)")
    return np.asarray(np.sqrt(np.mean(matrix * matrix, axis=1)), dtype=np.float64)


def _uniform_times(times_fs: ArrayLike) -> tuple[FloatArray, float]:
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times must contain at least two finite samples")
    delta = np.diff(times)
    if np.any(delta <= 0.0):
        raise ValueError("times must be strictly increasing")
    dt = float(np.mean(delta))
    if not np.allclose(delta, dt, rtol=0.0, atol=max(1.0e-10, 1.0e-10 * abs(dt))):
        raise ValueError("times must be uniformly sampled")
    return times, dt


def amplitude_window_metrics(
    times_fs: ArrayLike,
    amplitude_eV_per_fs: ArrayLike,
    start_fs: float,
    end_fs: float,
) -> AmplitudeWindowMetrics:
    times, _ = _uniform_times(times_fs)
    amplitude = np.asarray(amplitude_eV_per_fs, dtype=np.float64).reshape(-1)
    if amplitude.shape != times.shape or not np.all(np.isfinite(amplitude)):
        raise ValueError("amplitude must be finite and match times")
    start = float(start_fs)
    end = float(end_fs)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("window bounds must be finite and ordered")
    mask = (times >= start) & (times <= end)
    if np.count_nonzero(mask) < 2:
        raise ValueError("window must contain at least two samples")
    local = amplitude[mask]
    return AmplitudeWindowMetrics(
        start_fs=start,
        end_fs=end,
        center_fs=0.5 * (start + end),
        rms_amplitude_eV_per_fs=float(np.sqrt(np.mean(local * local))),
        peak_amplitude_eV_per_fs=float(np.max(local)),
    )


def sliding_amplitude_windows(
    times_fs: ArrayLike,
    amplitude_eV_per_fs: ArrayLike,
    start_fs: float,
    end_fs: float,
    *,
    width_fs: float,
    step_fs: float,
) -> list[AmplitudeWindowMetrics]:
    times, dt = _uniform_times(times_fs)
    width = float(width_fs)
    step = float(step_fs)
    if width <= 0.0 or step <= 0.0:
        raise ValueError("width and step must be positive")
    for value, name in ((width, "width_fs"), (step, "step_fs")):
        ratio = value / dt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{name} must be an integer multiple of the sampling interval")
    first = float(start_fs)
    last_start = float(end_fs) - width
    if last_start < first - 1.0e-10:
        return []
    n = int(np.floor((last_start - first) / step + 1.0e-10)) + 1
    starts = first + step * np.arange(n, dtype=np.float64)
    return [
        amplitude_window_metrics(times, amplitude_eV_per_fs, float(s), float(s + width))
        for s in starts
    ]


def empirical_percentile(value: float, background: ArrayLike) -> float:
    samples = np.asarray(background, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)) or not np.isfinite(value):
        raise ValueError("value and background must be finite and background non-empty")
    return float(100.0 * np.count_nonzero(samples <= float(value)) / samples.size)


def sustained_pattern_gate(
    records: list[dict],
    *,
    switch_time_fs: float,
    percentile_threshold: float = 95.0,
    late_offset_fs: float = 1000.0,
    minimum_passing_bins: int = 3,
    minimum_latest_offset_fs: float = 1500.0,
) -> dict:
    """Evaluate the preregistered late phase-structured memory gate."""
    t_switch = float(switch_time_fs)
    late = [record for record in records if float(record["center_fs"]) > t_switch + late_offset_fs]
    passing = [
        record
        for record in late
        if float(record["rms_percentile"]) >= percentile_threshold
        and float(record["peak_percentile"]) >= percentile_threshold
    ]
    latest = None if not passing else max(float(record["center_fs"]) for record in passing)
    sustained = bool(
        len(passing) >= int(minimum_passing_bins)
        and latest is not None
        and latest >= t_switch + float(minimum_latest_offset_fs)
    )
    return {
        "late_bin_count": int(len(late)),
        "late_passing_bin_count": int(len(passing)),
        "late_passing_fraction": 0.0 if not late else float(len(passing) / len(late)),
        "latest_passing_center_fs": latest,
        "sustained_pattern_memory": sustained,
    }
