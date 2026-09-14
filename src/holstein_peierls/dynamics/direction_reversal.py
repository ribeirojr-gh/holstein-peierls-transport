"""Energy-preserving x-Peierls traveling-direction reversal for IP1s.

At fixed vx coordinates, changing the sign of every non-special vx velocity
Fourier coefficient exchanges the +x and -x traveling amplitudes of each
harmonic mode while preserving modal energy.  qx=0 and the even-cell Nyquist
sector are left unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..parameters import StaticPolaronParameters
from .ehrenfest import lattice_masses_fs
from .mode_memory import harmonic_wave_numbers

FloatArray = NDArray[np.float64]


def vx_special_mask(nx: int) -> NDArray[np.bool_]:
    """Return the FFT-ordered qx mask for zero/Nyquist special sectors."""
    qx = harmonic_wave_numbers(int(nx))
    return np.asarray(
        (np.abs(qx) <= 1.0e-14) | (np.abs(np.abs(qx) - np.pi) <= 1.0e-14),
        dtype=bool,
    )


def reverse_non_special_vx_velocity(
    velocity_vx: ArrayLike,
    parameters: StaticPolaronParameters,
) -> FloatArray:
    """Reverse all non-special vx traveling directions at fixed coordinates.

    The transformation is an involution.  It preserves the Euclidean norm of
    the vx velocity field exactly up to floating-point roundoff because it
    multiplies orthonormal FFT sectors only by +/-1.
    """
    velocity = np.asarray(velocity_vx, dtype=np.float64)
    expected = (parameters.ny, parameters.nx)
    if velocity.shape != expected or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_vx must be finite and match the lattice shape")
    spectrum = np.fft.fft2(velocity, norm="ortho")
    special = vx_special_mask(parameters.nx)
    transformed = spectrum.copy()
    transformed[:, ~special] *= -1.0
    result = np.fft.ifft2(transformed, norm="ortho")
    imaginary = float(np.max(np.abs(result.imag)))
    if imaginary > 1.0e-12:
        raise FloatingPointError("direction-reversed velocity is not real to roundoff")
    return np.asarray(result.real, dtype=np.float64)


def vx_velocity_kinetic_energy_eV(
    velocity_vx: ArrayLike,
    parameters: StaticPolaronParameters,
) -> float:
    """Return the vx kinetic-energy contribution in eV."""
    velocity = np.asarray(velocity_vx, dtype=np.float64)
    expected = (parameters.ny, parameters.nx)
    if velocity.shape != expected or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_vx must be finite and match the lattice shape")
    mass_v = lattice_masses_fs(parameters)[1]
    return float(0.5 * mass_v * np.sum(velocity * velocity))


def direction_reversal_fourier_errors(
    original_velocity_vx: ArrayLike,
    reversed_velocity_vx: ArrayLike,
    parameters: StaticPolaronParameters,
) -> dict[str, float]:
    """Return direct Fourier-space intervention errors for audit gates."""
    original = np.asarray(original_velocity_vx, dtype=np.float64)
    reversed_velocity = np.asarray(reversed_velocity_vx, dtype=np.float64)
    expected = (parameters.ny, parameters.nx)
    if original.shape != expected or reversed_velocity.shape != expected:
        raise ValueError("vx velocities must match lattice shape")
    before = np.fft.fft2(original, norm="ortho")
    after = np.fft.fft2(reversed_velocity, norm="ortho")
    special = vx_special_mask(parameters.nx)
    special_error = float(np.max(np.abs(after[:, special] - before[:, special])))
    nonspecial_error = float(np.max(np.abs(after[:, ~special] + before[:, ~special])))
    return {
        "special_sector_unchanged_max_abs_A_per_fs": special_error,
        "non_special_sector_sign_reversal_max_abs_A_per_fs": nonspecial_error,
    }
