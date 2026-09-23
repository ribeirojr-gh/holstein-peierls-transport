"""Deterministic low-q Peierls velocity ensemble for IP2a.

The IP2 production ensemble must not be tuned using post-intervention outcomes.
This module therefore creates a fixed, auditable family of low-q velocity-only
Peierls perturbations.  Coordinates and the electronic state are untouched, so
the added matter energy at preparation is exactly the added lattice kinetic
energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..parameters import StaticPolaronParameters
from .ehrenfest import LatticeVelocity, lattice_masses_fs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PeierlsEnsembleMember:
    member_id: int
    pair_id: int
    pair_sign: int
    target_energy_eV: float
    velocity: LatticeVelocity


def _normalized_low_q_pattern(
    length: int,
    transverse_length: int,
    pair_id: int,
    *,
    axis: str,
    max_mode_index: int = 4,
) -> FloatArray:
    """Return a deterministic unit-norm real low-q pattern with zero mean."""
    n = int(length)
    nt = int(transverse_length)
    p = int(pair_id)
    kmax = int(max_mode_index)
    if n < 2 or nt < 1:
        raise ValueError("invalid lattice dimensions")
    if p < 0:
        raise ValueError("pair_id must be nonnegative")
    if not 1 <= kmax < n // 2:
        raise ValueError("max_mode_index must be a nonzero non-Nyquist sector")
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    coordinate = np.arange(n, dtype=np.float64)
    line = np.zeros(n, dtype=np.float64)
    # Fixed low-q amplitudes.  The phase maps are intentionally different for
    # x/y and use a 32-point phase grid, but no random generator.
    weights = np.asarray(0.72 ** np.arange(kmax, dtype=np.float64), dtype=np.float64)
    for j, (mode, weight) in enumerate(zip(range(1, kmax + 1), weights, strict=True)):
        if axis == "x":
            phase_index = ((p + 1) * (3 * mode + 1) + 5 * j) % 32
        else:
            phase_index = ((p + 1) * (5 * mode + 3) + 7 * j + 1) % 32
        phase = 2.0 * np.pi * float(phase_index) / 32.0
        line += float(weight) * np.cos(2.0 * np.pi * mode * coordinate / n + phase)

    # Remove roundoff-level zero-mode leakage before lifting to 2D.
    line -= float(np.mean(line))
    if axis == "x":
        field = np.broadcast_to(line[None, :], (nt, n)).copy()
    else:
        field = np.broadcast_to(line[:, None], (n, nt)).copy()

    field -= float(np.mean(field))
    norm = float(np.linalg.norm(field))
    if not np.isfinite(norm) or norm <= 1.0e-30:
        raise FloatingPointError("degenerate ensemble pattern")
    return np.asarray(field / norm, dtype=np.float64)


def peierls_velocity_member(
    parameters: StaticPolaronParameters,
    member_id: int,
    *,
    ensemble_size: int = 32,
    target_energy_eV: float = 1.0e-5,
    vx_energy_fraction: float = 0.5,
    max_mode_index: int = 4,
) -> PeierlsEnsembleMember:
    """Construct one energy-matched IP2a preparation.

    Members 0..N/2-1 and N/2..N-1 are exact sign-opposite pairs.  The
    perturbation is velocity-only, so coordinates/electronic state can remain
    exactly identical across the ensemble and the total added matter energy is
    exactly the target lattice kinetic energy.
    """
    size = int(ensemble_size)
    index = int(member_id)
    energy = float(target_energy_eV)
    fx = float(vx_energy_fraction)
    if size < 4 or size % 2:
        raise ValueError("ensemble_size must be an even integer >= 4")
    if not 0 <= index < size:
        raise ValueError("member_id outside ensemble")
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("target_energy_eV must be finite and positive")
    if not np.isfinite(fx) or not 0.0 < fx < 1.0:
        raise ValueError("vx_energy_fraction must be strictly between 0 and 1")

    half = size // 2
    pair_id = index % half
    pair_sign = 1 if index < half else -1
    _, mass_v = lattice_masses_fs(parameters)

    px = _normalized_low_q_pattern(
        parameters.nx,
        parameters.ny,
        pair_id,
        axis="x",
        max_mode_index=max_mode_index,
    )
    # y helper returns shape (ny, nx) by taking length=ny, transverse=nx.
    py = _normalized_low_q_pattern(
        parameters.ny,
        parameters.nx,
        pair_id,
        axis="y",
        max_mode_index=max_mode_index,
    )

    ex = energy * fx
    ey = energy * (1.0 - fx)
    scale_x = np.sqrt(2.0 * ex / mass_v)
    scale_y = np.sqrt(2.0 * ey / mass_v)
    velocity = LatticeVelocity(
        np.zeros((parameters.ny, parameters.nx), dtype=np.float64),
        np.asarray(pair_sign * scale_x * px, dtype=np.float64),
        np.asarray(pair_sign * scale_y * py, dtype=np.float64),
    )
    velocity.validate((parameters.ny, parameters.nx))
    return PeierlsEnsembleMember(index, pair_id, pair_sign, energy, velocity)


def ensemble_velocity_kinetic_energy_eV(
    member: PeierlsEnsembleMember,
    parameters: StaticPolaronParameters,
) -> float:
    """Kinetic energy carried by the prepared Peierls velocity perturbation."""
    _, mass_v = lattice_masses_fs(parameters)
    return float(
        0.5 * mass_v * (
            np.sum(member.velocity.vx * member.velocity.vx)
            + np.sum(member.velocity.vy * member.velocity.vy)
        )
    )


def peierls_fourier_support(
    velocity: LatticeVelocity,
    *,
    threshold: float = 1.0e-12,
) -> dict[str, list[tuple[int, int]]]:
    """Return FFT indices carrying appreciable vx/vy velocity amplitude."""
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    vx = np.fft.fft2(velocity.vx, norm="ortho")
    vy = np.fft.fft2(velocity.vy, norm="ortho")
    sx = np.argwhere(np.abs(vx) > float(threshold))
    sy = np.argwhere(np.abs(vy) > float(threshold))
    return {
        "vx_indices": [(int(i), int(j)) for i, j in sx],
        "vy_indices": [(int(i), int(j)) for i, j in sy],
    }
