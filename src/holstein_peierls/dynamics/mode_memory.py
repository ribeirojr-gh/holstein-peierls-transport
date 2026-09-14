"""Normal-mode diagnostics for frozen-electronic Holstein-Peierls memory.

On a fixed electronic Ehrenfest surface the electron-lattice terms are linear
in the classical coordinates.  They shift the equilibrium but do not change
the harmonic Hessian.  These helpers therefore subtract the exact fixed-surface
equilibrium and analyze the remaining autonomous lattice motion in the bare
normal-mode basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .ehrenfest import LatticeVelocity, lattice_masses_fs

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ModalEnergyArrays:
    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @property
    def total(self) -> FloatArray:
        return np.asarray(self.u + self.vx + self.vy, dtype=np.float64)

    @property
    def sums(self) -> dict[str, float]:
        eu = float(np.sum(self.u))
        evx = float(np.sum(self.vx))
        evy = float(np.sum(self.vy))
        return {
            "u_eV": eu,
            "vx_eV": evx,
            "vy_eV": evy,
            "total_eV": eu + evx + evy,
        }


def _fixed_state(values: ArrayLike, dimension: int) -> ComplexArray:
    psi = np.asarray(values, dtype=np.complex128).reshape(-1)
    if psi.size != int(dimension) or not np.all(np.isfinite(psi)):
        raise ValueError("fixed electronic state must be finite and match the lattice")
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("fixed electronic state must have nonzero norm")
    return psi


def harmonic_wave_numbers(length: int) -> FloatArray:
    """Return FFT-ordered dimensionless wave numbers in radians/site."""
    n = int(length)
    if n < 2:
        raise ValueError("length must be at least two")
    return np.asarray(2.0 * np.pi * np.fft.fftfreq(n), dtype=np.float64)


def chain_laplacian_eigenvalues(length: int) -> FloatArray:
    q = harmonic_wave_numbers(length)
    return np.asarray(4.0 * np.sin(0.5 * q) ** 2, dtype=np.float64)


def frozen_surface_equilibrium(
    parameters: StaticPolaronParameters,
    fixed_electronic_state: ArrayLike,
    held_phase,
) -> LatticeState:
    """Return the zero-chain-mean equilibrium on a fixed electronic surface."""
    psi_flat = _fixed_state(fixed_electronic_state, parameters.n_sites)
    psi = psi_flat.reshape((parameters.ny, parameters.nx), order="C")
    norm = float(np.vdot(psi_flat, psi_flat).real)
    population = np.abs(psi) ** 2 / norm
    phi_x, phi_y = held_phase.phases(0.0)
    phase_x = np.exp(1.0j * float(phi_x))
    phase_y = np.exp(1.0j * float(phi_y))

    left = np.roll(psi, 1, axis=1)
    right = np.roll(psi, -1, axis=1)
    up = np.roll(psi, 1, axis=0)
    down = np.roll(psi, -1, axis=0)
    coh_left = np.real(np.conj(left) * phase_x * psi) / norm
    coh_right = np.real(np.conj(psi) * phase_x * right) / norm
    coh_up = np.real(np.conj(up) * phase_y * psi) / norm
    coh_down = np.real(np.conj(psi) * phase_y * down) / norm

    u_eq = -parameters.alpha_intra * population / parameters.k1

    source_x = 2.0 * parameters.alpha_interx * (coh_left - coh_right)
    source_y = 2.0 * parameters.alpha_intery * (coh_up - coh_down)

    lam_x = chain_laplacian_eigenvalues(parameters.nx)
    lam_y = chain_laplacian_eigenvalues(parameters.ny)

    sx = np.fft.fft(source_x, axis=1, norm="ortho")
    vx_hat = np.zeros_like(sx, dtype=np.complex128)
    nonzero_x = lam_x > 1.0e-14
    vx_hat[:, nonzero_x] = (
        -sx[:, nonzero_x]
        / (parameters.k2 * lam_x[None, nonzero_x])
    )
    vx_eq = np.fft.ifft(vx_hat, axis=1, norm="ortho").real

    sy = np.fft.fft(source_y, axis=0, norm="ortho")
    vy_hat = np.zeros_like(sy, dtype=np.complex128)
    nonzero_y = lam_y > 1.0e-14
    vy_hat[nonzero_y, :] = (
        -sy[nonzero_y, :]
        / (parameters.k2 * lam_y[nonzero_y, None])
    )
    vy_eq = np.fft.ifft(vy_hat, axis=0, norm="ortho").real

    return LatticeState(
        np.asarray(u_eq, dtype=np.float64),
        np.asarray(vx_eq, dtype=np.float64),
        np.asarray(vy_eq, dtype=np.float64),
    )


def real_space_excitation_energies(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    equilibrium: LatticeState,
    parameters: StaticPolaronParameters,
) -> dict[str, float]:
    """Return harmonic excitation energy above the fixed-surface equilibrium."""
    lattice.validate()
    equilibrium.validate()
    velocity.validate((parameters.ny, parameters.nx))
    if lattice.shape != equilibrium.shape:
        raise ValueError("lattice and equilibrium shapes must match")
    mass_u, mass_v = lattice_masses_fs(parameters)
    du = lattice.u - equilibrium.u
    dvx = lattice.vx - equilibrium.vx
    dvy = lattice.vy - equilibrium.vy
    dx = np.roll(dvx, -1, axis=1) - dvx
    dy = np.roll(dvy, -1, axis=0) - dvy
    eu = float(
        0.5 * mass_u * np.sum(velocity.u**2)
        + 0.5 * parameters.k1 * np.sum(du**2)
    )
    evx = float(
        0.5 * mass_v * np.sum(velocity.vx**2)
        + 0.5 * parameters.k2 * np.sum(dx**2)
    )
    evy = float(
        0.5 * mass_v * np.sum(velocity.vy**2)
        + 0.5 * parameters.k2 * np.sum(dy**2)
    )
    return {"u_eV": eu, "vx_eV": evx, "vy_eV": evy, "total_eV": eu + evx + evy}


def modal_energy_arrays(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    equilibrium: LatticeState,
    parameters: StaticPolaronParameters,
) -> ModalEnergyArrays:
    """Return exact orthonormal-FFT modal energies for all three polarizations."""
    mass_u, mass_v = lattice_masses_fs(parameters)
    du = np.asarray(lattice.u - equilibrium.u, dtype=np.float64)
    dvx = np.asarray(lattice.vx - equilibrium.vx, dtype=np.float64)
    dvy = np.asarray(lattice.vy - equilibrium.vy, dtype=np.float64)

    qu = np.fft.fft2(du, norm="ortho")
    qx = np.fft.fft2(dvx, norm="ortho")
    qy = np.fft.fft2(dvy, norm="ortho")
    vu = np.fft.fft2(velocity.u, norm="ortho")
    vx = np.fft.fft2(velocity.vx, norm="ortho")
    vy = np.fft.fft2(velocity.vy, norm="ortho")

    lam_x = chain_laplacian_eigenvalues(parameters.nx)[None, :]
    lam_y = chain_laplacian_eigenvalues(parameters.ny)[:, None]

    eu = 0.5 * mass_u * np.abs(vu) ** 2 + 0.5 * parameters.k1 * np.abs(qu) ** 2
    evx = 0.5 * mass_v * np.abs(vx) ** 2 + 0.5 * parameters.k2 * lam_x * np.abs(qx) ** 2
    evy = 0.5 * mass_v * np.abs(vy) ** 2 + 0.5 * parameters.k2 * lam_y * np.abs(qy) ** 2
    return ModalEnergyArrays(
        np.asarray(eu, dtype=np.float64),
        np.asarray(evx, dtype=np.float64),
        np.asarray(evy, dtype=np.float64),
    )


def q_participation_ratio(energy_by_q: ArrayLike) -> float:
    energy = np.asarray(energy_by_q, dtype=np.float64).reshape(-1)
    if energy.size == 0 or np.any(energy < -1.0e-14) or not np.all(np.isfinite(energy)):
        raise ValueError("energy_by_q must be finite and nonnegative")
    energy = np.clip(energy, 0.0, None)
    denominator = float(np.sum(energy * energy))
    total = float(np.sum(energy))
    return 0.0 if denominator <= 1.0e-30 else float(total * total / denominator)


def traveling_vx_energy_split(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    equilibrium: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    carrier_dx_sites: int,
) -> dict[str, float]:
    """Split x-Peierls modal energy into +/-x traveling components.

    Positive-qx modes are used as one representative of each real-field
    conjugate pair.  `A_plus` propagates toward +x in the laboratory frame and
    `A_minus` toward -x.  qx=0 and the even-cell Nyquist sector are reported as
    nondirectional/special energy.
    """
    dx = int(carrier_dx_sites)
    if dx not in (-1, 1):
        raise ValueError("carrier_dx_sites must be +/-1")
    mass_v = lattice_masses_fs(parameters)[1]
    qfield = np.fft.fft2(lattice.vx - equilibrium.vx, norm="ortho")
    vfield = np.fft.fft2(velocity.vx, norm="ortho")
    qx = harmonic_wave_numbers(parameters.nx)
    omega = 2.0 * np.sqrt(parameters.k2 / mass_v) * np.abs(np.sin(0.5 * qx))
    positive = (qx > 1.0e-14) & (qx < np.pi - 1.0e-14)

    e_plus = 0.0
    e_minus = 0.0
    for ix in np.flatnonzero(positive):
        w = float(omega[ix])
        q = qfield[:, ix]
        v = vfield[:, ix]
        a_plus = 0.5 * (q + 1.0j * v / w)
        a_minus = 0.5 * (q - 1.0j * v / w)
        e_plus += float(np.sum(2.0 * mass_v * w * w * np.abs(a_plus) ** 2))
        e_minus += float(np.sum(2.0 * mass_v * w * w * np.abs(a_minus) ** 2))

    total_vx = modal_energy_arrays(lattice, velocity, equilibrium, parameters).sums["vx_eV"]
    special = max(0.0, float(total_vx - e_plus - e_minus))
    if dx < 0:
        retrograde = e_plus
        comoving = e_minus
    else:
        retrograde = e_minus
        comoving = e_plus
    resolved = retrograde + comoving
    return {
        "plus_x_eV": e_plus,
        "minus_x_eV": e_minus,
        "retrograde_eV": retrograde,
        "comoving_eV": comoving,
        "nondirectional_special_eV": special,
        "retrograde_fraction_of_direction_resolved": (
            0.0 if resolved <= 1.0e-30 else float(retrograde / resolved)
        ),
        "direction_resolved_fraction_of_vx": (
            0.0 if total_vx <= 1.0e-30 else float(resolved / total_vx)
        ),
    }


def axis_resolved_velocity_spectrum(
    velocity_frames: ArrayLike,
    sample_interval_fs: float,
    *,
    dispersive_axis: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return q-omega velocity power, integrated over the degenerate transverse q."""
    frames = np.asarray(velocity_frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] < 4 or not np.all(np.isfinite(frames)):
        raise ValueError("velocity_frames must have shape (time, ny, nx) and be finite")
    dt = float(sample_interval_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample_interval_fs must be positive")
    centered = frames - np.mean(frames, axis=0, keepdims=True)
    window = np.hanning(frames.shape[0]).astype(np.float64)
    temporal = np.fft.rfft(centered * window[:, None, None], axis=0)
    spatial = np.fft.fft2(temporal, axes=(1, 2), norm="ortho")
    power = np.abs(spatial) ** 2
    omega = np.asarray(2.0 * np.pi * np.fft.rfftfreq(frames.shape[0], d=dt), dtype=np.float64)
    if dispersive_axis == "x":
        q = harmonic_wave_numbers(frames.shape[2])
        qomega = np.sum(power, axis=1).T
    elif dispersive_axis == "y":
        q = harmonic_wave_numbers(frames.shape[1])
        qomega = np.sum(power, axis=2).T
    else:
        raise ValueError("dispersive_axis must be 'x' or 'y'")
    return q, omega, np.asarray(qomega, dtype=np.float64)


def integrated_velocity_spectrum(
    velocity_frames: ArrayLike,
    sample_interval_fs: float,
) -> tuple[FloatArray, FloatArray]:
    """Return spatially integrated positive-frequency velocity power."""
    frames = np.asarray(velocity_frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] < 4 or not np.all(np.isfinite(frames)):
        raise ValueError("velocity_frames must have shape (time, ny, nx) and be finite")
    dt = float(sample_interval_fs)
    centered = frames - np.mean(frames, axis=0, keepdims=True)
    window = np.hanning(frames.shape[0]).astype(np.float64)
    temporal = np.fft.rfft(centered * window[:, None, None], axis=0)
    omega = np.asarray(2.0 * np.pi * np.fft.rfftfreq(frames.shape[0], d=dt), dtype=np.float64)
    power = np.sum(np.abs(temporal) ** 2, axis=(1, 2))
    return omega, np.asarray(power, dtype=np.float64)


def ridge_diagnostics(
    qomega_power: ArrayLike,
    omega_axis: ArrayLike,
    analytic_omega_by_q: ArrayLike,
    energy_by_q: ArrayLike,
    *,
    minimum_energy_fraction: float = 0.005,
) -> dict:
    """Compare dominant q-omega peaks with the analytic harmonic dispersion."""
    power = np.asarray(qomega_power, dtype=np.float64)
    omega = np.asarray(omega_axis, dtype=np.float64).reshape(-1)
    analytic = np.asarray(analytic_omega_by_q, dtype=np.float64).reshape(-1)
    energy = np.asarray(energy_by_q, dtype=np.float64).reshape(-1)
    if power.shape != (analytic.size, omega.size) or energy.size != analytic.size:
        raise ValueError("ridge arrays have incompatible shapes")
    if omega.size < 2:
        raise ValueError("omega axis must contain at least two bins")
    total = float(np.sum(np.clip(energy, 0.0, None)))
    bin_width = float(omega[1] - omega[0])
    eligible = (
        (energy >= float(minimum_energy_fraction) * max(total, 1.0e-30))
        & (analytic >= bin_width)
    )
    records = []
    for iq in np.flatnonzero(eligible):
        row = power[iq]
        index = 1 + int(np.argmax(row[1:]))
        peak = float(omega[index])
        error = abs(peak - float(analytic[iq]))
        records.append(
            {
                "q_index": int(iq),
                "analytic_omega_per_fs": float(analytic[iq]),
                "peak_omega_per_fs": peak,
                "absolute_error_per_fs": float(error),
                "q_energy_eV": float(energy[iq]),
            }
        )
    if not records:
        return {
            "eligible_sector_count": 0,
            "frequency_bin_width_per_fs": bin_width,
            "energy_weighted_mean_absolute_error_per_fs": None,
            "maximum_absolute_error_per_fs": None,
            "within_one_frequency_bin": False,
            "records": [],
        }
    weights = np.asarray([item["q_energy_eV"] for item in records], dtype=np.float64)
    errors = np.asarray([item["absolute_error_per_fs"] for item in records], dtype=np.float64)
    weighted = float(np.sum(weights * errors) / np.sum(weights))
    return {
        "eligible_sector_count": int(len(records)),
        "frequency_bin_width_per_fs": bin_width,
        "energy_weighted_mean_absolute_error_per_fs": weighted,
        "maximum_absolute_error_per_fs": float(np.max(errors)),
        "within_one_frequency_bin": bool(weighted <= bin_width),
        "records": records,
    }
