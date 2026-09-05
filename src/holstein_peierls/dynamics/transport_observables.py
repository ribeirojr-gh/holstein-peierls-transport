"""Periodic one-carrier transport observables for TP1.

The primary transport coordinate is obtained from oriented nearest-neighbour
probability currents and their time integral.  This avoids the discontinuity of
a naive Cartesian position expectation when a localized carrier crosses a
periodic cell boundary.

The electric-field convention is the already validated electron-like Peierls
phase used by :mod:`holstein_peierls.dynamics.field`.  With that convention,

    P_field = - E dot v_particle,

where the field is in V/angstrom, the particle velocity is in angstrom/fs, and
power is in eV/fs.

A circular first-moment center is also provided as a localization diagnostic.
It must not be used as an unwrapped transport coordinate, and it is reported as
undefined when the circular resultant is too small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..hamiltonian import bond_transfer_integrals, build_sparse_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .field import UniformElectricField2D, build_sparse_field_hamiltonian
from .frozen import HBAR_EV_FS

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class BondProbabilityCurrents2D:
    """Oriented probability currents from each site to its +x/+y neighbour.

    ``jx_per_fs[y, x]`` is positive when probability flows from site ``(y,x)``
    to ``(y,x+1)``.  ``jy_per_fs`` is defined analogously toward ``(y+1,x)``.
    """

    jx_per_fs: FloatArray
    jy_per_fs: FloatArray

    @property
    def total_flux_x_per_fs(self) -> float:
        return float(np.sum(self.jx_per_fs))

    @property
    def total_flux_y_per_fs(self) -> float:
        return float(np.sum(self.jy_per_fs))


@dataclass(frozen=True, slots=True)
class TransportKinematics2D:
    """Instantaneous PBC-safe particle transport kinematics."""

    total_flux_x_per_fs: float
    total_flux_y_per_fs: float
    velocity_x_A_per_fs: float
    velocity_y_A_per_fs: float


@dataclass(frozen=True, slots=True)
class PeriodicCenter2D:
    """Circular first-moment center and reliability resultants.

    ``x_A``/``y_A`` are ``None`` when the corresponding circular resultant is
    below the caller-selected threshold.  The coordinates, when defined, lie in
    ``[0, N_axis * a_axis)``.
    """

    x_A: float | None
    y_A: float | None
    x_resultant: float
    y_resultant: float


def _validate_transport_cell(parameters: StaticPolaronParameters) -> None:
    # Two-site periodic axes have duplicated forward/reverse neighbour bonds in
    # the archived assignment semantics.  D1 already rejects this ambiguity for
    # a phased axis; TP1 keeps the current definition unambiguous by requiring a
    # genuine periodic ring in both directions.
    if parameters.nx < 3 or parameters.ny < 3:
        raise ValueError("periodic transport observables require nx >= 3 and ny >= 3")


def _normalized_state(
    electronic_state: ArrayLike,
    parameters: StaticPolaronParameters,
) -> tuple[ComplexArray, float]:
    psi = np.asarray(electronic_state, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != parameters.n_sites:
        raise ValueError("electronic state dimension does not match parameters")
    if not np.all(np.isfinite(psi)):
        raise ValueError("electronic state must contain only finite values")
    norm_squared = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("electronic state must have finite non-zero norm")
    return psi, norm_squared


def bond_probability_currents(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    *,
    field: UniformElectricField2D | None = None,
    time_fs: float = 0.0,
) -> BondProbabilityCurrents2D:
    """Return oriented nearest-neighbour probability currents in ``1/fs``.

    For the +x bond from site ``i`` to ``j`` the convention is

    ``J_i->j = -(2/hbar) Im[psi_i^* H_ij psi_j]``.

    The wavefunction need not be pre-normalized; the returned currents are
    divided by its norm squared without modifying the supplied state.
    """
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    _validate_transport_cell(parameters)
    psi_flat, norm_squared = _normalized_state(electronic_state, parameters)
    psi = psi_flat.reshape(lattice.shape, order="C")
    right = np.roll(psi, shift=-1, axis=1)
    down = np.roll(psi, shift=-1, axis=0)
    tx, ty = bond_transfer_integrals(lattice, parameters)

    if field is None:
        phase_x = 1.0 + 0.0j
        phase_y = 1.0 + 0.0j
    else:
        phi_x, phi_y = field.phases(float(time_fs))
        phase_x = np.exp(1.0j * phi_x)
        phase_y = np.exp(1.0j * phi_y)

    z_x = np.conj(psi) * tx * phase_x * right
    z_y = np.conj(psi) * ty * phase_y * down
    prefactor = -2.0 / (HBAR_EV_FS * norm_squared)
    jx = np.asarray(prefactor * np.imag(z_x), dtype=np.float64)
    jy = np.asarray(prefactor * np.imag(z_y), dtype=np.float64)
    return BondProbabilityCurrents2D(jx, jy)


def transport_kinematics(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    *,
    field: UniformElectricField2D | None = None,
    time_fs: float = 0.0,
    ax_angstrom: float | None = None,
    ay_angstrom: float | None = None,
) -> TransportKinematics2D:
    """Return summed bond flux and PBC-safe particle velocity.

    If a field is supplied its lattice spacings are used by default, ensuring
    exact consistency with the Peierls-phase convention.  Without a field the
    caller must supply positive ``ax_angstrom`` and ``ay_angstrom`` explicitly.
    """
    currents = bond_probability_currents(
        lattice,
        parameters,
        electronic_state,
        field=field,
        time_fs=time_fs,
    )
    if field is not None:
        ax = field.ax_angstrom if ax_angstrom is None else float(ax_angstrom)
        ay = field.ay_angstrom if ay_angstrom is None else float(ay_angstrom)
    else:
        if ax_angstrom is None or ay_angstrom is None:
            raise ValueError("ax_angstrom and ay_angstrom are required without a field")
        ax = float(ax_angstrom)
        ay = float(ay_angstrom)
    if not np.isfinite(ax) or not np.isfinite(ay) or ax <= 0.0 or ay <= 0.0:
        raise ValueError("transport lattice spacings must be finite and positive")

    flux_x = currents.total_flux_x_per_fs
    flux_y = currents.total_flux_y_per_fs
    return TransportKinematics2D(
        total_flux_x_per_fs=flux_x,
        total_flux_y_per_fs=flux_y,
        velocity_x_A_per_fs=float(ax * flux_x),
        velocity_y_A_per_fs=float(ay * flux_y),
    )


def continuity_population_derivative(
    currents: BondProbabilityCurrents2D,
) -> FloatArray:
    """Return ``d rho/dt`` from the discrete bond-current continuity equation."""
    jx = np.asarray(currents.jx_per_fs, dtype=np.float64)
    jy = np.asarray(currents.jy_per_fs, dtype=np.float64)
    if jx.ndim != 2 or jy.shape != jx.shape:
        raise ValueError("jx and jy must be two-dimensional arrays of equal shape")
    incoming_x = np.roll(jx, shift=1, axis=1)
    incoming_y = np.roll(jy, shift=1, axis=0)
    return np.asarray(incoming_x + incoming_y - jx - jy, dtype=np.float64)


def schrodinger_population_derivative(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    *,
    field: UniformElectricField2D | None = None,
    time_fs: float = 0.0,
) -> FloatArray:
    """Return the direct Schrodinger ``d |psi_i|^2/dt`` in ``1/fs``.

    This independent expression is useful for validating the bond-current
    continuity identity.
    """
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    _validate_transport_cell(parameters)
    psi, norm_squared = _normalized_state(electronic_state, parameters)
    if field is None:
        hamiltonian = build_sparse_hamiltonian(lattice, parameters)
    else:
        hamiltonian = build_sparse_field_hamiltonian(
            lattice,
            parameters,
            field,
            float(time_fs),
        )
    hpsi = hamiltonian @ psi
    derivative = (2.0 / (HBAR_EV_FS * norm_squared)) * np.imag(np.conj(psi) * hpsi)
    return np.asarray(derivative.reshape(lattice.shape, order="C"), dtype=np.float64)


def field_power_from_particle_velocity(
    field: UniformElectricField2D,
    kinematics: TransportKinematics2D,
) -> float:
    """Return electron-like field power ``-E dot v`` in eV/fs."""
    return float(
        -field.ex_v_per_angstrom * kinematics.velocity_x_A_per_fs
        - field.ey_v_per_angstrom * kinematics.velocity_y_A_per_fs
    )


def trapezoidal_displacement_increment(
    old_kinematics: TransportKinematics2D,
    new_kinematics: TransportKinematics2D,
    dt_fs: float,
) -> tuple[float, float]:
    """Return one trapezoidal unwrapped displacement increment in angstrom."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    dx = 0.5 * dt * (
        old_kinematics.velocity_x_A_per_fs + new_kinematics.velocity_x_A_per_fs
    )
    dy = 0.5 * dt * (
        old_kinematics.velocity_y_A_per_fs + new_kinematics.velocity_y_A_per_fs
    )
    return float(dx), float(dy)


def periodic_center_of_probability(
    electronic_state: ArrayLike,
    parameters: StaticPolaronParameters,
    *,
    ax_angstrom: float,
    ay_angstrom: float,
    minimum_resultant: float = 1.0e-12,
) -> PeriodicCenter2D:
    """Return circular first-moment centers for a one-particle probability.

    The result is a periodic localization diagnostic, not an unwrapped position.
    A component is returned as ``None`` when its circular resultant magnitude is
    below ``minimum_resultant``.
    """
    _validate_transport_cell(parameters)
    psi, norm_squared = _normalized_state(electronic_state, parameters)
    ax = float(ax_angstrom)
    ay = float(ay_angstrom)
    threshold = float(minimum_resultant)
    if not np.isfinite(ax) or not np.isfinite(ay) or ax <= 0.0 or ay <= 0.0:
        raise ValueError("lattice spacings must be finite and positive")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_resultant must be finite and non-negative")

    population = (np.abs(psi) ** 2 / norm_squared).reshape(
        (parameters.ny, parameters.nx), order="C"
    )
    x_phase = np.exp(2.0j * np.pi * np.arange(parameters.nx) / parameters.nx)
    y_phase = np.exp(2.0j * np.pi * np.arange(parameters.ny) / parameters.ny)
    z_x = complex(np.sum(population * x_phase[np.newaxis, :]))
    z_y = complex(np.sum(population * y_phase[:, np.newaxis]))
    r_x = float(abs(z_x))
    r_y = float(abs(z_y))

    def center(z: complex, resultant: float, n: int, spacing: float) -> float | None:
        if resultant < threshold:
            return None
        phase = float(np.angle(z)) % (2.0 * np.pi)
        return float((phase / (2.0 * np.pi)) * n * spacing)

    return PeriodicCenter2D(
        x_A=center(z_x, r_x, parameters.nx, ax),
        y_A=center(z_y, r_y, parameters.ny, ay),
        x_resultant=r_x,
        y_resultant=r_y,
    )
