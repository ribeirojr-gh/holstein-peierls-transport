"""Ehrenfest force and energy primitives for deterministic D2 dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..energy import lattice_energy
from ..gradients import LatticeGradient
from ..hamiltonian import build_sparse_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

AS2_PER_FS2 = 1_000_000.0


@dataclass(slots=True)
class LatticeVelocity:
    """Velocities of ``u``, ``vx`` and ``vy`` in angstrom/fs."""

    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @classmethod
    def zeros(cls, ny: int, nx: int) -> "LatticeVelocity":
        shape = (ny, nx)
        return cls(*(np.zeros(shape, dtype=np.float64) for _ in range(3)))

    def copy(self) -> "LatticeVelocity":
        return LatticeVelocity(self.u.copy(), self.vx.copy(), self.vy.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape

    def validate(self, expected_shape: tuple[int, int] | None = None) -> None:
        if self.u.shape != self.vx.shape or self.u.shape != self.vy.shape:
            raise ValueError("lattice velocity arrays must have identical shapes")
        if self.u.ndim != 2:
            raise ValueError("lattice velocity arrays must be two-dimensional")
        if expected_shape is not None and self.shape != expected_shape:
            raise ValueError("lattice velocity shape does not match lattice state")
        if not all(np.all(np.isfinite(a)) for a in (self.u, self.vx, self.vy)):
            raise ValueError("lattice velocities must contain only finite values")


@dataclass(frozen=True, slots=True)
class DynamicEnergyBreakdown:
    """Energy components for deterministic one-carrier dynamics."""

    intramolecular_lattice: float
    intermolecular_lattice: float
    lattice_kinetic: float
    electronic: float

    @property
    def total(self) -> float:
        return (
            self.intramolecular_lattice
            + self.intermolecular_lattice
            + self.lattice_kinetic
            + self.electronic
        )


def _state_vector(values: ArrayLike, dimension: int) -> tuple[ComplexArray, float]:
    psi = np.asarray(values, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != dimension:
        raise ValueError("electronic state dimension does not match lattice")
    if not np.all(np.isfinite(psi)):
        raise ValueError("electronic state must contain only finite values")
    norm_squared = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("electronic state must have finite non-zero norm")
    return psi, norm_squared


def legacy_mass_to_fs(mass_eV_as2_per_A2: float) -> float:
    """Convert eV as^2/A^2 to eV fs^2/A^2."""
    mass = float(mass_eV_as2_per_A2)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    return mass / AS2_PER_FS2


def lattice_masses_fs(parameters: StaticPolaronParameters) -> tuple[float, float]:
    """Return the ``u`` and intermolecular effective masses in fs units."""
    return legacy_mass_to_fs(parameters.m1), legacy_mass_to_fs(parameters.m2)


def electronic_energy_expectation(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
) -> float:
    """Return ``<psi|H(q)|psi>/<psi|psi>`` in eV."""
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi, norm_squared = _state_vector(electronic_state, parameters.n_sites)
    hamiltonian = build_sparse_hamiltonian(state, parameters)
    value = np.vdot(psi, hamiltonian @ psi) / norm_squared
    if abs(float(np.imag(value))) > 1.0e-10:
        raise FloatingPointError("electronic energy expectation is not real")
    return float(np.real(value))


def ehrenfest_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
) -> LatticeGradient:
    """Return ``d[V_lattice + <H>]/dq`` for a propagated complex state."""
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi, norm_squared = _state_vector(electronic_state, parameters.n_sites)
    psi = psi.reshape(state.shape, order="C")

    population = np.abs(psi) ** 2 / norm_squared
    left = np.roll(psi, 1, axis=1)
    right = np.roll(psi, -1, axis=1)
    up = np.roll(psi, 1, axis=0)
    down = np.roll(psi, -1, axis=0)

    coh_left = np.real(np.conj(left) * psi) / norm_squared
    coh_right = np.real(np.conj(psi) * right) / norm_squared
    coh_up = np.real(np.conj(up) * psi) / norm_squared
    coh_down = np.real(np.conj(psi) * down) / norm_squared

    vx_left = np.roll(state.vx, 1, axis=1)
    vx_right = np.roll(state.vx, -1, axis=1)
    vy_up = np.roll(state.vy, 1, axis=0)
    vy_down = np.roll(state.vy, -1, axis=0)

    grad_u = parameters.k1 * state.u + parameters.alpha_intra * population
    grad_vx = (
        parameters.k2 * (2.0 * state.vx - vx_left - vx_right)
        + 2.0 * parameters.alpha_interx * (coh_left - coh_right)
    )
    grad_vy = (
        parameters.k2 * (2.0 * state.vy - vy_up - vy_down)
        + 2.0 * parameters.alpha_intery * (coh_up - coh_down)
    )
    return LatticeGradient(grad_u, grad_vx, grad_vy)


def ehrenfest_force(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
) -> LatticeGradient:
    """Return the classical force ``-dE/dq``."""
    gradient = ehrenfest_gradient(state, parameters, electronic_state)
    return LatticeGradient(-gradient.u, -gradient.vx, -gradient.vy)


def lattice_kinetic_energy(
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
) -> float:
    """Return lattice kinetic energy in eV for velocities in A/fs."""
    velocity.validate((parameters.ny, parameters.nx))
    mass_u, mass_v = lattice_masses_fs(parameters)
    return float(
        0.5 * mass_u * np.sum(velocity.u * velocity.u)
        + 0.5 * mass_v * (np.sum(velocity.vx * velocity.vx) + np.sum(velocity.vy * velocity.vy))
    )


def dynamic_total_energy(
    state: LatticeState,
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
) -> DynamicEnergyBreakdown:
    """Return the deterministic D2 total-energy decomposition."""
    intra, inter = lattice_energy(state, parameters)
    return DynamicEnergyBreakdown(
        intramolecular_lattice=intra,
        intermolecular_lattice=inter,
        lattice_kinetic=lattice_kinetic_energy(velocity, parameters),
        electronic=electronic_energy_expectation(state, parameters, electronic_state),
    )
