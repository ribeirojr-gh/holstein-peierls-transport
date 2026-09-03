"""Field-driven zero-temperature coupled Ehrenfest dynamics for D3.

D3 adds only the already validated D1 Peierls electric-field phase to the D2
moving-lattice Ehrenfest equations.  The external field makes the matter energy
explicitly time dependent, so the relevant conservation law is the work balance

    E_matter(t) - E_matter(0) = integral <psi|dH/dt|psi> dt.

No thermostat, stochastic force, decoherence model, or GPU approximation is
introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..energy import lattice_energy
from ..hamiltonian import bond_transfer_integrals
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .classical import drift, half_kick
from .coupled import CoupledEhrenfestState, interpolate_lattice
from .ehrenfest import (
    DynamicEnergyBreakdown,
    LatticeVelocity,
    ehrenfest_force,
    lattice_kinetic_energy,
)
from .field import (
    UniformElectricField2D,
    build_sparse_field_hamiltonian,
)
from .time_dependent import cfm4_lanczos_step, rk4_time_dependent_step

ComplexArray = NDArray[np.complex128]
ElectronicMethod = Literal["cfm4_lanczos", "rk4"]


@dataclass(frozen=True, slots=True)
class DrivenCoupledPropagationResult:
    """Final D3 state, accumulated external work, and work counters."""

    state: CoupledEhrenfestState
    field_work_eV: float
    elapsed_seconds: float
    steps: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int


def _state_vector(
    values: ArrayLike,
    dimension: int,
) -> tuple[ComplexArray, float]:
    psi = np.asarray(values, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != dimension:
        raise ValueError("electronic state dimension does not match lattice")
    if not np.all(np.isfinite(psi)):
        raise ValueError("electronic state must contain only finite values")
    norm_squared = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("electronic state must have finite non-zero norm")
    return psi, norm_squared


def field_electronic_energy_expectation(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    field: UniformElectricField2D,
    time_fs: float,
) -> float:
    """Return ``<psi|H(q,t)|psi>/<psi|psi>`` in eV."""
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi, norm_squared = _state_vector(electronic_state, parameters.n_sites)
    hamiltonian = build_sparse_field_hamiltonian(state, parameters, field, time_fs)
    value = np.vdot(psi, hamiltonian @ psi) / norm_squared
    if abs(float(np.imag(value))) > 1.0e-10:
        raise FloatingPointError("field electronic energy expectation is not real")
    return float(np.real(value))


def field_ehrenfest_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    field: UniformElectricField2D,
    time_fs: float,
):
    """Return ``d[V_lattice + <H(q,t)>]/dq`` at fixed propagated state.

    The Peierls phase rotates the nearest-neighbour coherences entering the
    intermolecular force.  At zero field this reduces exactly to the D2
    complex-wavefunction Ehrenfest gradient.
    """
    # Local import avoids duplicating the public gradient container definition.
    from ..gradients import LatticeGradient

    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi_flat, norm_squared = _state_vector(electronic_state, parameters.n_sites)
    psi = psi_flat.reshape(state.shape, order="C")

    population = np.abs(psi) ** 2 / norm_squared
    left = np.roll(psi, 1, axis=1)
    right = np.roll(psi, -1, axis=1)
    up = np.roll(psi, 1, axis=0)
    down = np.roll(psi, -1, axis=0)

    phi_x, phi_y = field.phases(time_fs)
    phase_x = np.exp(1.0j * phi_x)
    phase_y = np.exp(1.0j * phi_y)

    coh_left = np.real(np.conj(left) * phase_x * psi) / norm_squared
    coh_right = np.real(np.conj(psi) * phase_x * right) / norm_squared
    coh_up = np.real(np.conj(up) * phase_y * psi) / norm_squared
    coh_down = np.real(np.conj(psi) * phase_y * down) / norm_squared

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


def field_ehrenfest_force(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    field: UniformElectricField2D,
    time_fs: float,
):
    """Return the D3 classical force ``-dE_matter/dq``."""
    from ..gradients import LatticeGradient

    if field.is_zero:
        return ehrenfest_force(state, parameters, electronic_state)
    gradient = field_ehrenfest_gradient(
        state,
        parameters,
        electronic_state,
        field,
        time_fs,
    )
    return LatticeGradient(-gradient.u, -gradient.vx, -gradient.vy)


def field_power(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    field: UniformElectricField2D,
    time_fs: float,
) -> float:
    """Return ``<psi|partial H/partial t|psi>`` in eV/fs.

    This is the instantaneous power delivered by the prescribed external field
    to the matter subsystem in the vector-potential gauge used by D1.
    """
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi_flat, norm_squared = _state_vector(electronic_state, parameters.n_sites)
    if field.is_zero:
        return 0.0

    psi = psi_flat.reshape(state.shape, order="C")
    right = np.roll(psi, -1, axis=1)
    down = np.roll(psi, -1, axis=0)
    tx, ty = bond_transfer_integrals(state, parameters)
    phi_x, phi_y = field.phases(time_fs)
    rate_x, rate_y = field.phase_rates_per_fs()

    z_x = np.conj(psi) * np.exp(1.0j * phi_x) * right
    z_y = np.conj(psi) * np.exp(1.0j * phi_y) * down
    power_x = -2.0 * rate_x * np.sum(tx * np.imag(z_x)) / norm_squared
    power_y = -2.0 * rate_y * np.sum(ty * np.imag(z_y)) / norm_squared
    return float(power_x + power_y)


def field_dynamic_total_energy(
    state: LatticeState,
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    field: UniformElectricField2D,
    time_fs: float,
) -> DynamicEnergyBreakdown:
    """Return the instantaneous matter-energy decomposition for D3."""
    intra, inter = lattice_energy(state, parameters)
    return DynamicEnergyBreakdown(
        intramolecular_lattice=intra,
        intermolecular_lattice=inter,
        lattice_kinetic=lattice_kinetic_energy(velocity, parameters),
        electronic=field_electronic_energy_expectation(
            state,
            parameters,
            electronic_state,
            field,
            time_fs,
        ),
    )


def coupled_field_verlet_step(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    time_fs: float,
    dt_fs: float,
    *,
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> tuple[CoupledEhrenfestState, float, int, int]:
    """Advance one D3 split step and return trapezoidal external work."""
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_fs must be finite and dt_fs positive")
    psi, _ = _state_vector(state.electronic_state, parameters.n_sites)
    state.lattice.validate()
    state.velocity.validate((parameters.ny, parameters.nx))

    power_old = field_power(state.lattice, parameters, psi, field, time)
    force_old = field_ehrenfest_force(
        state.lattice,
        parameters,
        psi,
        field,
        time,
    )
    half_velocity = half_kick(state.velocity, force_old, parameters, dt)
    lattice_new = drift(state.lattice, half_velocity, dt)

    def hamiltonian_at(stage_time_fs: float):
        fraction = (float(stage_time_fs) - time) / dt
        # Guard tiny stage-roundoff at interval endpoints.
        fraction = float(np.clip(fraction, 0.0, 1.0))
        lattice_stage = interpolate_lattice(state.lattice, lattice_new, fraction)
        return build_sparse_field_hamiltonian(
            lattice_stage,
            parameters,
            field,
            stage_time_fs,
        )

    if electronic_method == "cfm4_lanczos":
        electronic = cfm4_lanczos_step(
            hamiltonian_at,
            psi,
            time,
            dt,
            krylov_dimension=krylov_dimension,
        )
    elif electronic_method == "rk4":
        electronic = rk4_time_dependent_step(
            hamiltonian_at,
            psi,
            time,
            dt,
        )
    else:
        raise ValueError(f"unknown electronic method: {electronic_method}")

    psi_new = np.asarray(electronic.state, dtype=np.complex128)
    new_time = time + dt
    force_new = field_ehrenfest_force(
        lattice_new,
        parameters,
        psi_new,
        field,
        new_time,
    )
    velocity_new = half_kick(half_velocity, force_new, parameters, dt)
    power_new = field_power(
        lattice_new,
        parameters,
        psi_new,
        field,
        new_time,
    )
    work_increment = 0.5 * dt * (power_old + power_new)
    return (
        CoupledEhrenfestState(lattice_new, velocity_new, psi_new),
        float(work_increment),
        electronic.hamiltonian_evaluations,
        electronic.hamiltonian_applications,
    )


def integrate_coupled_field_verlet(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> DrivenCoupledPropagationResult:
    """Integrate D3 with velocity Verlet and moving-lattice field propagation."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    current = CoupledEhrenfestState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    current_time = float(initial_time_fs)
    if not np.isfinite(current_time):
        raise ValueError("initial_time_fs must be finite")
    accumulated_work = 0.0
    evaluations = 0
    applications = 0
    start = perf_counter()
    for _ in range(steps):
        current, work, h_eval, h_apply = coupled_field_verlet_step(
            current,
            parameters,
            field,
            current_time,
            dt_fs,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        accumulated_work += work
        evaluations += h_eval
        applications += h_apply
        current_time += float(dt_fs)
    elapsed = perf_counter() - start
    return DrivenCoupledPropagationResult(
        state=current,
        field_work_eV=float(accumulated_work),
        elapsed_seconds=float(elapsed),
        steps=steps,
        hamiltonian_evaluations=evaluations,
        hamiltonian_applications=applications,
    )
