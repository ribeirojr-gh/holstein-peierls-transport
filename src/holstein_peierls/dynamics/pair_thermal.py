"""D6c finite-temperature BAOAB dynamics for bipolaron and exciton sectors.

D6c transfers the validated D4 lattice thermostat to the matrix-free pair
Hamiltonians validated in D6a/D6b.  It deliberately changes only the classical
bath integration: the pair electronic dynamics remains coherent Ehrenfest
propagation with CF4-Lanczos.

The BAOAB ordering is

    B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2),

with an exact Ornstein-Uhlenbeck O step.  The electronic state is propagated
over the actual two-segment lattice path and the kinetic-energy jump across O is
recorded as heat from the classical bath.

As in D4, the exact uniform zero modes of the intermolecular vx/vy coordinates
are explicit:

- ``retain`` keeps them, with 3N kinetic degrees of freedom;
- ``project`` constrains their coordinate and velocity means to zero, with
  3N-2 kinetic degrees of freedom.

No electronic decoherence, electric field, or transport fit is included here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.random import Generator

from ..gradients import LatticeGradient
from ..lattice import LatticeState
from .coupled import interpolate_lattice
from .ehrenfest import LatticeVelocity
from .frozen import lanczos_exponential_step
from .langevin import BOLTZMANN_EV_PER_K, LangevinBath, ou_parameters
from .pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    PairParameters,
    pair_coupled_verlet_step,
    pair_dynamic_total_energy,
    pair_ehrenfest_force,
    pair_half_kick,
    pair_lattice_kinetic_energy,
)
from .pair_frozen import FrozenPairHamiltonianAction, PairSector, normalized_pair_state

ZeroModePolicy = Literal["retain", "project"]


@dataclass(frozen=True, slots=True)
class PairThermostattedStepResult:
    """Result of one D6c BAOAB/Ehrenfest pair step."""

    state: PairCoupledState
    bath_heat_eV: float
    hamiltonian_evaluations: int
    hamiltonian_applications: int


@dataclass(frozen=True, slots=True)
class PairThermostattedPropagationResult:
    """Final D6c pair state and accumulated bath exchange."""

    state: PairCoupledState
    bath_heat_eV: float
    elapsed_seconds: float
    steps: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    zero_mode_policy: ZeroModePolicy


def _validate_policy(policy: str) -> ZeroModePolicy:
    if policy not in ("retain", "project"):
        raise ValueError("zero_mode_policy must be 'retain' or 'project'")
    return policy  # type: ignore[return-value]


def _remove_mean(values: np.ndarray) -> np.ndarray:
    return np.asarray(values - float(np.mean(values)), dtype=np.float64)


def project_pair_zero_modes(state: PairCoupledState) -> PairCoupledState:
    """Return a copy with uniform vx/vy coordinate and velocity modes removed."""
    state.lattice.validate()
    state.velocity.validate(state.lattice.shape)
    return PairCoupledState(
        lattice=LatticeState(
            state.lattice.u.copy(),
            _remove_mean(state.lattice.vx),
            _remove_mean(state.lattice.vy),
        ),
        velocity=LatticeVelocity(
            state.velocity.u.copy(),
            _remove_mean(state.velocity.vx),
            _remove_mean(state.velocity.vy),
        ),
        electronic_state=np.asarray(state.electronic_state, dtype=np.complex128).copy(),
    )


def pair_zero_mode_means(state: PairCoupledState) -> dict[str, float]:
    """Return uniform intermolecular coordinate/velocity components."""
    return {
        "vx_A": float(np.mean(state.lattice.vx)),
        "vy_A": float(np.mean(state.lattice.vy)),
        "vx_velocity_A_per_fs": float(np.mean(state.velocity.vx)),
        "vy_velocity_A_per_fs": float(np.mean(state.velocity.vy)),
    }


def pair_thermostatted_degrees_of_freedom(
    parameters: PairParameters,
    zero_mode_policy: ZeroModePolicy,
) -> int:
    policy = _validate_policy(zero_mode_policy)
    return 3 * parameters.n_sites if policy == "retain" else 3 * parameters.n_sites - 2


def _require_projected_state(state: PairCoupledState, tolerance: float = 1.0e-12) -> None:
    if any(abs(value) > tolerance for value in pair_zero_mode_means(state).values()):
        raise ValueError(
            "project zero-mode policy requires an initially projected state; "
            "call project_pair_zero_modes first"
        )


def _project_force(force: LatticeGradient) -> LatticeGradient:
    return LatticeGradient(
        np.asarray(force.u, dtype=np.float64),
        _remove_mean(np.asarray(force.vx, dtype=np.float64)),
        _remove_mean(np.asarray(force.vy, dtype=np.float64)),
    )


def _project_velocity(velocity: LatticeVelocity) -> LatticeVelocity:
    return LatticeVelocity(
        np.asarray(velocity.u, dtype=np.float64),
        _remove_mean(np.asarray(velocity.vx, dtype=np.float64)),
        _remove_mean(np.asarray(velocity.vy, dtype=np.float64)),
    )


def _project_lattice(lattice: LatticeState) -> LatticeState:
    return LatticeState(
        np.asarray(lattice.u, dtype=np.float64),
        _remove_mean(np.asarray(lattice.vx, dtype=np.float64)),
        _remove_mean(np.asarray(lattice.vy, dtype=np.float64)),
    )


def pair_ou_velocity_step(
    velocity: LatticeVelocity,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    dt_fs: float,
    rng: Generator,
) -> LatticeVelocity:
    """Apply the exact D4 Ornstein-Uhlenbeck step using explicit pair masses."""
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    velocity.validate()
    shape = velocity.shape
    u_ou = ou_parameters(
        gamma_per_fs=bath.gamma_u_per_fs,
        temperature_K=bath.temperature_K,
        mass_eV_fs2_per_A2=masses.intramolecular,
        dt_fs=dt_fs,
    )
    v_ou = ou_parameters(
        gamma_per_fs=bath.gamma_v_per_fs,
        temperature_K=bath.temperature_K,
        mass_eV_fs2_per_A2=masses.intermolecular,
        dt_fs=dt_fs,
    )

    if u_ou.velocity_sigma == 0.0:
        u = np.asarray(u_ou.damping_factor * velocity.u, dtype=np.float64)
    else:
        u = np.asarray(
            u_ou.damping_factor * velocity.u
            + u_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )

    if v_ou.velocity_sigma == 0.0:
        vx = np.asarray(v_ou.damping_factor * velocity.vx, dtype=np.float64)
        vy = np.asarray(v_ou.damping_factor * velocity.vy, dtype=np.float64)
    else:
        vx = np.asarray(
            v_ou.damping_factor * velocity.vx
            + v_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )
        vy = np.asarray(
            v_ou.damping_factor * velocity.vy
            + v_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )
    return LatticeVelocity(u, vx, vy)


def pair_kinetic_temperature_K(
    velocity: LatticeVelocity,
    masses: PairLatticeMasses,
    *,
    degrees_of_freedom: int,
) -> float:
    """Return T = 2K/(N_dof k_B) for the pair-model lattice."""
    dof = int(degrees_of_freedom)
    if dof <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    kinetic = pair_lattice_kinetic_energy(velocity, masses)
    return float(2.0 * kinetic / (dof * BOLTZMANN_EV_PER_K))


def _weighted_action(
    first: FrozenPairHamiltonianAction,
    first_weight: float,
    second: FrozenPairHamiltonianAction,
    second_weight: float,
) -> FrozenPairHamiltonianAction:
    if first.sector != second.sector or first.n_sites != second.n_sites:
        raise ValueError("pair actions must belong to the same sector and lattice")
    w1 = float(first_weight)
    w2 = float(second_weight)
    return FrozenPairHamiltonianAction(
        left_hamiltonian=w1 * first.left_hamiltonian + w2 * second.left_hamiltonian,
        right_hamiltonian=w1 * first.right_hamiltonian + w2 * second.right_hamiltonian,
        interaction_eV=w1 * first.interaction_eV + w2 * second.interaction_eV,
        sector=first.sector,
    )


def _piecewise_pair_cfm4_step(
    factory: MovingPairHamiltonianFactory,
    electronic_state: np.ndarray,
    lattice_at,
    time_fs: float,
    dt_fs: float,
    *,
    krylov_dimension: int,
) -> tuple[np.ndarray, int, int]:
    """CF4-Lanczos step for an arbitrary lattice path callable."""
    dt = float(dt_fs)
    psi = normalized_pair_state(electronic_state, factory.n_sites)
    sqrt3 = float(np.sqrt(3.0))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a1 = (3.0 - 2.0 * sqrt3) / 12.0
    a2 = (3.0 + 2.0 * sqrt3) / 12.0
    h1 = factory.at(lattice_at(float(time_fs) + c1 * dt))
    h2 = factory.at(lattice_at(float(time_fs) + c2 * dt))
    right = _weighted_action(h1, a2, h2, a1)
    left = _weighted_action(h1, a1, h2, a2)
    first = lanczos_exponential_step(
        right, psi, dt, krylov_dimension=krylov_dimension
    )
    second = lanczos_exponential_step(
        left, first.state, dt, krylov_dimension=krylov_dimension
    )
    return (
        np.asarray(second.state, dtype=np.complex128),
        2,
        int(first.hamiltonian_applications + second.hamiltonian_applications),
    )


def pair_coupled_baoab_step(
    state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    rng: Generator,
    time_fs: float,
    dt_fs: float,
    *,
    zero_mode_policy: ZeroModePolicy = "project",
    krylov_dimension: int = 8,
    factory: MovingPairHamiltonianFactory | None = None,
) -> PairThermostattedStepResult:
    """Advance one finite-temperature pair BAOAB/Ehrenfest step."""
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_fs must be finite and dt_fs positive")
    policy = _validate_policy(zero_mode_policy)
    if factory is None:
        factory = MovingPairHamiltonianFactory(sector, parameters)
    psi = normalized_pair_state(state.electronic_state, parameters.n_sites)

    if bath.is_frictionless and policy == "retain":
        reduced, h_eval, h_apply = pair_coupled_verlet_step(
            state,
            sector,
            parameters,
            masses,
            dt,
            krylov_dimension=krylov_dimension,
            factory=factory,
        )
        return PairThermostattedStepResult(reduced, 0.0, h_eval, h_apply)

    if policy == "project":
        _require_projected_state(state)

    force_old = pair_ehrenfest_force(state.lattice, sector, parameters, psi)
    if policy == "project":
        force_old = _project_force(force_old)
    half_velocity = pair_half_kick(state.velocity, force_old, masses, dt)
    if policy == "project":
        half_velocity = _project_velocity(half_velocity)

    lattice_mid = LatticeState(
        np.asarray(state.lattice.u + 0.5 * dt * half_velocity.u, dtype=np.float64),
        np.asarray(state.lattice.vx + 0.5 * dt * half_velocity.vx, dtype=np.float64),
        np.asarray(state.lattice.vy + 0.5 * dt * half_velocity.vy, dtype=np.float64),
    )
    if policy == "project":
        lattice_mid = _project_lattice(lattice_mid)

    kinetic_before_o = pair_lattice_kinetic_energy(half_velocity, masses)
    thermostatted_velocity = pair_ou_velocity_step(
        half_velocity, masses, bath, dt, rng
    )
    if policy == "project":
        thermostatted_velocity = _project_velocity(thermostatted_velocity)
    kinetic_after_o = pair_lattice_kinetic_energy(thermostatted_velocity, masses)
    bath_heat = float(kinetic_after_o - kinetic_before_o)

    lattice_new = LatticeState(
        np.asarray(lattice_mid.u + 0.5 * dt * thermostatted_velocity.u, dtype=np.float64),
        np.asarray(lattice_mid.vx + 0.5 * dt * thermostatted_velocity.vx, dtype=np.float64),
        np.asarray(lattice_mid.vy + 0.5 * dt * thermostatted_velocity.vy, dtype=np.float64),
    )
    if policy == "project":
        lattice_new = _project_lattice(lattice_new)

    midpoint_time = time + 0.5 * dt

    def lattice_at(stage_time_fs: float) -> LatticeState:
        stage_time = float(stage_time_fs)
        if stage_time <= midpoint_time:
            fraction = float(np.clip((stage_time - time) / (0.5 * dt), 0.0, 1.0))
            return interpolate_lattice(state.lattice, lattice_mid, fraction)
        fraction = float(
            np.clip((stage_time - midpoint_time) / (0.5 * dt), 0.0, 1.0)
        )
        return interpolate_lattice(lattice_mid, lattice_new, fraction)

    psi_new, h_eval, h_apply = _piecewise_pair_cfm4_step(
        factory,
        psi,
        lattice_at,
        time,
        dt,
        krylov_dimension=krylov_dimension,
    )
    force_new = pair_ehrenfest_force(lattice_new, sector, parameters, psi_new)
    if policy == "project":
        force_new = _project_force(force_new)
    velocity_new = pair_half_kick(
        thermostatted_velocity, force_new, masses, dt
    )
    if policy == "project":
        velocity_new = _project_velocity(velocity_new)

    return PairThermostattedStepResult(
        state=PairCoupledState(lattice_new, velocity_new, psi_new),
        bath_heat_eV=bath_heat,
        hamiltonian_evaluations=h_eval,
        hamiltonian_applications=h_apply,
    )


def integrate_pair_coupled_baoab(
    initial_state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    rng: Generator,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
    zero_mode_policy: ZeroModePolicy = "project",
    krylov_dimension: int = 8,
) -> PairThermostattedPropagationResult:
    """Integrate a D6c finite-temperature pair trajectory."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    policy = _validate_policy(zero_mode_policy)
    if policy == "project":
        _require_projected_state(initial_state)
    current = PairCoupledState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    factory = MovingPairHamiltonianFactory(sector, parameters)
    time = float(initial_time_fs)
    bath_heat = 0.0
    h_eval = 0
    h_apply = 0
    start = perf_counter()
    for _ in range(steps):
        result = pair_coupled_baoab_step(
            current,
            sector,
            parameters,
            masses,
            bath,
            rng,
            time,
            dt_fs,
            zero_mode_policy=policy,
            krylov_dimension=krylov_dimension,
            factory=factory,
        )
        current = result.state
        bath_heat += result.bath_heat_eV
        h_eval += result.hamiltonian_evaluations
        h_apply += result.hamiltonian_applications
        time += float(dt_fs)
    return PairThermostattedPropagationResult(
        state=current,
        bath_heat_eV=float(bath_heat),
        elapsed_seconds=float(perf_counter() - start),
        steps=int(steps),
        hamiltonian_evaluations=int(h_eval),
        hamiltonian_applications=int(h_apply),
        zero_mode_policy=policy,
    )


def pair_generalized_energy_residual_eV(
    initial_state: PairCoupledState,
    final_state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    bath_heat_eV: float,
) -> float:
    """Return Delta E_matter - Q_bath for a zero-field D6c trajectory."""
    factory = MovingPairHamiltonianFactory(sector, parameters)
    initial_energy = pair_dynamic_total_energy(
        initial_state, sector, parameters, masses, factory=factory
    ).total
    final_energy = pair_dynamic_total_energy(
        final_state, sector, parameters, masses, factory=factory
    ).total
    return float((final_energy - initial_energy) - float(bath_heat_eV))
