"""Finite-temperature coupled one-carrier Ehrenfest dynamics for D4b.

D4b couples the D4a BAOAB Langevin lattice bath to the already validated D2/D3
one-carrier Ehrenfest dynamics. Electronic propagation remains coherent and
unitary when CF4-Lanczos is selected; electronic decoherence/thermalization is a
separate D5 physics choice.

The lattice path within one BAOAB step is piecewise linear,

    B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2),

and the electronic propagator samples that actual path at its internal stage
times. The O step changes only lattice velocities, so its kinetic-energy change
is recorded as heat exchanged with the classical bath.

Two policies for the exact zero modes of the intermolecular ``vx`` and ``vy``
coordinates are explicit:

- ``retain``: keep the unconstrained collective modes, matching the mathematical
  coordinate representation of the legacy model;
- ``project``: constrain the uniform ``vx`` and ``vy`` coordinate/velocity modes
  to zero. The corresponding kinetic degree-of-freedom count is ``3N-2``.

No zero-mode choice is made silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ..hamiltonian import build_sparse_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .classical import drift, half_kick
from .coupled import (
    CoupledEhrenfestState,
    coupled_verlet_step,
    interpolate_lattice,
)
from .driven import (
    coupled_field_verlet_step,
    field_ehrenfest_force,
    field_power,
)
from .ehrenfest import (
    LatticeVelocity,
    ehrenfest_force,
    lattice_kinetic_energy,
)
from .field import UniformElectricField2D, build_sparse_field_hamiltonian
from .langevin import LangevinBath, ou_velocity_step
from .time_dependent import cfm4_lanczos_step, rk4_time_dependent_step

ComplexArray = NDArray[np.complex128]
ElectronicMethod = Literal["cfm4_lanczos", "rk4"]
ZeroModePolicy = Literal["retain", "project"]


@dataclass(frozen=True, slots=True)
class ThermostattedStepResult:
    """Result of one coupled BAOAB/Ehrenfest step."""

    state: CoupledEhrenfestState
    bath_heat_eV: float
    field_work_eV: float
    hamiltonian_evaluations: int
    hamiltonian_applications: int


@dataclass(frozen=True, slots=True)
class ThermostattedPropagationResult:
    """Final D4b state and accumulated external energy exchanges."""

    state: CoupledEhrenfestState
    bath_heat_eV: float
    field_work_eV: float
    elapsed_seconds: float
    steps: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    zero_mode_policy: ZeroModePolicy


def _validate_state(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
) -> ComplexArray:
    state.lattice.validate()
    shape = (parameters.ny, parameters.nx)
    if state.lattice.shape != shape:
        raise ValueError("lattice shape does not match parameters")
    state.velocity.validate(shape)
    psi = np.asarray(state.electronic_state, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != parameters.n_sites:
        raise ValueError("electronic state dimension does not match parameters")
    if not np.all(np.isfinite(psi)):
        raise ValueError("electronic state must contain only finite values")
    if float(np.linalg.norm(psi)) == 0.0:
        raise ValueError("electronic state must be non-zero")
    return psi


def _validate_policy(policy: str) -> ZeroModePolicy:
    if policy not in ("retain", "project"):
        raise ValueError("zero_mode_policy must be 'retain' or 'project'")
    return policy  # type: ignore[return-value]


def _remove_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values - float(np.mean(values)), dtype=np.float64)


def project_inter_molecular_zero_modes(
    state: CoupledEhrenfestState,
) -> CoupledEhrenfestState:
    """Return a copy with uniform ``vx``/``vy`` coordinate and velocity modes removed.

    ``u`` and the electronic state are unchanged. This is an explicit gauge/
    constraint operation; the coupled integrator never projects an arbitrary
    initial state without the caller selecting the ``project`` policy.
    """
    state.lattice.validate()
    state.velocity.validate(state.lattice.shape)
    return CoupledEhrenfestState(
        LatticeState(
            state.lattice.u.copy(),
            _remove_mean(state.lattice.vx),
            _remove_mean(state.lattice.vy),
        ),
        LatticeVelocity(
            state.velocity.u.copy(),
            _remove_mean(state.velocity.vx),
            _remove_mean(state.velocity.vy),
        ),
        np.asarray(state.electronic_state, dtype=np.complex128).copy(),
    )


def zero_mode_means(state: CoupledEhrenfestState) -> dict[str, float]:
    """Return the four uniform intermolecular coordinate/velocity components."""
    return {
        "vx_A": float(np.mean(state.lattice.vx)),
        "vy_A": float(np.mean(state.lattice.vy)),
        "vx_velocity_A_per_fs": float(np.mean(state.velocity.vx)),
        "vy_velocity_A_per_fs": float(np.mean(state.velocity.vy)),
    }


def thermostatted_kinetic_degrees_of_freedom(
    parameters: StaticPolaronParameters,
    zero_mode_policy: ZeroModePolicy,
) -> int:
    """Return kinetic degrees of freedom for the selected collective-mode policy."""
    policy = _validate_policy(zero_mode_policy)
    return 3 * parameters.n_sites if policy == "retain" else 3 * parameters.n_sites - 2


def _require_projected_state(
    state: CoupledEhrenfestState,
    *,
    tolerance: float = 1.0e-12,
) -> None:
    means = zero_mode_means(state)
    if any(abs(value) > tolerance for value in means.values()):
        raise ValueError(
            "project zero-mode policy requires an initially projected state; "
            "call project_inter_molecular_zero_modes first"
        )


def _project_force(force):
    from ..gradients import LatticeGradient

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


def _force(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    psi: ComplexArray,
    field: UniformElectricField2D | None,
    time_fs: float,
):
    if field is None:
        return ehrenfest_force(lattice, parameters, psi)
    return field_ehrenfest_force(lattice, parameters, psi, field, time_fs)


def _power(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    psi: ComplexArray,
    field: UniformElectricField2D | None,
    time_fs: float,
) -> float:
    if field is None:
        return 0.0
    return field_power(lattice, parameters, psi, field, time_fs)


def coupled_baoab_step(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    rng: Generator,
    time_fs: float,
    dt_fs: float,
    *,
    field: UniformElectricField2D | None = None,
    zero_mode_policy: ZeroModePolicy = "retain",
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> ThermostattedStepResult:
    """Advance one finite-temperature coupled BAOAB/Ehrenfest step.

    For ``zero_mode_policy='retain'`` and zero friction, this function delegates
    directly to the validated D2/D3 step. The reduction is therefore exact and
    no random variate is consumed.

    For finite friction, the lattice path is the two-segment BAOAB drift path.
    The electronic propagator samples that path at its actual internal stage
    times. ``bath_heat_eV`` is the kinetic-energy change across the O step;
    ``field_work_eV`` is the trapezoidal D3 external-field work increment.
    """
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_fs must be finite and dt_fs positive")
    policy = _validate_policy(zero_mode_policy)
    psi = _validate_state(state, parameters)

    # Strong deterministic reduction gates. These paths also guarantee that a
    # frictionless bath consumes no RNG state.
    if bath.is_frictionless and policy == "retain":
        if field is None:
            reduced, h_eval, h_apply = coupled_verlet_step(
                state,
                parameters,
                dt,
                electronic_method=electronic_method,
                krylov_dimension=krylov_dimension,
            )
            return ThermostattedStepResult(reduced, 0.0, 0.0, h_eval, h_apply)
        reduced, work, h_eval, h_apply = coupled_field_verlet_step(
            state,
            parameters,
            field,
            time,
            dt,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        return ThermostattedStepResult(reduced, 0.0, work, h_eval, h_apply)

    if policy == "project":
        _require_projected_state(state)

    power_old = _power(state.lattice, parameters, psi, field, time)
    force_old = _force(state.lattice, parameters, psi, field, time)
    if policy == "project":
        force_old = _project_force(force_old)

    half_velocity = half_kick(state.velocity, force_old, parameters, dt)
    if policy == "project":
        half_velocity = _project_velocity(half_velocity)

    lattice_mid = drift(state.lattice, half_velocity, 0.5 * dt)
    if policy == "project":
        lattice_mid = _project_lattice(lattice_mid)

    kinetic_before_o = lattice_kinetic_energy(half_velocity, parameters)
    thermostatted_velocity = ou_velocity_step(
        half_velocity,
        parameters,
        bath,
        dt,
        rng,
    )
    if policy == "project":
        thermostatted_velocity = _project_velocity(thermostatted_velocity)
    kinetic_after_o = lattice_kinetic_energy(thermostatted_velocity, parameters)
    bath_heat = float(kinetic_after_o - kinetic_before_o)

    lattice_new = drift(lattice_mid, thermostatted_velocity, 0.5 * dt)
    if policy == "project":
        lattice_new = _project_lattice(lattice_new)

    midpoint_time = time + 0.5 * dt
    new_time = time + dt

    def lattice_at(stage_time_fs: float) -> LatticeState:
        stage_time = float(stage_time_fs)
        if stage_time <= midpoint_time:
            fraction = (stage_time - time) / (0.5 * dt)
            fraction = float(np.clip(fraction, 0.0, 1.0))
            return interpolate_lattice(state.lattice, lattice_mid, fraction)
        fraction = (stage_time - midpoint_time) / (0.5 * dt)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return interpolate_lattice(lattice_mid, lattice_new, fraction)

    def hamiltonian_at(stage_time_fs: float):
        lattice_stage = lattice_at(stage_time_fs)
        if field is None:
            return build_sparse_hamiltonian(lattice_stage, parameters)
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
    force_new = _force(lattice_new, parameters, psi_new, field, new_time)
    if policy == "project":
        force_new = _project_force(force_new)
    velocity_new = half_kick(
        thermostatted_velocity,
        force_new,
        parameters,
        dt,
    )
    if policy == "project":
        velocity_new = _project_velocity(velocity_new)

    power_new = _power(lattice_new, parameters, psi_new, field, new_time)
    field_work = float(0.5 * dt * (power_old + power_new))

    return ThermostattedStepResult(
        state=CoupledEhrenfestState(lattice_new, velocity_new, psi_new),
        bath_heat_eV=bath_heat,
        field_work_eV=field_work,
        hamiltonian_evaluations=electronic.hamiltonian_evaluations,
        hamiltonian_applications=electronic.hamiltonian_applications,
    )


def integrate_coupled_baoab(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    rng: Generator,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
    field: UniformElectricField2D | None = None,
    zero_mode_policy: ZeroModePolicy = "retain",
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> ThermostattedPropagationResult:
    """Integrate a finite-temperature one-carrier Ehrenfest trajectory."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    policy = _validate_policy(zero_mode_policy)
    _validate_state(initial_state, parameters)
    if policy == "project":
        _require_projected_state(initial_state)

    current = CoupledEhrenfestState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    current_time = float(initial_time_fs)
    if not np.isfinite(current_time):
        raise ValueError("initial_time_fs must be finite")

    bath_heat = 0.0
    field_work = 0.0
    h_evaluations = 0
    h_applications = 0
    start = perf_counter()
    for _ in range(steps):
        result = coupled_baoab_step(
            current,
            parameters,
            bath,
            rng,
            current_time,
            dt_fs,
            field=field,
            zero_mode_policy=policy,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        current = result.state
        bath_heat += result.bath_heat_eV
        field_work += result.field_work_eV
        h_evaluations += result.hamiltonian_evaluations
        h_applications += result.hamiltonian_applications
        current_time += float(dt_fs)
    elapsed = perf_counter() - start

    return ThermostattedPropagationResult(
        state=current,
        bath_heat_eV=float(bath_heat),
        field_work_eV=float(field_work),
        elapsed_seconds=float(elapsed),
        steps=steps,
        hamiltonian_evaluations=h_evaluations,
        hamiltonian_applications=h_applications,
        zero_mode_policy=policy,
    )
