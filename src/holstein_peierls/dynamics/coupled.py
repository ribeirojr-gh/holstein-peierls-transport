"""Coupled one-carrier zero-temperature Ehrenfest propagation for D2.

The production-candidate split uses velocity Verlet for the classical lattice.
Within each full lattice step, the lattice path is linearly interpolated between
its old and drifted coordinates and supplied to the electronic propagator at its
actual internal stage times.  This avoids treating the moving Hamiltonian as
frozen while retaining a simple, auditable second-order classical splitting.

A full-system adaptive DOP853 integrator is provided as an independent tightened
reference.  No thermostat, random force or electric field is present here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from ..hamiltonian import build_sparse_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .classical import drift, half_kick, lattice_acceleration
from .ehrenfest import LatticeVelocity, ehrenfest_force
from .frozen import HBAR_EV_FS
from .time_dependent import cfm4_lanczos_step, rk4_time_dependent_step

ComplexArray = NDArray[np.complex128]
ElectronicMethod = Literal["cfm4_lanczos", "rk4"]


@dataclass(frozen=True, slots=True)
class CoupledEhrenfestState:
    """Electronic and classical variables at one common trajectory time."""

    lattice: LatticeState
    velocity: LatticeVelocity
    electronic_state: ComplexArray


@dataclass(frozen=True, slots=True)
class CoupledPropagationResult:
    """Final coupled state and numerical work counters."""

    state: CoupledEhrenfestState
    elapsed_seconds: float
    steps: int | None = None
    rhs_evaluations: int | None = None
    hamiltonian_evaluations: int = 0
    hamiltonian_applications: int = 0
    success: bool = True
    message: str = ""


def _validate_coupled_state(
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
    if not np.all(np.isfinite(psi)) or float(np.linalg.norm(psi)) == 0.0:
        raise ValueError("electronic state must be finite and non-zero")
    return psi


def interpolate_lattice(
    start: LatticeState,
    end: LatticeState,
    fraction: float,
) -> LatticeState:
    """Linearly interpolate a lattice path for electronic internal stages."""
    start.validate()
    end.validate()
    if start.shape != end.shape:
        raise ValueError("lattice endpoints must have identical shapes")
    value = float(fraction)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("interpolation fraction must lie in [0, 1]")
    return LatticeState(
        np.asarray(start.u + value * (end.u - start.u), dtype=np.float64),
        np.asarray(start.vx + value * (end.vx - start.vx), dtype=np.float64),
        np.asarray(start.vy + value * (end.vy - start.vy), dtype=np.float64),
    )


def coupled_verlet_step(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    dt_fs: float,
    *,
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> tuple[CoupledEhrenfestState, int, int]:
    """Advance one zero-field Ehrenfest step with a moving-lattice H(t).

    The old force supplies the first half kick.  The resulting half-step
    velocity determines the drifted lattice endpoint.  Electronic propagation
    then samples the linear path between the two lattice endpoints.  The new
    propagated state supplies the force for the second half kick.
    """
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    psi = _validate_coupled_state(state, parameters)

    force_old = ehrenfest_force(state.lattice, parameters, psi)
    half_velocity = half_kick(
        state.velocity,
        force_old,
        parameters,
        dt,
    )
    lattice_new = drift(state.lattice, half_velocity, dt)

    def hamiltonian_at(local_time_fs: float):
        fraction = float(local_time_fs) / dt
        lattice_stage = interpolate_lattice(state.lattice, lattice_new, fraction)
        return build_sparse_hamiltonian(lattice_stage, parameters)

    if electronic_method == "cfm4_lanczos":
        electronic = cfm4_lanczos_step(
            hamiltonian_at,
            psi,
            0.0,
            dt,
            krylov_dimension=krylov_dimension,
        )
    elif electronic_method == "rk4":
        electronic = rk4_time_dependent_step(
            hamiltonian_at,
            psi,
            0.0,
            dt,
        )
    else:
        raise ValueError(f"unknown electronic method: {electronic_method}")

    psi_new = np.asarray(electronic.state, dtype=np.complex128)
    force_new = ehrenfest_force(lattice_new, parameters, psi_new)
    velocity_new = half_kick(
        half_velocity,
        force_new,
        parameters,
        dt,
    )
    return (
        CoupledEhrenfestState(lattice_new, velocity_new, psi_new),
        electronic.hamiltonian_evaluations,
        electronic.hamiltonian_applications,
    )


def integrate_coupled_verlet(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    *,
    dt_fs: float,
    steps: int,
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> CoupledPropagationResult:
    """Integrate a deterministic coupled trajectory with the D2 split scheme."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    _validate_coupled_state(initial_state, parameters)
    current = CoupledEhrenfestState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    h_evaluations = 0
    h_applications = 0
    start = perf_counter()
    for _ in range(steps):
        current, evaluations, applications = coupled_verlet_step(
            current,
            parameters,
            dt_fs,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        h_evaluations += evaluations
        h_applications += applications
    elapsed = perf_counter() - start
    return CoupledPropagationResult(
        state=current,
        elapsed_seconds=float(elapsed),
        steps=steps,
        hamiltonian_evaluations=h_evaluations,
        hamiltonian_applications=h_applications,
    )


def _pack_reference_state(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
) -> ComplexArray:
    psi = _validate_coupled_state(state, parameters)
    arrays = (
        psi,
        state.lattice.u.ravel(order="C"),
        state.lattice.vx.ravel(order="C"),
        state.lattice.vy.ravel(order="C"),
        state.velocity.u.ravel(order="C"),
        state.velocity.vx.ravel(order="C"),
        state.velocity.vy.ravel(order="C"),
    )
    return np.concatenate([np.asarray(a, dtype=np.complex128) for a in arrays])


def _unpack_reference_state(
    values: ArrayLike,
    parameters: StaticPolaronParameters,
) -> CoupledEhrenfestState:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    n = parameters.n_sites
    if vector.size != 7 * n:
        raise ValueError("packed coupled state has the wrong size")
    psi = np.asarray(vector[:n], dtype=np.complex128)
    blocks = [np.real(vector[(i * n) : ((i + 1) * n)]) for i in range(1, 7)]
    shape = (parameters.ny, parameters.nx)
    lattice = LatticeState(
        blocks[0].reshape(shape, order="C"),
        blocks[1].reshape(shape, order="C"),
        blocks[2].reshape(shape, order="C"),
    )
    velocity = LatticeVelocity(
        blocks[3].reshape(shape, order="C"),
        blocks[4].reshape(shape, order="C"),
        blocks[5].reshape(shape, order="C"),
    )
    return CoupledEhrenfestState(lattice, velocity, psi)


def integrate_coupled_dop853(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    *,
    final_time_fs: float,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_step_fs: float = np.inf,
) -> CoupledPropagationResult:
    """Integrate the complete coupled ODE with adaptive DOP853 as a reference."""
    final_time = float(final_time_fs)
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time_fs must be finite and positive")
    if rtol <= 0.0 or atol <= 0.0 or max_step_fs <= 0.0:
        raise ValueError("DOP853 tolerances and max_step_fs must be positive")
    y0 = _pack_reference_state(initial_state, parameters)
    evaluations = 0

    def rhs(_time: float, values: ComplexArray) -> ComplexArray:
        nonlocal evaluations
        current = _unpack_reference_state(values, parameters)
        psi = current.electronic_state
        hamiltonian = build_sparse_hamiltonian(current.lattice, parameters)
        dpsi = np.asarray(
            (-1.0j / HBAR_EV_FS) * (hamiltonian @ psi),
            dtype=np.complex128,
        )
        force = ehrenfest_force(current.lattice, parameters, psi)
        acceleration = lattice_acceleration(force, parameters)
        evaluations += 1
        derivatives = (
            dpsi,
            current.velocity.u.ravel(order="C"),
            current.velocity.vx.ravel(order="C"),
            current.velocity.vy.ravel(order="C"),
            acceleration.u.ravel(order="C"),
            acceleration.vx.ravel(order="C"),
            acceleration.vy.ravel(order="C"),
        )
        return np.concatenate(
            [np.asarray(a, dtype=np.complex128) for a in derivatives]
        )

    start = perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        y0,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_fs),
    )
    elapsed = perf_counter() - start
    final_state = _unpack_reference_state(solution.y[:, -1], parameters)
    return CoupledPropagationResult(
        state=final_state,
        elapsed_seconds=float(elapsed),
        steps=max(0, int(solution.t.size - 1)),
        rhs_evaluations=int(solution.nfev),
        hamiltonian_evaluations=evaluations,
        hamiltonian_applications=evaluations,
        success=bool(solution.success),
        message=str(solution.message),
    )
