"""Adaptive full-system reference integrator for D3.

The D3 production candidate uses a velocity-Verlet / electronic-propagator
splitting.  This module supplies an independent reference by integrating the
complete coupled electron-lattice ODE and the accumulated external work in one
DOP853 solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .classical import lattice_acceleration
from .coupled import CoupledEhrenfestState
from .driven import field_ehrenfest_force, field_power
from .ehrenfest import LatticeVelocity
from .field import UniformElectricField2D, build_sparse_field_hamiltonian
from .frozen import HBAR_EV_FS

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class DrivenReferenceResult:
    """Final D3 reference state and integrated external work."""

    state: CoupledEhrenfestState
    field_work_eV: float
    elapsed_seconds: float
    accepted_steps: int
    rhs_evaluations: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    success: bool
    message: str


def _validate_state(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
) -> None:
    state.lattice.validate()
    expected_shape = (parameters.ny, parameters.nx)
    if state.lattice.shape != expected_shape:
        raise ValueError("lattice shape does not match parameters")
    state.velocity.validate(expected_shape)
    psi = np.asarray(state.electronic_state, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != parameters.n_sites:
        raise ValueError("electronic state dimension does not match parameters")
    if not np.all(np.isfinite(psi)) or float(np.linalg.norm(psi)) == 0.0:
        raise ValueError("electronic state must be finite and non-zero")


def _pack_state_with_work(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    work_eV: float,
) -> ComplexArray:
    _validate_state(state, parameters)
    work = float(work_eV)
    if not np.isfinite(work):
        raise ValueError("work_eV must be finite")
    arrays = (
        np.asarray(state.electronic_state, dtype=np.complex128),
        state.lattice.u.ravel(order="C"),
        state.lattice.vx.ravel(order="C"),
        state.lattice.vy.ravel(order="C"),
        state.velocity.u.ravel(order="C"),
        state.velocity.vx.ravel(order="C"),
        state.velocity.vy.ravel(order="C"),
        np.asarray([work], dtype=np.float64),
    )
    return np.concatenate([np.asarray(a, dtype=np.complex128) for a in arrays])


def _unpack_state_with_work(
    values: ArrayLike,
    parameters: StaticPolaronParameters,
) -> tuple[CoupledEhrenfestState, float]:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    n = parameters.n_sites
    if vector.size != 7 * n + 1:
        raise ValueError("packed driven state has the wrong size")

    psi = np.asarray(vector[:n], dtype=np.complex128)
    blocks = [
        np.real(vector[(i * n) : ((i + 1) * n)])
        for i in range(1, 7)
    ]
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
    work_value = vector[-1]
    if abs(float(np.imag(work_value))) > 1.0e-12:
        raise FloatingPointError("integrated field work acquired an imaginary part")
    return CoupledEhrenfestState(lattice, velocity, psi), float(np.real(work_value))


def integrate_coupled_field_dop853(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    *,
    final_time_fs: float,
    initial_time_fs: float = 0.0,
    initial_work_eV: float = 0.0,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_step_fs: float = np.inf,
) -> DrivenReferenceResult:
    """Integrate the complete D3 ODE plus external work with DOP853."""
    initial_time = float(initial_time_fs)
    final_time = float(final_time_fs)
    if not np.isfinite(initial_time) or not np.isfinite(final_time):
        raise ValueError("integration times must be finite")
    if final_time <= initial_time:
        raise ValueError("final_time_fs must exceed initial_time_fs")
    if rtol <= 0.0 or atol <= 0.0 or max_step_fs <= 0.0:
        raise ValueError("DOP853 tolerances and max_step_fs must be positive")

    y0 = _pack_state_with_work(initial_state, parameters, initial_work_eV)
    evaluations = 0

    def rhs(time_fs: float, values: ComplexArray) -> ComplexArray:
        nonlocal evaluations
        current, _work = _unpack_state_with_work(values, parameters)
        psi = current.electronic_state
        hamiltonian = build_sparse_field_hamiltonian(
            current.lattice,
            parameters,
            field,
            time_fs,
        )
        dpsi = np.asarray(
            (-1.0j / HBAR_EV_FS) * (hamiltonian @ psi),
            dtype=np.complex128,
        )
        force = field_ehrenfest_force(
            current.lattice,
            parameters,
            psi,
            field,
            time_fs,
        )
        acceleration = lattice_acceleration(force, parameters)
        dwork = field_power(
            current.lattice,
            parameters,
            psi,
            field,
            time_fs,
        )
        evaluations += 1
        derivatives = (
            dpsi,
            current.velocity.u.ravel(order="C"),
            current.velocity.vx.ravel(order="C"),
            current.velocity.vy.ravel(order="C"),
            acceleration.u.ravel(order="C"),
            acceleration.vx.ravel(order="C"),
            acceleration.vy.ravel(order="C"),
            np.asarray([dwork], dtype=np.float64),
        )
        return np.concatenate(
            [np.asarray(a, dtype=np.complex128) for a in derivatives]
        )

    start = perf_counter()
    solution = solve_ivp(
        rhs,
        (initial_time, final_time),
        y0,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_fs),
    )
    elapsed = perf_counter() - start
    final_state, final_work = _unpack_state_with_work(
        solution.y[:, -1],
        parameters,
    )
    return DrivenReferenceResult(
        state=final_state,
        field_work_eV=final_work,
        elapsed_seconds=float(elapsed),
        accepted_steps=max(0, int(solution.t.size - 1)),
        rhs_evaluations=int(solution.nfev),
        hamiltonian_evaluations=evaluations,
        hamiltonian_applications=evaluations,
        success=bool(solution.success),
        message=str(solution.message),
    )
