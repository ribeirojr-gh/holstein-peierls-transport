"""D6b deterministic zero-temperature coupled pair dynamics.

This module extends the validated D2 velocity-Verlet/CF4-Lanczos architecture
to the matrix-free two-particle sectors introduced in D6a:

- a symmetric spatial singlet bipolaron; and
- a distinguishable electron-hole exciton.

The electronic state is always propagated as a complex ordered product-basis
vector of dimension ``N**2``.  No dense pair Hamiltonian is formed.

The pair interaction matrices used by the current static models are independent
of the classical lattice coordinates.  They therefore contribute to the
propagated-state electronic energy but have no explicit derivative with respect
to ``u``, ``vx`` or ``vy``.  They still modify the Ehrenfest force indirectly
through the propagated reduced density matrices.

D6b contains no thermostat, decoherence, electric field, or transport fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from ..exciton.interaction import electron_hole_interaction_matrix
from ..exciton.parameters import ExcitonParameters
from ..exciton.solver import build_carrier_hamiltonian
from ..gradients import LatticeGradient
from ..lattice import LatticeState
from ..two_particle.interaction import pair_interaction_matrix
from ..two_particle.parameters import BipolaronParameters
from ..two_particle.peierls import build_one_particle_hamiltonian
from .coupled import interpolate_lattice
from .ehrenfest import AS2_PER_FS2, LatticeVelocity
from .frozen import HBAR_EV_FS, lanczos_exponential_step
from .pair_frozen import (
    FrozenPairHamiltonianAction,
    PairSector,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
    normalized_pair_state,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
PairParameters: TypeAlias = BipolaronParameters | ExcitonParameters


@dataclass(frozen=True, slots=True)
class PairLatticeMasses:
    """Effective lattice masses in eV fs^2 / angstrom^2.

    Pair static parameter objects intentionally contain only the quantities
    required by the adiabatic solvers.  Dynamics therefore receives masses as
    an explicit independent object instead of silently assuming material values.
    """

    intramolecular: float
    intermolecular: float

    def __post_init__(self) -> None:
        for name, value in (
            ("intramolecular", self.intramolecular),
            ("intermolecular", self.intermolecular),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} mass must be finite and positive")

    @classmethod
    def from_legacy_as2(
        cls,
        intramolecular_eV_as2_per_A2: float,
        intermolecular_eV_as2_per_A2: float,
    ) -> "PairLatticeMasses":
        """Convert the legacy eV as^2/A^2 masses to femtosecond units."""
        return cls(
            float(intramolecular_eV_as2_per_A2) / AS2_PER_FS2,
            float(intermolecular_eV_as2_per_A2) / AS2_PER_FS2,
        )


@dataclass(frozen=True, slots=True)
class PairCoupledState:
    """Pair electronic state and classical lattice variables at one time."""

    lattice: LatticeState
    velocity: LatticeVelocity
    electronic_state: ComplexArray


@dataclass(frozen=True, slots=True)
class PairDynamicEnergy:
    """Propagated-state energy decomposition for D6b."""

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


@dataclass(frozen=True, slots=True)
class PairPropagationResult:
    """Final D6b state and numerical work counters."""

    state: PairCoupledState
    elapsed_seconds: float
    steps: int | None = None
    rhs_evaluations: int | None = None
    hamiltonian_evaluations: int = 0
    hamiltonian_applications: int = 0
    success: bool = True
    message: str = ""


class MovingPairHamiltonianFactory:
    """Build complex-safe pair actions while caching lattice-independent terms."""

    def __init__(self, sector: PairSector, parameters: PairParameters) -> None:
        self.sector = sector
        self.parameters = parameters
        if sector == "bipolaron":
            if not isinstance(parameters, BipolaronParameters):
                raise TypeError("bipolaron sector requires BipolaronParameters")
            self.interaction = pair_interaction_matrix(parameters)
        elif sector == "exciton":
            if not isinstance(parameters, ExcitonParameters):
                raise TypeError("exciton sector requires ExcitonParameters")
            self.interaction = electron_hole_interaction_matrix(parameters)
        else:
            raise ValueError("sector must be 'bipolaron' or 'exciton'")

    @property
    def n_sites(self) -> int:
        return int(self.parameters.n_sites)

    @property
    def dimension(self) -> int:
        return self.n_sites * self.n_sites

    def at(self, lattice: LatticeState) -> FrozenPairHamiltonianAction:
        """Return the pair Hamiltonian action at one lattice configuration."""
        if self.sector == "bipolaron":
            assert isinstance(self.parameters, BipolaronParameters)
            h1 = build_one_particle_hamiltonian(lattice, self.parameters)
            return FrozenPairHamiltonianAction(
                h1,
                h1,
                self.interaction,
                "bipolaron",
            )
        assert isinstance(self.parameters, ExcitonParameters)
        return FrozenPairHamiltonianAction(
            build_carrier_hamiltonian(lattice, self.parameters, "electron"),
            build_carrier_hamiltonian(lattice, self.parameters, "hole"),
            self.interaction,
            "exciton",
        )


def _weighted_pair_action(
    first: FrozenPairHamiltonianAction,
    first_weight: float,
    second: FrozenPairHamiltonianAction,
    second_weight: float,
) -> FrozenPairHamiltonianAction:
    """Form a weighted pair action without materializing the N^2 Hamiltonian."""
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


def _validate_pair_state(
    state: PairCoupledState,
    parameters: PairParameters,
) -> ComplexArray:
    state.lattice.validate()
    shape = (parameters.ny, parameters.nx)
    if state.lattice.shape != shape:
        raise ValueError("lattice shape does not match pair parameters")
    state.velocity.validate(shape)
    psi = np.asarray(state.electronic_state, dtype=np.complex128).reshape(-1)
    if psi.size != parameters.n_sites * parameters.n_sites:
        raise ValueError("pair electronic state has the wrong dimension")
    if not np.all(np.isfinite(psi)) or float(np.linalg.norm(psi)) == 0.0:
        raise ValueError("pair electronic state must be finite and non-zero")
    return psi


def pair_lattice_energy(
    lattice: LatticeState,
    parameters: PairParameters,
) -> tuple[float, float]:
    """Return intramolecular and intermolecular elastic energies in eV."""
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match pair parameters")
    intra = 0.5 * parameters.k1 * float(np.sum(lattice.u * lattice.u))
    dx = np.roll(lattice.vx, -1, axis=1) - lattice.vx
    dy = np.roll(lattice.vy, -1, axis=0) - lattice.vy
    inter = 0.5 * parameters.k2 * float(
        np.sum(dx * dx) + np.sum(dy * dy)
    )
    return intra, inter


def pair_electronic_energy(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
) -> float:
    """Return <Psi|H_pair(q)|Psi>/<Psi|Psi> in eV."""
    if factory is None:
        factory = MovingPairHamiltonianFactory(sector, parameters)
    psi = normalized_pair_state(electronic_state, parameters.n_sites)
    action = factory.at(lattice)
    value = np.vdot(psi, action(psi))
    if abs(float(value.imag)) > 1.0e-10:
        raise FloatingPointError("pair electronic energy expectation is not real")
    return float(value.real)


def _neighbour_indices(parameters: PairParameters):
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    return (
        sites,
        np.roll(sites, 1, axis=1),
        np.roll(sites, -1, axis=1),
        np.roll(sites, 1, axis=0),
        np.roll(sites, -1, axis=0),
    )


def pair_ehrenfest_gradient(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
) -> LatticeGradient:
    """Return d[V_lattice + <H_pair>]/dq for a propagated complex pair state."""
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match pair parameters")
    sites, left, right, up, down = _neighbour_indices(parameters)

    vx_left = np.roll(lattice.vx, 1, axis=1)
    vx_right = np.roll(lattice.vx, -1, axis=1)
    vy_up = np.roll(lattice.vy, 1, axis=0)
    vy_down = np.roll(lattice.vy, -1, axis=0)

    if sector == "bipolaron":
        if not isinstance(parameters, BipolaronParameters):
            raise TypeError("bipolaron sector requires BipolaronParameters")
        gamma = bipolaron_one_body_density_matrix(
            electronic_state, parameters.n_sites
        )
        density = np.real(np.diag(gamma)).reshape(lattice.shape, order="C")
        coh_left = np.real(gamma[sites, left])
        coh_right = np.real(gamma[sites, right])
        coh_up = np.real(gamma[sites, up])
        coh_down = np.real(gamma[sites, down])
        grad_u = parameters.k1 * lattice.u + parameters.alpha_intra * density
        grad_vx = (
            parameters.k2 * (2.0 * lattice.vx - vx_left - vx_right)
            + 2.0 * parameters.alpha_interx * (coh_left - coh_right)
        )
        grad_vy = (
            parameters.k2 * (2.0 * lattice.vy - vy_up - vy_down)
            + 2.0 * parameters.alpha_intery * (coh_up - coh_down)
        )
    elif sector == "exciton":
        if not isinstance(parameters, ExcitonParameters):
            raise TypeError("exciton sector requires ExcitonParameters")
        gamma_e, gamma_h = exciton_one_body_density_matrices(
            electronic_state, parameters.n_sites
        )
        density_e = np.real(np.diag(gamma_e)).reshape(lattice.shape, order="C")
        density_h = np.real(np.diag(gamma_h)).reshape(lattice.shape, order="C")
        grad_u = (
            parameters.k1 * lattice.u
            + parameters.electron_alpha_intra * density_e
            + parameters.hole_alpha_intra * density_h
        )
        grad_vx = parameters.k2 * (2.0 * lattice.vx - vx_left - vx_right)
        grad_vx += 2.0 * parameters.electron_alpha_interx * (
            np.real(gamma_e[sites, left]) - np.real(gamma_e[sites, right])
        )
        grad_vx += 2.0 * parameters.hole_alpha_interx * (
            np.real(gamma_h[sites, left]) - np.real(gamma_h[sites, right])
        )
        grad_vy = parameters.k2 * (2.0 * lattice.vy - vy_up - vy_down)
        grad_vy += 2.0 * parameters.electron_alpha_intery * (
            np.real(gamma_e[sites, up]) - np.real(gamma_e[sites, down])
        )
        grad_vy += 2.0 * parameters.hole_alpha_intery * (
            np.real(gamma_h[sites, up]) - np.real(gamma_h[sites, down])
        )
    else:
        raise ValueError("sector must be 'bipolaron' or 'exciton'")

    return LatticeGradient(
        np.asarray(grad_u, dtype=np.float64),
        np.asarray(grad_vx, dtype=np.float64),
        np.asarray(grad_vy, dtype=np.float64),
    )


def pair_ehrenfest_force(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
) -> LatticeGradient:
    """Return the classical force -dE/dq."""
    gradient = pair_ehrenfest_gradient(
        lattice, sector, parameters, electronic_state
    )
    return LatticeGradient(-gradient.u, -gradient.vx, -gradient.vy)


def pair_lattice_acceleration(
    force: LatticeGradient,
    masses: PairLatticeMasses,
    shape: tuple[int, int],
) -> LatticeGradient:
    """Convert pair-model lattice forces to accelerations in A/fs^2."""
    if force.u.shape != shape or force.vx.shape != shape or force.vy.shape != shape:
        raise ValueError("force shape does not match lattice")
    return LatticeGradient(
        np.asarray(force.u / masses.intramolecular, dtype=np.float64),
        np.asarray(force.vx / masses.intermolecular, dtype=np.float64),
        np.asarray(force.vy / masses.intermolecular, dtype=np.float64),
    )


def pair_half_kick(
    velocity: LatticeVelocity,
    force: LatticeGradient,
    masses: PairLatticeMasses,
    dt_fs: float,
) -> LatticeVelocity:
    """Apply one half velocity kick for D6b."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    velocity.validate()
    acceleration = pair_lattice_acceleration(force, masses, velocity.shape)
    return LatticeVelocity(
        np.asarray(velocity.u + 0.5 * dt * acceleration.u, dtype=np.float64),
        np.asarray(velocity.vx + 0.5 * dt * acceleration.vx, dtype=np.float64),
        np.asarray(velocity.vy + 0.5 * dt * acceleration.vy, dtype=np.float64),
    )


def pair_lattice_kinetic_energy(
    velocity: LatticeVelocity,
    masses: PairLatticeMasses,
) -> float:
    """Return classical pair-model lattice kinetic energy in eV."""
    velocity.validate()
    return float(
        0.5 * masses.intramolecular * np.sum(velocity.u * velocity.u)
        + 0.5
        * masses.intermolecular
        * (np.sum(velocity.vx * velocity.vx) + np.sum(velocity.vy * velocity.vy))
    )


def pair_dynamic_total_energy(
    state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
) -> PairDynamicEnergy:
    """Return the deterministic D6b propagated-state energy decomposition."""
    psi = _validate_pair_state(state, parameters)
    intra, inter = pair_lattice_energy(state.lattice, parameters)
    return PairDynamicEnergy(
        intramolecular_lattice=intra,
        intermolecular_lattice=inter,
        lattice_kinetic=pair_lattice_kinetic_energy(state.velocity, masses),
        electronic=pair_electronic_energy(
            state.lattice,
            sector,
            parameters,
            psi,
            factory=factory,
        ),
    )


def pair_cfm4_lanczos_step(
    factory: MovingPairHamiltonianFactory,
    lattice_start: LatticeState,
    lattice_end: LatticeState,
    electronic_state: ArrayLike,
    dt_fs: float,
    *,
    krylov_dimension: int = 8,
    breakdown_tolerance: float = 1.0e-13,
) -> tuple[ComplexArray, int, int]:
    """Propagate a pair state over a linearly moving lattice with CF4-Lanczos."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    psi = normalized_pair_state(electronic_state, factory.n_sites)
    sqrt3 = float(np.sqrt(3.0))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a1 = (3.0 - 2.0 * sqrt3) / 12.0
    a2 = (3.0 + 2.0 * sqrt3) / 12.0

    first_stage = factory.at(interpolate_lattice(lattice_start, lattice_end, c1))
    second_stage = factory.at(interpolate_lattice(lattice_start, lattice_end, c2))
    right = _weighted_pair_action(first_stage, a2, second_stage, a1)
    left = _weighted_pair_action(first_stage, a1, second_stage, a2)

    first = lanczos_exponential_step(
        right,
        psi,
        dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    second = lanczos_exponential_step(
        left,
        first.state,
        dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    return (
        np.asarray(second.state, dtype=np.complex128),
        2,
        int(first.hamiltonian_applications + second.hamiltonian_applications),
    )


def pair_coupled_verlet_step(
    state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    dt_fs: float,
    *,
    krylov_dimension: int = 8,
    factory: MovingPairHamiltonianFactory | None = None,
) -> tuple[PairCoupledState, int, int]:
    """Advance one zero-field pair Ehrenfest velocity-Verlet/CF4 step."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    psi = _validate_pair_state(state, parameters)
    if factory is None:
        factory = MovingPairHamiltonianFactory(sector, parameters)

    force_old = pair_ehrenfest_force(state.lattice, sector, parameters, psi)
    half_velocity = pair_half_kick(state.velocity, force_old, masses, dt)
    lattice_new = LatticeState(
        np.asarray(state.lattice.u + dt * half_velocity.u, dtype=np.float64),
        np.asarray(state.lattice.vx + dt * half_velocity.vx, dtype=np.float64),
        np.asarray(state.lattice.vy + dt * half_velocity.vy, dtype=np.float64),
    )
    psi_new, h_evaluations, h_applications = pair_cfm4_lanczos_step(
        factory,
        state.lattice,
        lattice_new,
        psi,
        dt,
        krylov_dimension=krylov_dimension,
    )
    force_new = pair_ehrenfest_force(
        lattice_new, sector, parameters, psi_new
    )
    velocity_new = pair_half_kick(
        half_velocity, force_new, masses, dt
    )
    return (
        PairCoupledState(lattice_new, velocity_new, psi_new),
        h_evaluations,
        h_applications,
    )


def integrate_pair_coupled_verlet(
    initial_state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    *,
    dt_fs: float,
    steps: int,
    krylov_dimension: int = 8,
) -> PairPropagationResult:
    """Integrate a deterministic D6b pair trajectory."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    psi = _validate_pair_state(initial_state, parameters)
    current = PairCoupledState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        psi.copy(),
    )
    factory = MovingPairHamiltonianFactory(sector, parameters)
    evaluations = 0
    applications = 0
    start = perf_counter()
    for _ in range(steps):
        current, e, a = pair_coupled_verlet_step(
            current,
            sector,
            parameters,
            masses,
            dt_fs,
            krylov_dimension=krylov_dimension,
            factory=factory,
        )
        evaluations += e
        applications += a
    return PairPropagationResult(
        state=current,
        elapsed_seconds=float(perf_counter() - start),
        steps=int(steps),
        hamiltonian_evaluations=int(evaluations),
        hamiltonian_applications=int(applications),
    )


def _pack_reference_state(
    state: PairCoupledState,
    parameters: PairParameters,
) -> ComplexArray:
    psi = _validate_pair_state(state, parameters)
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
    parameters: PairParameters,
) -> PairCoupledState:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    n = parameters.n_sites
    pair_dimension = n * n
    if vector.size != pair_dimension + 6 * n:
        raise ValueError("packed pair coupled state has the wrong size")
    psi = np.asarray(vector[:pair_dimension], dtype=np.complex128)
    blocks = []
    start = pair_dimension
    for index in range(6):
        block = np.real(vector[start + index * n : start + (index + 1) * n])
        blocks.append(block)
    shape = (parameters.ny, parameters.nx)
    return PairCoupledState(
        LatticeState(
            blocks[0].reshape(shape, order="C"),
            blocks[1].reshape(shape, order="C"),
            blocks[2].reshape(shape, order="C"),
        ),
        LatticeVelocity(
            blocks[3].reshape(shape, order="C"),
            blocks[4].reshape(shape, order="C"),
            blocks[5].reshape(shape, order="C"),
        ),
        psi,
    )


def integrate_pair_coupled_dop853(
    initial_state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    *,
    final_time_fs: float,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_step_fs: float = np.inf,
) -> PairPropagationResult:
    """Integrate the complete D6b pair ODE with adaptive DOP853 as reference."""
    final_time = float(final_time_fs)
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time_fs must be finite and positive")
    if rtol <= 0.0 or atol <= 0.0 or max_step_fs <= 0.0:
        raise ValueError("DOP853 tolerances and max_step_fs must be positive")
    y0 = _pack_reference_state(initial_state, parameters)
    factory = MovingPairHamiltonianFactory(sector, parameters)
    evaluations = 0

    def rhs(_time: float, values: ComplexArray) -> ComplexArray:
        nonlocal evaluations
        current = _unpack_reference_state(values, parameters)
        psi = current.electronic_state
        action = factory.at(current.lattice)
        dpsi = np.asarray(
            (-1.0j / HBAR_EV_FS) * action(psi), dtype=np.complex128
        )
        force = pair_ehrenfest_force(
            current.lattice, sector, parameters, psi
        )
        acceleration = pair_lattice_acceleration(
            force, masses, current.lattice.shape
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
    return PairPropagationResult(
        state=_unpack_reference_state(solution.y[:, -1], parameters),
        elapsed_seconds=float(elapsed),
        steps=max(0, int(solution.t.size - 1)),
        rhs_evaluations=int(solution.nfev),
        hamiltonian_evaluations=int(evaluations),
        hamiltonian_applications=int(evaluations),
        success=bool(solution.success),
        message=str(solution.message),
    )
