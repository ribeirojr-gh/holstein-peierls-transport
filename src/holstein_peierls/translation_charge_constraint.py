"""Charge-constrained one-site polaron translation diagnostics.

IP0c demonstrated that a single global lattice-projection coordinate can be
satisfied while the electronic polaron remains in an endpoint localization
basin.  This module therefore defines the translation coordinate from the
*complete endpoint charge densities*.

For a relaxed endpoint A and its exact one-site translation B, let

    w_i = n_i(A) - n_i(B).

The diagonal operator W = diag(w) is normalized so that its expectation differs
by exactly two between the endpoints.  A target value of <W> is imposed by a
Lagrange multiplier lambda through the auxiliary Hamiltonian

    H_lambda(q) = H(q) + lambda W.

The lowest eigenstate of H_lambda is varied until its *unbiased* charge-order
expectation equals the requested target.  The reported physical electronic
energy is always <psi_lambda|H(q)|psi_lambda>; the bias contribution is removed.

This is the exact one-electron constrained minimum for the chosen linear charge
order parameter (up to eigensolver/root tolerances).  It is a diagnostic of
charge transfer on a prescribed lattice image.  It is not yet a relaxed
minimum-energy path, a finite-temperature free-energy barrier, a hopping rate,
or a mobility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh

from .electronic import SolverName, solve_ground_state
from .energy import lattice_energy
from .hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from .lattice import LatticeState
from .parameters import StaticPolaronParameters
from .translation_barrier import (
    Direction,
    interpolate_lattice_states,
    translate_lattice_state,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TranslationChargeOrderParameter:
    """Diagonal electronic order parameter distinguishing translated endpoints."""

    weights: FloatArray
    offset: float
    start_expectation: float
    end_expectation: float

    def target(self, fraction: float) -> float:
        s = float(fraction)
        if not np.isfinite(s) or s < 0.0 or s > 1.0:
            raise ValueError("fraction must lie in [0, 1]")
        return float((1.0 - s) * self.start_expectation + s * self.end_expectation)

    def centered_coordinate(self, expectation: float) -> float:
        """Return the normalized coordinate: +1 at A and -1 at B."""
        return float(expectation - self.offset)


@dataclass(frozen=True, slots=True)
class ChargeConstrainedElectronicState:
    """Lowest physical-energy one-electron state at fixed charge order parameter."""

    wavefunction: FloatArray
    physical_energy_eV: float
    biased_eigenvalue_eV: float
    bias_lambda_eV: float
    expectation: float
    target_expectation: float
    constraint_residual: float
    norm_error: float
    biased_energy_identity_error_eV: float
    diagonalizations: int

    @property
    def charge_density(self) -> FloatArray:
        return np.square(self.wavefunction)


@dataclass(frozen=True, slots=True)
class ChargeConstrainedFrozenImage:
    fraction: float
    target_expectation: float
    achieved_expectation: float
    centered_coordinate: float
    physical_total_energy_eV: float
    unconstrained_total_energy_eV: float
    electronic_physical_energy_eV: float
    unconstrained_electronic_energy_eV: float
    constraint_energy_penalty_eV: float
    bias_lambda_eV: float
    constraint_residual: float
    norm_error: float
    biased_energy_identity_error_eV: float
    diagonalizations: int
    ipr: float
    participation_number: float
    source_population: float
    target_population: float


@dataclass(frozen=True, slots=True)
class ChargeConstrainedFrozenProfile:
    direction: Direction
    source_site: int
    target_site: int
    order_parameter: TranslationChargeOrderParameter
    images: tuple[ChargeConstrainedFrozenImage, ...]
    barrier_eV: float
    barrier_image_index: int
    endpoint_energy_mismatch_eV: float
    maximum_constraint_residual: float
    maximum_norm_error: float
    maximum_biased_identity_error_eV: float
    minimum_constraint_energy_penalty_eV: float


def _normalized_vector(values: ArrayLike, dimension: int) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != dimension:
        raise ValueError("wavefunction dimension does not match Hamiltonian")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("wavefunction must have finite non-zero norm")
    return vector / norm


def translation_charge_order_parameter(
    relaxed_state: LatticeState,
    parameters: StaticPolaronParameters,
    direction: Direction,
    *,
    solver: SolverName = "sparse",
) -> TranslationChargeOrderParameter:
    """Build a full-charge-cloud order parameter for a one-site translation.

    The raw endpoint-density difference is rescaled so that the two endpoint
    expectations differ by exactly two.  After subtracting ``offset`` the
    endpoint coordinates are therefore +1 and -1 to numerical precision.
    """
    relaxed_state.validate()
    if relaxed_state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    translated = translate_lattice_state(relaxed_state, direction)
    ground_a = solve_ground_state(relaxed_state, parameters, solver=solver)
    ground_b = solve_ground_state(translated, parameters, solver=solver)
    density_a = np.square(np.asarray(ground_a.wavefunction, dtype=np.float64))
    density_b = np.square(np.asarray(ground_b.wavefunction, dtype=np.float64))
    raw = density_a - density_b
    raw_start = float(np.dot(raw, density_a))
    raw_end = float(np.dot(raw, density_b))
    half_span = 0.5 * (raw_start - raw_end)
    if not np.isfinite(half_span) or abs(half_span) <= 1.0e-12:
        raise ValueError("translated endpoint charge densities are not distinguishable")
    weights = np.asarray(raw / half_span, dtype=np.float64)
    start = float(np.dot(weights, density_a))
    end = float(np.dot(weights, density_b))
    if start < end:
        weights = -weights
        start = -start
        end = -end
    offset = 0.5 * (start + end)
    return TranslationChargeOrderParameter(
        weights=weights,
        offset=float(offset),
        start_expectation=float(start),
        end_expectation=float(end),
    )


def _lowest_biased_state(
    hamiltonian: FloatArray | csr_matrix,
    weights: FloatArray,
    bias_lambda_eV: float,
    *,
    solver: SolverName,
    initial_wavefunction: FloatArray | None = None,
) -> tuple[float, FloatArray]:
    n = int(weights.size)
    lam = float(bias_lambda_eV)
    if solver == "sparse" and n >= 4:
        if not issparse(hamiltonian):
            base = csr_matrix(hamiltonian)
        else:
            base = hamiltonian.tocsr()
        biased = base + diags(lam * weights, offsets=0, format="csr")
        if initial_wavefunction is None:
            v0 = np.ones(n, dtype=np.float64)
            v0 /= np.linalg.norm(v0)
        else:
            v0 = _normalized_vector(initial_wavefunction, n)
        values, vectors = eigsh(
            biased,
            k=1,
            which="SA",
            tol=1.0e-12,
            maxiter=10_000,
            v0=v0,
        )
        energy = float(values[0])
        wavefunction = np.asarray(vectors[:, 0], dtype=np.float64)
    else:
        base_dense = (
            np.asarray(hamiltonian.toarray(), dtype=np.float64)
            if issparse(hamiltonian)
            else np.asarray(hamiltonian, dtype=np.float64)
        )
        biased = base_dense.copy()
        biased[np.diag_indices_from(biased)] += lam * weights
        values, vectors = eigh(
            biased,
            subset_by_index=(0, 0),
            driver="evr",
            overwrite_a=False,
            check_finite=False,
        )
        energy = float(values[0])
        wavefunction = np.asarray(vectors[:, 0], dtype=np.float64)
    anchor = int(np.argmax(np.abs(wavefunction)))
    if wavefunction[anchor] < 0.0:
        wavefunction = -wavefunction
    return energy, _normalized_vector(wavefunction, n)


def solve_charge_constrained_state(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    order_parameter: TranslationChargeOrderParameter,
    target_expectation: float,
    *,
    solver: SolverName = "sparse",
    expectation_tolerance: float = 1.0e-10,
    initial_bias_eV: float = 0.05,
    maximum_bias_eV: float = 20.0,
    initial_wavefunction: FloatArray | None = None,
) -> ChargeConstrainedElectronicState:
    """Solve the Lagrange-multiplier constrained one-electron state.

    ``lambda`` is bracketed automatically.  The physical energy is evaluated
    with the unbiased Holstein-Peierls Hamiltonian even though the auxiliary
    eigenproblem contains ``lambda * W``.
    """
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    weights = np.asarray(order_parameter.weights, dtype=np.float64).reshape(-1)
    if weights.size != parameters.n_sites or not np.all(np.isfinite(weights)):
        raise ValueError("order-parameter weights have the wrong dimension")
    target = float(target_expectation)
    if not np.isfinite(target):
        raise ValueError("target expectation must be finite")
    tol = float(expectation_tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("expectation_tolerance must be finite and positive")
    initial_bias = float(initial_bias_eV)
    maximum_bias = float(maximum_bias_eV)
    if initial_bias <= 0.0 or maximum_bias <= initial_bias:
        raise ValueError("bias bounds must satisfy 0 < initial < maximum")
    if target < float(np.min(weights)) - tol or target > float(np.max(weights)) + tol:
        raise ValueError("target charge expectation lies outside the operator range")

    if solver == "sparse":
        base = build_sparse_hamiltonian(lattice, parameters)
    else:
        base = build_dense_hamiltonian(lattice, parameters)

    evaluations: dict[float, tuple[float, FloatArray, float]] = {}
    last_vector = initial_wavefunction
    diagonalizations = 0

    def evaluate(lam: float) -> tuple[float, FloatArray, float]:
        nonlocal last_vector, diagonalizations
        key = float(lam)
        if key in evaluations:
            return evaluations[key]
        biased_energy, psi = _lowest_biased_state(
            base,
            weights,
            key,
            solver=solver,
            initial_wavefunction=last_vector,
        )
        last_vector = psi
        expectation = float(np.dot(weights, np.square(psi)))
        result = (biased_energy, psi, expectation)
        evaluations[key] = result
        diagonalizations += 1
        return result

    _, _, q0 = evaluate(0.0)
    f0 = q0 - target
    if abs(f0) <= tol:
        root = 0.0
    else:
        radius = initial_bias
        bracket: tuple[float, float] | None = None
        while radius <= maximum_bias * (1.0 + 1.0e-14):
            lo = -radius
            hi = radius
            flo = evaluate(lo)[2] - target
            fhi = evaluate(hi)[2] - target
            if abs(flo) <= tol:
                root = lo
                bracket = None
                break
            if abs(fhi) <= tol:
                root = hi
                bracket = None
                break
            if flo * fhi < 0.0:
                bracket = (lo, hi)
                break
            radius *= 2.0
        else:
            raise RuntimeError("failed to bracket charge-constraint Lagrange multiplier")

        if 'root' not in locals():
            assert bracket is not None
            root = float(
                brentq(
                    lambda lam: evaluate(float(lam))[2] - target,
                    bracket[0],
                    bracket[1],
                    xtol=1.0e-12,
                    rtol=1.0e-12,
                    maxiter=100,
                )
            )

    biased_energy, psi, achieved = evaluate(float(root))
    if issparse(base):
        hpsi = base @ psi
    else:
        hpsi = np.asarray(base, dtype=np.float64) @ psi
    physical = float(np.dot(psi, hpsi))
    norm_error = float(abs(np.dot(psi, psi) - 1.0))
    identity_error = float(
        abs(biased_energy - (physical + float(root) * achieved))
    )
    return ChargeConstrainedElectronicState(
        wavefunction=psi,
        physical_energy_eV=physical,
        biased_eigenvalue_eV=float(biased_energy),
        bias_lambda_eV=float(root),
        expectation=float(achieved),
        target_expectation=target,
        constraint_residual=float(abs(achieved - target)),
        norm_error=norm_error,
        biased_energy_identity_error_eV=identity_error,
        diagonalizations=int(diagonalizations),
    )


def charge_constrained_frozen_profile(
    relaxed_state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    direction: Direction = "+x",
    image_count: int = 7,
    solver: SolverName = "sparse",
    expectation_tolerance: float = 1.0e-10,
) -> ChargeConstrainedFrozenProfile:
    """Evaluate charge-constrained states on the frozen endpoint interpolation."""
    relaxed_state.validate()
    if relaxed_state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    count = int(image_count)
    if count < 3 or count % 2 == 0:
        raise ValueError("image_count must be an odd integer >= 3")

    translated = translate_lattice_state(relaxed_state, direction)
    order = translation_charge_order_parameter(
        relaxed_state, parameters, direction, solver=solver
    )
    ground_a = solve_ground_state(relaxed_state, parameters, solver=solver)
    ground_b = solve_ground_state(translated, parameters, solver=solver)
    source = int(np.argmax(np.square(ground_a.wavefunction)))
    target_site = int(np.argmax(np.square(ground_b.wavefunction)))

    images: list[ChargeConstrainedFrozenImage] = []
    previous_wavefunction: FloatArray | None = None
    for fraction in np.linspace(0.0, 1.0, count):
        lattice = interpolate_lattice_states(relaxed_state, translated, float(fraction))
        unconstrained = solve_ground_state(
            lattice,
            parameters,
            solver=solver,
            initial_wavefunction=previous_wavefunction,
        )
        target_expectation = order.target(float(fraction))
        constrained = solve_charge_constrained_state(
            lattice,
            parameters,
            order,
            target_expectation,
            solver=solver,
            expectation_tolerance=expectation_tolerance,
            initial_wavefunction=unconstrained.wavefunction,
        )
        previous_wavefunction = constrained.wavefunction
        intra, inter = lattice_energy(lattice, parameters)
        lattice_total = float(intra + inter)
        constrained_total = lattice_total + constrained.physical_energy_eV
        unconstrained_total = lattice_total + float(unconstrained.energy)
        population = constrained.charge_density
        ipr = float(np.sum(population * population))
        images.append(
            ChargeConstrainedFrozenImage(
                fraction=float(fraction),
                target_expectation=float(target_expectation),
                achieved_expectation=float(constrained.expectation),
                centered_coordinate=float(
                    order.centered_coordinate(constrained.expectation)
                ),
                physical_total_energy_eV=float(constrained_total),
                unconstrained_total_energy_eV=float(unconstrained_total),
                electronic_physical_energy_eV=float(constrained.physical_energy_eV),
                unconstrained_electronic_energy_eV=float(unconstrained.energy),
                constraint_energy_penalty_eV=float(constrained_total - unconstrained_total),
                bias_lambda_eV=float(constrained.bias_lambda_eV),
                constraint_residual=float(constrained.constraint_residual),
                norm_error=float(constrained.norm_error),
                biased_energy_identity_error_eV=float(
                    constrained.biased_energy_identity_error_eV
                ),
                diagonalizations=int(constrained.diagonalizations),
                ipr=ipr,
                participation_number=float(1.0 / ipr),
                source_population=float(population[source]),
                target_population=float(population[target_site]),
            )
        )

    endpoint_reference = 0.5 * (
        images[0].physical_total_energy_eV + images[-1].physical_total_energy_eV
    )
    energies = np.asarray(
        [image.physical_total_energy_eV for image in images], dtype=np.float64
    )
    barrier_index = int(np.argmax(energies))
    barrier = float(max(0.0, energies[barrier_index] - endpoint_reference))
    return ChargeConstrainedFrozenProfile(
        direction=direction,
        source_site=source,
        target_site=target_site,
        order_parameter=order,
        images=tuple(images),
        barrier_eV=barrier,
        barrier_image_index=barrier_index,
        endpoint_energy_mismatch_eV=float(
            abs(images[-1].physical_total_energy_eV - images[0].physical_total_energy_eV)
        ),
        maximum_constraint_residual=float(
            max(image.constraint_residual for image in images)
        ),
        maximum_norm_error=float(max(image.norm_error for image in images)),
        maximum_biased_identity_error_eV=float(
            max(image.biased_energy_identity_error_eV for image in images)
        ),
        minimum_constraint_energy_penalty_eV=float(
            min(image.constraint_energy_penalty_eV for image in images)
        ),
    )
