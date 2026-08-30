"""Matrix-free adiabatic Holstein-Hubbard singlet bipolaron solver.

The two-particle wavefunction is represented as ``psi[i, j]`` on the ordered
site-product basis. We remain in the spin-singlet sector by starting the
Krylov iteration with a symmetric vector; the Hamiltonian preserves particle
exchange symmetry. The returned state is explicitly symmetrized to remove
round-off-level antisymmetric contamination.

This module deliberately contains *only* the intramolecular Holstein coordinate
``u`` and an on-site Hubbard repulsion ``U``. Peierls coupling is added only
after this reference problem passes its analytic and numerical regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from .parameters import BipolaronParameters

FloatArray = NDArray[np.float64]
InitializationMode = Literal["onsite", "separated", "zero"]


@dataclass(frozen=True, slots=True)
class BipolaronGroundState:
    """Lowest singlet two-particle eigenstate for a fixed lattice distortion."""

    energy: float
    wavefunction: FloatArray

    @property
    def probability(self) -> FloatArray:
        return np.square(self.wavefunction)

    @property
    def exchange_symmetry_error(self) -> float:
        return float(np.max(np.abs(self.wavefunction - self.wavefunction.T)))

    @property
    def one_body_density_matrix(self) -> FloatArray:
        """Spin-summed one-body reduced density matrix with trace two."""
        psi = self.wavefunction
        return 2.0 * (psi @ psi.T)

    @property
    def site_density(self) -> FloatArray:
        return np.diag(self.one_body_density_matrix)

    @property
    def onsite_pair_probability(self) -> float:
        return float(np.sum(np.square(np.diag(self.wavefunction))))


@dataclass(frozen=True, slots=True)
class BipolaronEnergy:
    electronic: float
    lattice: float
    total: float


@dataclass(frozen=True, slots=True)
class BipolaronRelaxationDiagnostics:
    iterations: int
    converged: bool
    final_max_update: float
    final_max_gradient: float


@dataclass(frozen=True, slots=True)
class BipolaronResult:
    u: FloatArray
    ground_state: BipolaronGroundState
    energy: BipolaronEnergy
    diagnostics: BipolaronRelaxationDiagnostics

    @property
    def onsite_pair_probability(self) -> float:
        return self.ground_state.onsite_pair_probability


def _site_neighbours(parameters: BipolaronParameters) -> tuple[FloatArray, FloatArray, FloatArray]:
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    return (
        sites.ravel(),
        np.roll(sites, -1, axis=1).ravel(),
        np.roll(sites, -1, axis=0).ravel(),
    )


def build_one_particle_hamiltonian(
    u: FloatArray,
    parameters: BipolaronParameters,
) -> csr_matrix:
    """Build the one-carrier Holstein Hamiltonian for a fixed ``u`` field."""
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (parameters.ny, parameters.nx):
        raise ValueError("u shape does not match bipolaron parameters")

    n = parameters.n_sites
    diagonal = parameters.alpha_intra * u.ravel(order="C")

    # Avoid duplicate-bond ambiguity for very small periodic dimensions by
    # assigning a dense reference matrix exactly once per directed neighbour.
    if parameters.nx < 3 or parameters.ny < 3:
        h = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(h, diagonal)
        sites = np.arange(n, dtype=np.int64).reshape(parameters.ny, parameters.nx)
        for y in range(parameters.ny):
            for x in range(parameters.nx):
                i = int(sites[y, x])
                jx = int(sites[y, (x + 1) % parameters.nx])
                jy = int(sites[(y + 1) % parameters.ny, x])
                h[i, jx] = h[jx, i] = -parameters.j0x
                h[i, jy] = h[jy, i] = -parameters.j0y
        return csr_matrix(h)

    sites, right, down = _site_neighbours(parameters)
    rows = np.concatenate((sites, sites, right, sites, down))
    cols = np.concatenate((sites, right, sites, down, sites))
    data = np.concatenate(
        (
            diagonal,
            np.full(n, -parameters.j0x),
            np.full(n, -parameters.j0x),
            np.full(n, -parameters.j0y),
            np.full(n, -parameters.j0y),
        )
    )
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _apply_two_particle_hamiltonian(
    psi: FloatArray,
    one_particle_hamiltonian: csr_matrix,
    hubbard_u: float,
) -> FloatArray:
    """Apply ``H1⊗I + I⊗H1 + U delta_ij`` without forming an N^2 matrix."""
    left = one_particle_hamiltonian @ psi
    right = (one_particle_hamiltonian @ psi.T).T
    result = np.asarray(left + right, dtype=np.float64)
    if hubbard_u != 0.0:
        diagonal = np.diag_indices_from(result)
        result[diagonal] += hubbard_u * psi[diagonal]
    return result


def two_particle_linear_operator(
    u: FloatArray,
    parameters: BipolaronParameters,
) -> LinearOperator:
    """Return the matrix-free two-particle Hamiltonian as a LinearOperator."""
    h1 = build_one_particle_hamiltonian(u, parameters)
    n = parameters.n_sites

    def matvec(vector: FloatArray) -> FloatArray:
        psi = np.asarray(vector, dtype=np.float64).reshape((n, n), order="C")
        return _apply_two_particle_hamiltonian(psi, h1, parameters.hubbard_u).ravel(
            order="C"
        )

    return LinearOperator((n * n, n * n), matvec=matvec, dtype=np.float64)


def _normalize_symmetric_wavefunction(wavefunction: FloatArray) -> FloatArray:
    psi = 0.5 * (wavefunction + wavefunction.T)
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm < 1.0e-14:
        raise RuntimeError("singlet projection produced a zero or invalid wavefunction")
    psi /= norm
    anchor = np.unravel_index(int(np.argmax(np.abs(psi))), psi.shape)
    if psi[anchor] < 0.0:
        psi = -psi
    return psi


def _initial_singlet_vector(parameters: BipolaronParameters) -> FloatArray:
    n = parameters.n_sites
    psi = np.zeros((n, n), dtype=np.float64)
    psi[parameters.pair_index, parameters.pair_index] = 1.0
    return psi.ravel(order="C")


def solve_bipolaron_ground_state(
    u: FloatArray,
    parameters: BipolaronParameters,
    *,
    initial_wavefunction: FloatArray | None = None,
) -> BipolaronGroundState:
    """Solve the lowest state in the symmetric spatial (spin-singlet) sector."""
    n = parameters.n_sites
    operator = two_particle_linear_operator(u, parameters)

    if initial_wavefunction is None:
        v0 = _initial_singlet_vector(parameters)
    else:
        psi0 = np.asarray(initial_wavefunction, dtype=np.float64).reshape((n, n))
        v0 = _normalize_symmetric_wavefunction(psi0).ravel(order="C")

    _, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=v0,
        tol=parameters.eigensolver_tolerance,
        maxiter=parameters.eigensolver_max_iterations,
    )
    psi = _normalize_symmetric_wavefunction(eigenvectors[:, 0].reshape((n, n)))

    # Recompute the Rayleigh quotient after explicit singlet projection.
    applied = operator @ psi.ravel(order="C")
    energy = float(np.dot(psi.ravel(order="C"), applied))
    return BipolaronGroundState(energy=energy, wavefunction=psi)


def total_energy(
    u: FloatArray,
    parameters: BipolaronParameters,
    *,
    ground_state: BipolaronGroundState | None = None,
) -> tuple[BipolaronEnergy, BipolaronGroundState]:
    """Return electronic, elastic, and total adiabatic energy."""
    if ground_state is None:
        ground_state = solve_bipolaron_ground_state(u, parameters)
    lattice = 0.5 * parameters.k1 * float(np.sum(np.square(u)))
    return (
        BipolaronEnergy(
            electronic=ground_state.energy,
            lattice=lattice,
            total=ground_state.energy + lattice,
        ),
        ground_state,
    )


def energy_gradient_u(
    u: FloatArray,
    parameters: BipolaronParameters,
    *,
    ground_state: BipolaronGroundState | None = None,
) -> tuple[FloatArray, BipolaronGroundState]:
    """Hellmann-Feynman gradient of the adiabatic energy with respect to ``u``."""
    if ground_state is None:
        ground_state = solve_bipolaron_ground_state(u, parameters)
    density = ground_state.site_density.reshape((parameters.ny, parameters.nx))
    gradient = parameters.k1 * np.asarray(u) + parameters.alpha_intra * density
    return np.asarray(gradient, dtype=np.float64), ground_state


def expectation_energy(
    wavefunction: FloatArray,
    u: FloatArray,
    parameters: BipolaronParameters,
) -> BipolaronEnergy:
    """Evaluate the energy of an arbitrary normalized symmetric trial state."""
    n = parameters.n_sites
    psi = _normalize_symmetric_wavefunction(
        np.asarray(wavefunction, dtype=np.float64).reshape((n, n))
    )
    h1 = build_one_particle_hamiltonian(u, parameters)
    applied = _apply_two_particle_hamiltonian(psi, h1, parameters.hubbard_u)
    electronic = float(np.sum(psi * applied))
    lattice = 0.5 * parameters.k1 * float(np.sum(np.square(u)))
    return BipolaronEnergy(electronic, lattice, electronic + lattice)


def atomic_limit_binding_energy(parameters: BipolaronParameters) -> float:
    """Analytic onsite-vs-separated binding energy A^2/K1 - U at J=0."""
    return parameters.atomic_holstein_pairing_scale - parameters.hubbard_u


def initial_distortion(
    parameters: BipolaronParameters,
    mode: InitializationMode = "onsite",
) -> FloatArray:
    """Generate physically motivated RPROP seeds for competing minima."""
    u = np.zeros((parameters.ny, parameters.nx), dtype=np.float64)
    center = parameters.pair_index
    cy, cx = divmod(center, parameters.nx)

    if mode == "zero":
        return u
    if mode == "onsite":
        u[cy, cx] = -2.0 * parameters.alpha_intra / parameters.k1
        return u
    if mode == "separated":
        u[cy, cx] = -parameters.alpha_intra / parameters.k1
        sy = (cy + parameters.ny // 2) % parameters.ny
        sx = (cx + parameters.nx // 2) % parameters.nx
        u[sy, sx] = -parameters.alpha_intra / parameters.k1
        return u
    raise ValueError(f"unknown bipolaron initialization mode: {mode}")


def relax_static_bipolaron(
    parameters: BipolaronParameters,
    *,
    initialization: InitializationMode = "onsite",
    initial_u: FloatArray | None = None,
) -> BipolaronResult:
    """Relax ``u`` with RPROP and require displacement and force convergence."""
    if initial_u is None:
        u = initial_distortion(parameters, initialization)
    else:
        u = np.asarray(initial_u, dtype=np.float64).copy()
        if u.shape != (parameters.ny, parameters.nx):
            raise ValueError("initial_u shape does not match bipolaron parameters")

    previous_gradient = np.zeros_like(u)
    step_size = np.full_like(u, parameters.update_start)
    cached_state: BipolaronGroundState | None = None
    converged = False
    final_max_update = np.inf
    final_max_gradient = np.inf
    final_energy: BipolaronEnergy | None = None
    final_state: BipolaronGroundState | None = None

    for iteration in range(1, parameters.max_iterations + 1):
        gradient, current_state = energy_gradient_u(
            u, parameters, ground_state=cached_state
        )
        product = gradient * previous_gradient
        positive = product > 0.0
        negative = product < 0.0

        step_size[positive] = np.minimum(
            step_size[positive] * parameters.acceleration_factor,
            parameters.update_max,
        )
        step_size[negative] = np.maximum(
            step_size[negative] * parameters.deceleration_factor,
            parameters.update_min,
        )

        effective_gradient = gradient.copy()
        effective_gradient[negative] = 0.0
        delta = -np.sign(effective_gradient) * step_size
        u += delta

        final_state = solve_bipolaron_ground_state(
            u,
            parameters,
            initial_wavefunction=current_state.wavefunction,
        )
        final_energy, _ = total_energy(u, parameters, ground_state=final_state)
        final_gradient, _ = energy_gradient_u(
            u,
            parameters,
            ground_state=final_state,
        )
        cached_state = final_state
        previous_gradient = effective_gradient

        final_max_update = float(np.max(np.abs(delta)))
        final_max_gradient = float(np.max(np.abs(final_gradient)))
        converged = (
            final_max_update < parameters.convergence_criterion
            and final_max_gradient < parameters.gradient_convergence_criterion
        )
        if converged:
            break

    assert final_energy is not None and final_state is not None
    return BipolaronResult(
        u=u,
        ground_state=final_state,
        energy=final_energy,
        diagnostics=BipolaronRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_max_update,
            final_max_gradient=final_max_gradient,
        ),
    )
