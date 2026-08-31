"""Generalized Holstein-Peierls mode coupling on a molecular bond graph.

Phase G3 keeps this implementation parallel to the validated production
solvers. It generalizes the two legacy Peierls fields to an arbitrary local
mode vector and assembles hopping, elastic energy, and Hellmann-Feynman forces
by scattering contributions over oriented bond families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .molecular_hamiltonian import MolecularTightBindingModel

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class MolecularModeState:
    """Holstein coordinate ``u`` plus an arbitrary local Peierls-mode vector ``q``."""

    u: FloatArray
    q: FloatArray

    def copy(self) -> "MolecularModeState":
        return MolecularModeState(u=self.u.copy(), q=self.q.copy())


@dataclass(frozen=True, slots=True)
class MolecularPeierlsModel:
    """Generalized static Holstein-Peierls parameters on a molecular graph."""

    tight_binding: MolecularTightBindingModel
    n_modes: int
    holstein_alpha_ev_per_angstrom: float
    holstein_k_ev_per_angstrom2: float
    bond_mode_couplings: tuple[tuple[str, tuple[float, ...]], ...]
    bond_stiffness_matrices: tuple[
        tuple[str, tuple[tuple[float, ...], ...]], ...
    ]

    def __post_init__(self) -> None:
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive")
        if not np.isfinite(self.holstein_alpha_ev_per_angstrom):
            raise ValueError("holstein coupling must be finite")
        if (
            not np.isfinite(self.holstein_k_ev_per_angstrom2)
            or self.holstein_k_ev_per_angstrom2 <= 0.0
        ):
            raise ValueError("holstein stiffness must be finite and positive")

        expected = {family.label for family in self.tight_binding.lattice.bonds}
        coupling_labels = [label for label, _ in self.bond_mode_couplings]
        stiffness_labels = [label for label, _ in self.bond_stiffness_matrices]
        if len(set(coupling_labels)) != len(coupling_labels):
            raise ValueError("bond mode-coupling labels must be unique")
        if len(set(stiffness_labels)) != len(stiffness_labels):
            raise ValueError("bond stiffness labels must be unique")
        if set(coupling_labels) != expected or set(stiffness_labels) != expected:
            raise ValueError(
                "mode-coupling and stiffness labels must match lattice bond families"
            )

        for label, vector in self.bond_mode_couplings:
            values = np.asarray(vector, dtype=np.float64)
            if values.shape != (self.n_modes,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"mode coupling for {label!r} must contain n_modes finite values"
                )

        for label, matrix in self.bond_stiffness_matrices:
            values = np.asarray(matrix, dtype=np.float64)
            if values.shape != (self.n_modes, self.n_modes):
                raise ValueError(
                    f"stiffness for {label!r} must have shape (n_modes, n_modes)"
                )
            if not np.all(np.isfinite(values)) or not np.allclose(
                values, values.T, rtol=0.0, atol=1.0e-14
            ):
                raise ValueError(
                    f"stiffness for {label!r} must be finite and symmetric"
                )
            if float(np.min(np.linalg.eigvalsh(values))) < -1.0e-12:
                raise ValueError(
                    f"stiffness for {label!r} must be positive semidefinite"
                )

    @property
    def coupling_vectors(self) -> dict[str, FloatArray]:
        return {
            label: np.asarray(vector, dtype=np.float64)
            for label, vector in self.bond_mode_couplings
        }

    @property
    def stiffness_matrices(self) -> dict[str, FloatArray]:
        return {
            label: np.asarray(matrix, dtype=np.float64)
            for label, matrix in self.bond_stiffness_matrices
        }


def _validate_state(state: MolecularModeState, model: MolecularPeierlsModel) -> None:
    n_sites = model.tight_binding.lattice.n_sites
    u = np.asarray(state.u, dtype=np.float64)
    q = np.asarray(state.q, dtype=np.float64)
    if u.shape != (n_sites,):
        raise ValueError("u must have shape (n_sites,)")
    if q.shape != (n_sites, model.n_modes):
        raise ValueError("q must have shape (n_sites, n_modes)")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(q)):
        raise ValueError("lattice coordinates must be finite")


def build_molecular_peierls_hamiltonian(
    state: MolecularModeState,
    model: MolecularPeierlsModel,
) -> csr_matrix:
    """Build the real symmetric graph Hamiltonian for one lattice configuration."""
    _validate_state(state, model)
    lattice = model.tight_binding.lattice
    transfer = model.tight_binding.transfer_integrals
    coupling = model.coupling_vectors
    onsite_basis = model.tight_binding.onsite_energies

    diagonal = np.empty(lattice.n_sites, dtype=np.float64)
    for site in range(lattice.n_sites):
        _, _, basis = lattice.site_coordinates(site)
        diagonal[site] = (
            onsite_basis[basis]
            + model.holstein_alpha_ev_per_angstrom * state.u[site]
        )

    bonds = lattice.generated_bonds()
    source = np.fromiter((bond.source for bond in bonds), dtype=np.int64)
    target = np.fromiter((bond.target for bond in bonds), dtype=np.int64)
    hopping = np.empty(len(bonds), dtype=np.float64)
    for index, bond in enumerate(bonds):
        delta_q = state.q[bond.family_target] - state.q[bond.family_source]
        hopping[index] = transfer[bond.family] + float(
            np.dot(coupling[bond.family], delta_q)
        )

    sites = np.arange(lattice.n_sites, dtype=np.int64)
    rows = np.concatenate((sites, source, target))
    cols = np.concatenate((sites, target, source))
    data = np.concatenate((diagonal, hopping, hopping))
    return coo_matrix(
        (data, (rows, cols)), shape=(lattice.n_sites, lattice.n_sites)
    ).tocsr()


def molecular_lattice_energy(
    state: MolecularModeState,
    model: MolecularPeierlsModel,
) -> tuple[float, float]:
    """Return Holstein and generalized Peierls elastic energies."""
    _validate_state(state, model)
    intra = 0.5 * model.holstein_k_ev_per_angstrom2 * float(
        np.dot(state.u, state.u)
    )
    stiffness = model.stiffness_matrices
    inter = 0.0
    for bond in model.tight_binding.lattice.generated_bonds():
        delta_q = state.q[bond.family_target] - state.q[bond.family_source]
        inter += 0.5 * float(delta_q @ stiffness[bond.family] @ delta_q)
    return intra, inter


def molecular_energy_gradient(
    state: MolecularModeState,
    model: MolecularPeierlsModel,
    one_body_density_matrix: NDArray[np.floating | np.complexfloating],
) -> MolecularModeState:
    """Return Hellmann-Feynman derivatives with respect to ``u`` and ``q``.

    The density matrix may be real or complex. Because the static graph
    Hamiltonian is real, each bond force depends on ``2 Re(gamma_ij)``.
    """
    _validate_state(state, model)
    lattice = model.tight_binding.lattice
    gamma = np.asarray(one_body_density_matrix)
    if gamma.shape != (lattice.n_sites, lattice.n_sites):
        raise ValueError("one-body density matrix has the wrong shape")
    if not np.all(np.isfinite(gamma)):
        raise ValueError("one-body density matrix must be finite")

    density = np.real(np.diag(gamma)).astype(np.float64, copy=False)
    grad_u = (
        model.holstein_k_ev_per_angstrom2 * state.u
        + model.holstein_alpha_ev_per_angstrom * density
    )
    grad_q = np.zeros_like(state.q, dtype=np.float64)
    coupling = model.coupling_vectors
    stiffness = model.stiffness_matrices

    for bond in lattice.generated_bonds():
        i = bond.family_source
        j = bond.family_target
        delta_q = state.q[j] - state.q[i]
        elastic = stiffness[bond.family] @ delta_q
        electronic = 2.0 * float(np.real(gamma[i, j])) * coupling[bond.family]
        force = elastic + electronic
        grad_q[i] -= force
        grad_q[j] += force

    return MolecularModeState(
        u=np.asarray(grad_u, dtype=np.float64),
        q=np.asarray(grad_q, dtype=np.float64),
    )
