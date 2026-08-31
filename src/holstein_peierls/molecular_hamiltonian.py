"""Graph-based one-particle tight-binding Hamiltonians for molecular lattices.

Phase G2 remains independent of the validated Holstein-Peierls production
solvers. The module consumes only the geometry introduced in G1 and fixed
(one-particle, undistorted) transfer integrals. Peierls coordinates and forces
are intentionally deferred to G3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .molecular_lattice import BondFamily, PeriodicMolecularLattice2D

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class MolecularTightBindingModel:
    """Fixed one-particle parameters attached to a periodic molecular graph.

    ``bond_transfer_integrals_ev`` stores the actual Hamiltonian matrix element
    for each bond family, including its sign. Therefore the legacy convention
    with positive ``Jx``/``Jy`` is represented by transfer integrals ``-Jx`` and
    ``-Jy``.
    """

    lattice: PeriodicMolecularLattice2D
    bond_transfer_integrals_ev: tuple[tuple[str, float], ...]
    onsite_energies_ev: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        labels = [label for label, _ in self.bond_transfer_integrals_ev]
        if len(set(labels)) != len(labels):
            raise ValueError("bond transfer-integral labels must be unique")
        supplied = set(labels)
        expected = {family.label for family in self.lattice.bonds}
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                "bond transfer-integral labels must match lattice bond families: "
                + ", ".join(details)
            )
        if not all(np.isfinite(value) for _, value in self.bond_transfer_integrals_ev):
            raise ValueError("bond transfer integrals must be finite")

        if self.onsite_energies_ev:
            if len(self.onsite_energies_ev) != self.lattice.n_basis:
                raise ValueError("onsite_energies_ev must have one value per basis site")
            if not all(np.isfinite(value) for value in self.onsite_energies_ev):
                raise ValueError("onsite energies must be finite")

    @property
    def transfer_integrals(self) -> dict[str, float]:
        return dict(self.bond_transfer_integrals_ev)

    @property
    def onsite_energies(self) -> FloatArray:
        if self.onsite_energies_ev:
            return np.asarray(self.onsite_energies_ev, dtype=np.float64)
        return np.zeros(self.lattice.n_basis, dtype=np.float64)


def build_molecular_hamiltonian(model: MolecularTightBindingModel) -> csr_matrix:
    """Build the real symmetric finite-supercell one-particle Hamiltonian."""
    lattice = model.lattice
    transfer = model.transfer_integrals
    onsite_basis = model.onsite_energies

    diagonal = np.empty(lattice.n_sites, dtype=np.float64)
    for site in range(lattice.n_sites):
        _, _, basis = lattice.site_coordinates(site)
        diagonal[site] = onsite_basis[basis]

    bonds = lattice.generated_bonds()
    if not bonds:
        return csr_matrix(np.diag(diagonal))

    source = np.fromiter((bond.source for bond in bonds), dtype=np.int64)
    target = np.fromiter((bond.target for bond in bonds), dtype=np.int64)
    hopping = np.fromiter(
        (transfer[bond.family] for bond in bonds), dtype=np.float64
    )
    sites = np.arange(lattice.n_sites, dtype=np.int64)
    rows = np.concatenate((sites, source, target))
    cols = np.concatenate((sites, target, source))
    data = np.concatenate((diagonal, hopping, hopping))
    return coo_matrix(
        (data, (rows, cols)), shape=(lattice.n_sites, lattice.n_sites)
    ).tocsr()


def bond_displacement_angstrom(
    lattice: PeriodicMolecularLattice2D,
    family: BondFamily,
) -> FloatArray:
    """Return the primitive-lattice displacement represented by one bond family."""
    source_fractional = np.asarray(
        lattice.basis[family.source_basis].position_fractional, dtype=np.float64
    )
    target_fractional = np.asarray(
        lattice.basis[family.target_basis].position_fractional, dtype=np.float64
    )
    delta_fractional = (
        target_fractional
        - source_fractional
        + np.asarray(family.cell_offset, dtype=np.float64)
    )
    return lattice.bravais_matrix @ delta_fractional


def bloch_hamiltonian(
    model: MolecularTightBindingModel,
    k_cartesian_per_angstrom: tuple[float, float] | FloatArray,
) -> ComplexArray:
    """Return the primitive-cell Bloch Hamiltonian at Cartesian wavevector ``k``."""
    lattice = model.lattice
    k = np.asarray(k_cartesian_per_angstrom, dtype=np.float64)
    if k.shape != (2,) or not np.all(np.isfinite(k)):
        raise ValueError("k must contain two finite Cartesian components")

    h = np.zeros((lattice.n_basis, lattice.n_basis), dtype=np.complex128)
    np.fill_diagonal(h, model.onsite_energies)
    transfer = model.transfer_integrals

    for family in lattice.bonds:
        displacement = bond_displacement_angstrom(lattice, family)
        phase = np.exp(1.0j * float(np.dot(k, displacement)))
        value = transfer[family.label] * phase
        source = family.source_basis
        target = family.target_basis
        h[source, target] += value
        h[target, source] += np.conjugate(value)
    return h


def reciprocal_vectors_per_angstrom(lattice: PeriodicMolecularLattice2D) -> FloatArray:
    """Return reciprocal primitive vectors ``[b1 b2]`` as columns."""
    return 2.0 * np.pi * np.linalg.inv(lattice.bravais_matrix).T


def sampled_supercell_kpoints(lattice: PeriodicMolecularLattice2D) -> FloatArray:
    """Return primitive-cell k points commensurate with the finite supercell."""
    reciprocal = reciprocal_vectors_per_angstrom(lattice)
    points = []
    for m2 in range(lattice.n2):
        for m1 in range(lattice.n1):
            points.append(
                (m1 / lattice.n1) * reciprocal[:, 0]
                + (m2 / lattice.n2) * reciprocal[:, 1]
            )
    return np.asarray(points, dtype=np.float64)


def sampled_bloch_spectrum(model: MolecularTightBindingModel) -> FloatArray:
    """Return all Bloch eigenvalues on the finite-supercell k-point mesh."""
    eigenvalues = []
    for k in sampled_supercell_kpoints(model.lattice):
        eigenvalues.extend(np.linalg.eigvalsh(bloch_hamiltonian(model, k)))
    return np.sort(np.asarray(eigenvalues, dtype=np.float64))
