import numpy as np

from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.molecular_hamiltonian import (
    MolecularTightBindingModel,
    bloch_hamiltonian,
    build_molecular_hamiltonian,
    reciprocal_vectors_per_angstrom,
    sampled_bloch_spectrum,
)
from holstein_peierls.molecular_lattice import (
    BondFamily,
    MolecularBasisSite,
    PeriodicMolecularLattice2D,
    rectangular_legacy_lattice,
)
from holstein_peierls.parameters import StaticPolaronParameters


def test_legacy_rectangular_hamiltonian_is_bitwise_identical_without_distortion() -> None:
    parameters = StaticPolaronParameters(nx=5, ny=4, j0x=0.100, j0y=0.015)
    zero = np.zeros((parameters.ny, parameters.nx), dtype=np.float64)
    state = LatticeState(u=zero.copy(), vx=zero.copy(), vy=zero.copy())
    reference = build_dense_hamiltonian(state, parameters)

    lattice = rectangular_legacy_lattice(5, 4, 6.0, 8.0)
    model = MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(("x", -parameters.j0x), ("y", -parameters.j0y)),
    )
    candidate = build_molecular_hamiltonian(model).toarray()
    assert np.array_equal(candidate, reference)


def test_legacy_rectangular_bloch_dispersion_is_analytic() -> None:
    ax, ay = 6.2, 7.8
    jx, jy = 0.100, 0.015
    lattice = rectangular_legacy_lattice(5, 4, ax, ay)
    model = MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(("x", -jx), ("y", -jy)),
    )
    k = np.array((0.17, -0.09))
    numerical = float(np.real(bloch_hamiltonian(model, k)[0, 0]))
    analytical = -2.0 * jx * np.cos(k[0] * ax) - 2.0 * jy * np.cos(k[1] * ay)
    assert np.isclose(numerical, analytical, rtol=0.0, atol=2.0e-16)


def test_rectangular_supercell_spectrum_matches_sampled_bloch_dispersion() -> None:
    lattice = rectangular_legacy_lattice(5, 4, 6.0, 8.0)
    model = MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(("x", -0.100), ("y", -0.015)),
    )
    supercell = np.linalg.eigvalsh(build_molecular_hamiltonian(model).toarray())
    bloch = sampled_bloch_spectrum(model)
    assert np.allclose(np.sort(supercell), bloch, rtol=0.0, atol=5.0e-16)


def _two_basis_skew_model(reverse_bond_order: bool = False) -> MolecularTightBindingModel:
    bonds = (
        BondFamily("AB0", 0, 1, (0, 0)),
        BondFamily("ABx", 0, 1, (-1, 0)),
        BondFamily("ABy", 0, 1, (0, -1)),
        BondFamily("AAx", 0, 0, (1, 0)),
    )
    if reverse_bond_order:
        bonds = tuple(reversed(bonds))
    lattice = PeriodicMolecularLattice2D(
        n1=4,
        n2=3,
        a1_angstrom=(5.0, 0.0),
        a2_angstrom=(1.2, 4.3),
        basis=(
            MolecularBasisSite("A", (0.0, 0.0)),
            MolecularBasisSite("B", (0.48, 0.52)),
        ),
        bonds=bonds,
    )
    return MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(
            ("AB0", -0.081),
            ("ABx", -0.052),
            ("ABy", 0.031),
            ("AAx", -0.014),
        ),
        onsite_energies_ev=(0.012, -0.007),
    )


def test_two_basis_bloch_hamiltonian_is_hermitian_at_arbitrary_k() -> None:
    model = _two_basis_skew_model()
    h = bloch_hamiltonian(model, (0.23, -0.19))
    assert np.allclose(h, h.conj().T, rtol=0.0, atol=1.0e-15)


def test_hamiltonians_are_invariant_to_bond_family_ordering() -> None:
    first = _two_basis_skew_model(False)
    second = _two_basis_skew_model(True)
    h1 = build_molecular_hamiltonian(first).toarray()
    h2 = build_molecular_hamiltonian(second).toarray()
    assert np.array_equal(h1, h2)
    k = (0.13, 0.27)
    assert np.allclose(
        bloch_hamiltonian(first, k),
        bloch_hamiltonian(second, k),
        rtol=0.0,
        atol=1.0e-16,
    )


def test_two_basis_skew_supercell_spectrum_matches_sampled_bloch_bands() -> None:
    model = _two_basis_skew_model()
    supercell = np.sort(np.linalg.eigvalsh(build_molecular_hamiltonian(model).toarray()))
    bloch = sampled_bloch_spectrum(model)
    assert np.allclose(supercell, bloch, rtol=0.0, atol=2.0e-15)


def test_reciprocal_vectors_satisfy_duality_for_skew_lattice() -> None:
    model = _two_basis_skew_model()
    a = model.lattice.bravais_matrix
    b = reciprocal_vectors_per_angstrom(model.lattice)
    assert np.allclose(a.T @ b, 2.0 * np.pi * np.eye(2), rtol=0.0, atol=1.0e-15)
