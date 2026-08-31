import numpy as np
from scipy.linalg import eigh

from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.energy import lattice_energy
from holstein_peierls.gradients import energy_gradient
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.molecular_hamiltonian import MolecularTightBindingModel
from holstein_peierls.molecular_lattice import (
    BondFamily,
    MolecularBasisSite,
    PeriodicMolecularLattice2D,
    rectangular_legacy_lattice,
)
from holstein_peierls.molecular_peierls import (
    MolecularModeState,
    MolecularPeierlsModel,
    build_molecular_peierls_hamiltonian,
    molecular_energy_gradient,
    molecular_lattice_energy,
)
from holstein_peierls.parameters import StaticPolaronParameters


def _legacy_model(parameters: StaticPolaronParameters) -> MolecularPeierlsModel:
    lattice = rectangular_legacy_lattice(parameters.nx, parameters.ny, 6.0, 8.0)
    tight_binding = MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(
            ("x", -parameters.j0x),
            ("y", -parameters.j0y),
        ),
    )
    return MolecularPeierlsModel(
        tight_binding=tight_binding,
        n_modes=2,
        holstein_alpha_ev_per_angstrom=parameters.alpha_intra,
        holstein_k_ev_per_angstrom2=parameters.k1,
        bond_mode_couplings=(
            ("x", (parameters.alpha_interx, 0.0)),
            ("y", (0.0, parameters.alpha_intery)),
        ),
        bond_stiffness_matrices=(
            ("x", ((parameters.k2, 0.0), (0.0, 0.0))),
            ("y", ((0.0, 0.0), (0.0, parameters.k2))),
        ),
    )


def _convert_legacy_state(state: LatticeState) -> MolecularModeState:
    return MolecularModeState(
        u=state.u.ravel(order="C").copy(),
        q=np.column_stack((state.vx.ravel(order="C"), state.vy.ravel(order="C"))),
    )


def test_generalized_rectangular_hamiltonian_matches_legacy_with_distortion() -> None:
    p = StaticPolaronParameters(
        nx=5,
        ny=4,
        j0x=0.100,
        j0y=0.015,
        alpha_interx=0.10,
        alpha_intery=0.12,
    )
    rng = np.random.default_rng(3001)
    shape = (p.ny, p.nx)
    legacy = LatticeState(
        u=rng.normal(scale=0.02, size=shape),
        vx=rng.normal(scale=0.03, size=shape),
        vy=rng.normal(scale=0.03, size=shape),
    )
    reference = build_dense_hamiltonian(legacy, p)
    candidate = build_molecular_peierls_hamiltonian(
        _convert_legacy_state(legacy), _legacy_model(p)
    ).toarray()
    assert np.array_equal(candidate, reference)


def test_generalized_rectangular_lattice_energy_matches_legacy() -> None:
    p = StaticPolaronParameters(nx=5, ny=4, k1=16.51, k2=0.51)
    rng = np.random.default_rng(3002)
    shape = (p.ny, p.nx)
    legacy = LatticeState(
        u=rng.normal(scale=0.04, size=shape),
        vx=rng.normal(scale=0.03, size=shape),
        vy=rng.normal(scale=0.03, size=shape),
    )
    reference = lattice_energy(legacy, p)
    candidate = molecular_lattice_energy(_convert_legacy_state(legacy), _legacy_model(p))
    assert np.allclose(candidate, reference, rtol=0.0, atol=2.0e-16)


def test_generalized_rectangular_gradient_matches_legacy_optimized_gradient() -> None:
    p = StaticPolaronParameters(
        nx=5,
        ny=4,
        j0x=0.100,
        j0y=0.015,
        alpha_interx=0.10,
        alpha_intery=0.12,
    )
    rng = np.random.default_rng(3003)
    shape = (p.ny, p.nx)
    legacy = LatticeState(
        u=rng.normal(scale=0.02, size=shape),
        vx=rng.normal(scale=0.02, size=shape),
        vy=rng.normal(scale=0.02, size=shape),
    )
    ground = solve_ground_state(legacy, p, solver="dense_lowest")
    reference, _ = energy_gradient(legacy, p, ground_state=ground, mode="optimized")
    gamma = np.outer(ground.wavefunction, ground.wavefunction)
    candidate = molecular_energy_gradient(
        _convert_legacy_state(legacy), _legacy_model(p), gamma
    )
    assert np.allclose(candidate.u.reshape(shape), reference.u, rtol=0.0, atol=2.0e-15)
    assert np.allclose(
        candidate.q[:, 0].reshape(shape), reference.vx, rtol=0.0, atol=2.0e-15
    )
    assert np.allclose(
        candidate.q[:, 1].reshape(shape), reference.vy, rtol=0.0, atol=2.0e-15
    )


def _skew_model_and_state() -> tuple[MolecularPeierlsModel, MolecularModeState]:
    lattice = PeriodicMolecularLattice2D(
        n1=3,
        n2=3,
        a1_angstrom=(5.1, 0.0),
        a2_angstrom=(1.1, 4.4),
        basis=(
            MolecularBasisSite("A", (0.0, 0.0)),
            MolecularBasisSite("B", (0.46, 0.53)),
        ),
        bonds=(
            BondFamily("AB0", 0, 1, (0, 0)),
            BondFamily("ABx", 0, 1, (-1, 0)),
            BondFamily("ABy", 0, 1, (0, -1)),
            BondFamily("AAx", 0, 0, (1, 0)),
        ),
    )
    tight_binding = MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(
            ("AB0", -0.081),
            ("ABx", -0.052),
            ("ABy", 0.031),
            ("AAx", -0.014),
        ),
        onsite_energies_ev=(0.01, -0.006),
    )
    model = MolecularPeierlsModel(
        tight_binding=tight_binding,
        n_modes=2,
        holstein_alpha_ev_per_angstrom=2.7,
        holstein_k_ev_per_angstrom2=15.0,
        bond_mode_couplings=(
            ("AB0", (0.08, -0.03)),
            ("ABx", (-0.04, 0.07)),
            ("ABy", (0.02, 0.05)),
            ("AAx", (0.06, 0.01)),
        ),
        bond_stiffness_matrices=(
            ("AB0", ((0.70, 0.10), (0.10, 0.45))),
            ("ABx", ((0.55, -0.08), (-0.08, 0.60))),
            ("ABy", ((0.40, 0.04), (0.04, 0.75))),
            ("AAx", ((0.62, 0.00), (0.00, 0.30))),
        ),
    )
    rng = np.random.default_rng(3004)
    state = MolecularModeState(
        u=rng.normal(scale=0.015, size=lattice.n_sites),
        q=rng.normal(scale=0.020, size=(lattice.n_sites, 2)),
    )
    state.u[0] -= 0.04
    return model, state


def _ground_energy_and_vector(
    state: MolecularModeState, model: MolecularPeierlsModel
) -> tuple[float, np.ndarray]:
    h = build_molecular_peierls_hamiltonian(state, model).toarray()
    values, vectors = eigh(h, subset_by_index=(0, 0), driver="evr")
    return float(values[0]), np.asarray(vectors[:, 0], dtype=np.float64)


def _total_energy(state: MolecularModeState, model: MolecularPeierlsModel) -> float:
    electronic, _ = _ground_energy_and_vector(state, model)
    intra, inter = molecular_lattice_energy(state, model)
    return electronic + intra + inter


def test_skew_two_basis_generalized_gradient_matches_finite_differences() -> None:
    model, state = _skew_model_and_state()
    _, psi = _ground_energy_and_vector(state, model)
    gamma = np.outer(psi, psi)
    analytical = molecular_energy_gradient(state, model, gamma)
    epsilon = 2.0e-6

    checks = [
        ("u", (0,), analytical.u[0]),
        ("q", (0, 0), analytical.q[0, 0]),
        ("q", (1, 1), analytical.q[1, 1]),
        ("q", (7, 0), analytical.q[7, 0]),
    ]
    for field, index, expected in checks:
        plus = state.copy()
        minus = state.copy()
        getattr(plus, field)[index] += epsilon
        getattr(minus, field)[index] -= epsilon
        numerical = (_total_energy(plus, model) - _total_energy(minus, model)) / (
            2.0 * epsilon
        )
        assert np.isclose(expected, numerical, rtol=5.0e-5, atol=5.0e-7)


def test_generalized_gradient_matches_correlated_bipolaron_rdm() -> None:
    from holstein_peierls.two_particle.parameters import BipolaronParameters
    from holstein_peierls.two_particle.peierls import (
        energy_gradient as bipolaron_energy_gradient,
        solve_holstein_peierls_ground_state,
    )

    p = BipolaronParameters(
        nx=4,
        ny=3,
        pair_position=6,
        j0x=0.100,
        j0y=0.015,
        alpha_interx=0.10,
        alpha_intery=0.12,
        hubbard_u=0.525,
        nearest_neighbor_v=0.008,
        eigensolver_tolerance=1.0e-12,
    )
    rng = np.random.default_rng(3005)
    shape = (p.ny, p.nx)
    legacy = LatticeState(
        u=rng.normal(scale=0.015, size=shape),
        vx=rng.normal(scale=0.012, size=shape),
        vy=rng.normal(scale=0.012, size=shape),
    )
    legacy.u[1, 1] -= 0.06

    ground = solve_holstein_peierls_ground_state(legacy, p)
    reference, _ = bipolaron_energy_gradient(legacy, p, ground_state=ground)
    candidate = molecular_energy_gradient(
        _convert_legacy_state(legacy), _legacy_model(p), ground.one_body_density_matrix
    )

    assert np.allclose(candidate.u.reshape(shape), reference.u, rtol=0.0, atol=3.0e-15)
    assert np.allclose(
        candidate.q[:, 0].reshape(shape), reference.vx, rtol=0.0, atol=3.0e-15
    )
    assert np.allclose(
        candidate.q[:, 1].reshape(shape), reference.vy, rtol=0.0, atol=3.0e-15
    )
