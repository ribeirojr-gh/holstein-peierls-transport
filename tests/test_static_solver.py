from dataclasses import replace

import numpy as np

from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.energy import total_energy
from holstein_peierls.gradients import energy_gradient
from holstein_peierls.hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def test_hamiltonian_periodic_neighbours_and_signs() -> None:
    p = StaticPolaronParameters(nx=3, ny=3, polaron_position=1)
    s = LatticeState.zeros(3, 3)
    s.u[0, 1] = -0.2
    s.vx[0] = np.array([0.0, 0.1, 0.3])
    s.vy[:, 0] = np.array([0.0, 0.2, 0.0])
    h = build_dense_hamiltonian(s, p)
    assert np.allclose(h, h.T)
    assert np.isclose(h[1, 1], p.alpha_intra * -0.2)
    assert np.isclose(h[0, 1], -p.j0x + p.alpha_interx * 0.1)
    assert np.isclose(h[2, 0], -p.j0x + p.alpha_interx * (0.0 - 0.3))
    assert np.isclose(h[0, 3], -p.j0y + p.alpha_intery * 0.2)


def test_direct_sparse_hamiltonian_matches_dense_reference() -> None:
    p = replace(StaticPolaronParameters(), nx=4, ny=5, polaron_position=7)
    rng = np.random.default_rng(12)
    s = LatticeState(
        u=rng.normal(scale=0.03, size=(5, 4)),
        vx=rng.normal(scale=0.03, size=(5, 4)),
        vy=rng.normal(scale=0.03, size=(5, 4)),
    )
    dense = build_dense_hamiltonian(s, p)
    sparse = build_sparse_hamiltonian(s, p).toarray()
    assert np.array_equal(dense, sparse)


def test_analytical_gradient_matches_finite_difference() -> None:
    p = replace(StaticPolaronParameters(), nx=3, ny=3, polaron_position=5, j0y=0.07)
    rng = np.random.default_rng(7)
    s = LatticeState(
        u=rng.normal(scale=0.02, size=(3, 3)),
        vx=rng.normal(scale=0.02, size=(3, 3)),
        vy=rng.normal(scale=0.02, size=(3, 3)),
    )
    gradient, _ = energy_gradient(s, p, solver="dense_full")
    epsilon = 1e-6
    for name, index in (("u", (1, 1)), ("vx", (0, 2)), ("vy", (2, 0))):
        plus, minus = s.copy(), s.copy()
        getattr(plus, name)[index] += epsilon
        getattr(minus, name)[index] -= epsilon
        ep, _ = total_energy(plus, p, solver="dense_full")
        em, _ = total_energy(minus, p, solver="dense_full")
        numerical = (ep.total - em.total) / (2 * epsilon)
        assert np.isclose(getattr(gradient, name)[index], numerical, rtol=3e-6, atol=3e-8)


def test_electronic_solvers_agree() -> None:
    p = replace(StaticPolaronParameters(), nx=4, ny=4, polaron_position=6)
    s = LatticeState.zeros(4, 4)
    s.u[1, 1] = -0.1
    full = solve_ground_state(s, p, solver="dense_full")
    low = solve_ground_state(s, p, solver="dense_lowest")
    sparse = solve_ground_state(s, p, solver="sparse")
    assert np.isclose(full.energy, low.energy, atol=1e-12)
    assert np.isclose(full.energy, sparse.energy, atol=1e-10)
    assert np.allclose(full.charge_density, low.charge_density, atol=1e-10)
    assert np.allclose(full.charge_density, sparse.charge_density, atol=1e-9)


def test_converged_legacy_reference() -> None:
    p = replace(
        StaticPolaronParameters(),
        nx=4,
        ny=4,
        polaron_position=6,
        max_iterations=2000,
        convergence_criterion=1e-8,
    )
    result = solve_static_polaron(p, solver="dense_full", legacy_convergence=True)
    assert np.isclose(result.formation_energy, 0.5864295596422433, atol=2e-13)
    assert np.isclose(np.max(result.charge_density), 0.49634800043158167, atol=3e-8)
