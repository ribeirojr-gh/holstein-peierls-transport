from dataclasses import replace

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.bipolaron import (
    expectation_energy,
    solve_bipolaron_ground_state,
)
from holstein_peierls.two_particle.interaction import (
    interaction_expectation,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    energy_gradient,
    solve_holstein_peierls_ground_state,
    total_energy,
)


def _symmetric_pair(n: int, first: int, second: int) -> np.ndarray:
    psi = np.zeros((n, n), dtype=float)
    if first == second:
        psi[first, first] = 1.0
    else:
        amplitude = 1.0 / np.sqrt(2.0)
        psi[first, second] = amplitude
        psi[second, first] = amplitude
    return psi


def test_extended_hubbard_matrix_uses_periodic_nearest_neighbours() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.71,
        nearest_neighbor_v=0.13,
    )
    interaction = pair_interaction_matrix(p)

    assert np.isclose(interaction[0, 0], 0.71)
    assert np.isclose(interaction[0, 1], 0.13)
    assert np.isclose(interaction[0, 3], 0.13)  # periodic x boundary
    assert np.isclose(interaction[0, 4], 0.13)
    assert np.isclose(interaction[0, 12], 0.13)  # periodic y boundary
    assert np.isclose(interaction[0, 5], 0.0)


def test_interaction_expectation_distinguishes_onsite_nn_and_far_pairs() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.60,
        nearest_neighbor_v=0.08,
        j0x=0.0,
        j0y=0.0,
        alpha_intra=0.0,
    )
    n = p.n_sites
    onsite = _symmetric_pair(n, 5, 5)
    nearest = _symmetric_pair(n, 5, 6)
    separated = _symmetric_pair(n, 5, 15)

    assert np.isclose(interaction_expectation(onsite, p), p.hubbard_u)
    assert np.isclose(interaction_expectation(nearest, p), p.nearest_neighbor_v)
    assert np.isclose(interaction_expectation(separated, p), 0.0)

    u = np.zeros((p.ny, p.nx))
    assert np.isclose(expectation_energy(onsite, u, p).total, p.hubbard_u)
    assert np.isclose(expectation_energy(nearest, u, p).total, p.nearest_neighbor_v)
    assert np.isclose(expectation_energy(separated, u, p).total, 0.0)


def test_ground_state_hellmann_feynman_derivative_with_respect_to_v1() -> None:
    base = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.55,
        nearest_neighbor_v=0.07,
        j0x=0.10,
        j0y=0.04,
        eigensolver_tolerance=1.0e-12,
    )
    u = np.zeros((3, 3))
    u[1, 1] = -0.11
    u[1, 2] = -0.07

    ground = solve_bipolaron_ground_state(u, base)
    obs = pair_observables(ground, base)

    epsilon = 2.0e-6
    plus = solve_bipolaron_ground_state(
        u, replace(base, nearest_neighbor_v=base.nearest_neighbor_v + epsilon)
    )
    minus = solve_bipolaron_ground_state(
        u, replace(base, nearest_neighbor_v=base.nearest_neighbor_v - epsilon)
    )
    numerical = (plus.energy - minus.energy) / (2.0 * epsilon)

    assert np.isclose(
        numerical,
        obs.nearest_neighbour_probability,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_structural_gradient_remains_correct_with_finite_v1() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.55,
        nearest_neighbor_v=0.06,
        k2=0.73,
        j0x=0.10,
        j0y=0.04,
        alpha_interx=0.10,
        alpha_intery=0.07,
        eigensolver_tolerance=1.0e-12,
    )
    rng = np.random.default_rng(202)
    state = LatticeState(
        u=rng.normal(scale=0.02, size=(3, 3)),
        vx=rng.normal(scale=0.01, size=(3, 3)),
        vy=rng.normal(scale=0.01, size=(3, 3)),
    )
    state.u[1, 1] -= 0.10

    ground = solve_holstein_peierls_ground_state(state, p)
    gradient, _ = energy_gradient(state, p, ground_state=ground)
    epsilon = 2.0e-6

    for field, index, analytical in (
        ("u", (1, 1), gradient.u[1, 1]),
        ("vx", (1, 1), gradient.vx[1, 1]),
        ("vy", (1, 1), gradient.vy[1, 1]),
    ):
        plus = state.copy()
        minus = state.copy()
        getattr(plus, field)[index] += epsilon
        getattr(minus, field)[index] -= epsilon
        ep, _ = total_energy(plus, p)
        em, _ = total_energy(minus, p)
        numerical = (ep.total - em.total) / (2.0 * epsilon)
        assert np.isclose(analytical, numerical, rtol=4.0e-5, atol=4.0e-7)
