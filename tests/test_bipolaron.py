from dataclasses import replace

import numpy as np

from holstein_peierls.two_particle.bipolaron import (
    atomic_limit_binding_energy,
    energy_gradient_u,
    expectation_energy,
    solve_bipolaron_ground_state,
    total_energy,
)
from holstein_peierls.two_particle.parameters import BipolaronParameters


def test_rigid_noninteracting_limit_is_two_single_particle_ground_energies() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        alpha_intra=0.0,
        hubbard_u=0.0,
        j0x=0.10,
        j0y=0.05,
    )
    u = np.zeros((3, 3))
    ground = solve_bipolaron_ground_state(u, p)
    expected_one_particle = -2.0 * p.j0x - 2.0 * p.j0y
    assert np.isclose(ground.energy, 2.0 * expected_one_particle, atol=2e-11)


def test_ground_state_remains_in_singlet_spatial_sector() -> None:
    p = BipolaronParameters(nx=3, ny=3, pair_position=5, hubbard_u=0.25)
    u = np.zeros((3, 3))
    u[1, 1] = -0.15
    ground = solve_bipolaron_ground_state(u, p)
    assert ground.exchange_symmetry_error < 2e-14


def test_one_body_density_matrix_has_two_particles() -> None:
    p = BipolaronParameters(nx=3, ny=3, pair_position=5, hubbard_u=0.2)
    u = np.zeros((3, 3))
    u[1, 1] = -0.12
    ground = solve_bipolaron_ground_state(u, p)
    gamma = ground.one_body_density_matrix
    assert np.isclose(np.trace(gamma), 2.0, atol=2e-12)
    assert np.isclose(np.sum(ground.site_density), 2.0, atol=2e-12)


def test_holstein_gradient_matches_finite_difference() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.18,
        j0x=0.10,
        j0y=0.04,
    )
    rng = np.random.default_rng(41)
    u = rng.normal(scale=0.02, size=(3, 3))
    u[1, 1] -= 0.10

    ground = solve_bipolaron_ground_state(u, p)
    gradient, _ = energy_gradient_u(u, p, ground_state=ground)

    epsilon = 2.0e-6
    index = (1, 1)
    plus = u.copy()
    minus = u.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    ep, _ = total_energy(plus, p)
    em, _ = total_energy(minus, p)
    numerical = (ep.total - em.total) / (2.0 * epsilon)
    assert np.isclose(gradient[index], numerical, rtol=2e-5, atol=2e-7)


def test_atomic_limit_recovers_exact_holstein_hubbard_binding_threshold() -> None:
    base = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        j0x=0.0,
        j0y=0.0,
        alpha_intra=3.0,
        k1=16.51,
    )
    n = base.n_sites
    i = base.pair_index
    j = 0 if i != 0 else n - 1

    for hubbard_u in (0.30, 0.70):
        p = replace(base, hubbard_u=hubbard_u)

        psi_onsite = np.zeros((n, n))
        psi_onsite[i, i] = 1.0
        u_onsite = np.zeros((p.ny, p.nx))
        iy, ix = divmod(i, p.nx)
        u_onsite[iy, ix] = -2.0 * p.alpha_intra / p.k1
        e_onsite = expectation_energy(psi_onsite, u_onsite, p).total

        psi_separated = np.zeros((n, n))
        psi_separated[i, j] = 1.0 / np.sqrt(2.0)
        psi_separated[j, i] = 1.0 / np.sqrt(2.0)
        u_separated = np.zeros((p.ny, p.nx))
        jy, jx = divmod(j, p.nx)
        u_separated[iy, ix] = -p.alpha_intra / p.k1
        u_separated[jy, jx] = -p.alpha_intra / p.k1
        e_separated = expectation_energy(psi_separated, u_separated, p).total

        numerical_binding = e_separated - e_onsite
        analytic_binding = atomic_limit_binding_energy(p)
        assert np.isclose(numerical_binding, analytic_binding, atol=2e-13)
        assert np.isclose(
            analytic_binding,
            p.alpha_intra**2 / p.k1 - p.hubbard_u,
            atol=1e-15,
        )


def test_atomic_limit_critical_u_matches_current_parameter_scale() -> None:
    p = BipolaronParameters()
    assert np.isclose(p.atomic_holstein_pairing_scale, 9.0 / 16.51, atol=1e-15)
    assert 0.545 < p.atomic_holstein_pairing_scale < 0.546
