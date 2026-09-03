"""Validation tests for D5a electronic thermalization diagnostics."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.electronic_thermalization import (
    canonical_occupations,
    diagnose_electronic_thermalization,
    distribution_energy_eV,
    effective_inverse_temperature_eV_inv,
    generalized_boltzmann_occupations,
    jensen_shannon_distance,
    occupation_entropy,
    occupation_participation_number,
    total_variation_distance,
)
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control() -> tuple[LatticeState, StaticPolaronParameters]:
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=6)
    rng = np.random.default_rng(20260903)
    state = LatticeState(
        0.01 * rng.normal(size=(4, 4)),
        0.02 * rng.normal(size=(4, 4)),
        0.02 * rng.normal(size=(4, 4)),
    )
    return state, parameters


def test_generalized_boltzmann_weights_are_shift_invariant_and_normalized():
    energies = np.array([-0.3, -0.1, 0.2, 0.8])
    beta = 7.25
    first = generalized_boltzmann_occupations(energies, beta)
    second = generalized_boltzmann_occupations(energies + 13.7, beta)
    assert np.allclose(first, second, rtol=0.0, atol=2.0e-15)
    assert np.isclose(np.sum(first), 1.0, rtol=0.0, atol=2.0e-16)
    assert np.all(first > 0.0)


def test_effective_beta_recovers_positive_and_negative_canonical_controls():
    energies = np.array([-0.35, -0.12, 0.04, 0.31, 0.62])
    for beta in (18.0, 3.5, -2.25, -11.0):
        population = generalized_boltzmann_occupations(energies, beta)
        target = distribution_energy_eV(energies, population)
        recovered = effective_inverse_temperature_eV_inv(energies, target)
        assert np.isclose(recovered, beta, rtol=2.0e-11, atol=2.0e-11)


def test_uniform_distribution_is_exact_infinite_temperature_control():
    energies = np.array([-0.5, -0.2, 0.1, 0.4, 0.9])
    uniform = np.full(energies.size, 1.0 / energies.size)
    target = distribution_energy_eV(energies, uniform)
    assert effective_inverse_temperature_eV_inv(energies, target) == 0.0
    assert occupation_participation_number(uniform) == float(energies.size)
    assert np.isclose(occupation_entropy(uniform), np.log(energies.size))


def test_probability_distances_have_expected_endpoints():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.7])
    assert total_variation_distance(p, p) == 0.0
    assert jensen_shannon_distance(p, p) == 0.0
    assert total_variation_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert total_variation_distance(p, q) > 0.0
    assert jensen_shannon_distance(p, q) > 0.0


def test_snapshot_recovers_constructed_canonical_adiabatic_populations():
    state, parameters = _control()
    temperature = 300.0
    hamiltonian = build_dense_hamiltonian(state, parameters)
    energies, vectors = np.linalg.eigh(hamiltonian)
    canonical = canonical_occupations(energies, temperature)
    phases = np.exp(1.0j * np.linspace(0.0, 1.3, energies.size))
    psi = vectors @ (np.sqrt(canonical) * phases)

    snapshot = diagnose_electronic_thermalization(
        state,
        parameters,
        psi,
        temperature,
    )

    assert np.allclose(snapshot.adiabatic_occupations, canonical, rtol=0.0, atol=2.0e-14)
    assert abs(snapshot.heating_coordinate) < 2.0e-13
    assert snapshot.tv_to_canonical < 2.0e-14
    assert snapshot.js_to_canonical < 2.0e-8
    assert np.isclose(snapshot.beta_eff_over_beta_bath, 1.0, rtol=2.0e-10, atol=2.0e-10)
    assert np.isclose(snapshot.effective_temperature_K, temperature, rtol=2.0e-10, atol=2.0e-8)


def test_snapshot_uniform_adiabatic_populations_give_heating_coordinate_one():
    state, parameters = _control()
    hamiltonian = build_dense_hamiltonian(state, parameters)
    energies, vectors = np.linalg.eigh(hamiltonian)
    psi = vectors @ np.full(energies.size, 1.0 / np.sqrt(energies.size), dtype=np.complex128)

    snapshot = diagnose_electronic_thermalization(state, parameters, psi, 300.0)

    assert np.isclose(snapshot.heating_coordinate, 1.0, rtol=0.0, atol=2.0e-13)
    assert snapshot.tv_to_uniform < 2.0e-14
    assert snapshot.beta_eff_eV_inv == 0.0
    assert np.isposinf(snapshot.effective_temperature_K)
    assert np.isclose(snapshot.normalized_occupation_entropy, 1.0, rtol=0.0, atol=2.0e-14)
    assert np.isclose(
        snapshot.occupation_participation_number,
        float(parameters.n_sites),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_snapshot_energy_matches_direct_expectation_and_is_global_phase_invariant():
    state, parameters = _control()
    rng = np.random.default_rng(77)
    psi = rng.normal(size=parameters.n_sites) + 1.0j * rng.normal(size=parameters.n_sites)
    psi = psi / np.linalg.norm(psi)
    phase = np.exp(1.0j * 0.731)

    first = diagnose_electronic_thermalization(state, parameters, psi, 300.0)
    second = diagnose_electronic_thermalization(state, parameters, phase * psi, 300.0)
    hamiltonian = build_dense_hamiltonian(state, parameters)
    direct = float(np.vdot(psi, hamiltonian @ psi).real)

    assert np.isclose(first.propagated_energy_eV, direct, rtol=0.0, atol=2.0e-14)
    assert np.allclose(
        first.adiabatic_occupations,
        second.adiabatic_occupations,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.isclose(first.heating_coordinate, second.heating_coordinate, rtol=0.0, atol=2.0e-14)
