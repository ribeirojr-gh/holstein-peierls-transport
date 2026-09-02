import numpy as np

from holstein_peierls.electronic import GroundState
from holstein_peierls.exciton.solver import ExcitonGroundState
from holstein_peierls.projection_observables import (
    bipolaron_one_particle_rdm,
    bipolaron_pair_state_vector,
    configuration_expansion_norm,
    exciton_one_particle_rdms,
    exciton_pair_state_vector,
    instantaneous_occupations,
    polaron_configuration,
    polaron_one_particle_rdm,
    product_basis_channel_yield,
    rectangular_pair_channel,
    slater_yield,
    spin_adapted_configuration_expansion,
    spin_adapted_rdms,
)
from holstein_peierls.spin_adapted.excitation_reference import ReferencedExcitationState
from holstein_peierls.spin_adapted.open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
)
from holstein_peierls.spin_adapted.orbital_optimization import (
    OpenShellOrbitalResult,
    OrbitalOptimizationDiagnostics,
    shell_projectors_from_complete_orbitals,
)
from holstein_peierls.spin_adapted.spin import SpinMultiplicity
from holstein_peierls.two_particle.bipolaron import BipolaronGroundState


def _diagnostics() -> OrbitalOptimizationDiagnostics:
    return OrbitalOptimizationDiagnostics(
        iterations=0,
        converged=True,
        final_energy=0.0,
        final_max_gradient=0.0,
        accepted_steps=0,
        rejected_steps=0,
    )


def _referenced_s0_state(multiplicity: SpinMultiplicity) -> ReferencedExcitationState:
    orbitals = np.eye(4, dtype=np.float64)
    neutral_shell_sizes = (2,)
    neutral = OpenShellOrbitalResult(
        orbitals=orbitals,
        projectors=shell_projectors_from_complete_orbitals(
            orbitals,
            neutral_shell_sizes,
            CLOSED_SHELL_SINGLET,
        ),
        energy=0.0,
        diagnostics=_diagnostics(),
    )

    if multiplicity is SpinMultiplicity.SINGLET:
        definition = OPEN_SHELL_SINGLET
        excited_shell_sizes = (1, 1, 1)
    else:
        definition = HIGH_SPIN_TRIPLET
        excited_shell_sizes = (1, 2)
    excited = OpenShellOrbitalResult(
        orbitals=orbitals,
        projectors=shell_projectors_from_complete_orbitals(
            orbitals,
            excited_shell_sizes,
            definition,
        ),
        energy=0.0,
        diagnostics=_diagnostics(),
    )
    return ReferencedExcitationState(
        neutral=neutral,
        excited=excited,
        multiplicity=multiplicity,
        neutral_shell_sizes=neutral_shell_sizes,
        excited_shell_sizes=excited_shell_sizes,
    )


def test_polaron_adapter_preserves_rank_one_coherence() -> None:
    wavefunction = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)
    state = GroundState(energy=-1.0, wavefunction=wavefunction)

    gamma = polaron_one_particle_rdm(state)
    expected = np.outer(wavefunction, wavefunction)
    np.testing.assert_allclose(gamma, expected, rtol=0.0, atol=1.0e-15)
    assert abs(np.trace(gamma).real - 1.0) < 1.0e-15
    assert abs(gamma[0, 1]) > 0.49

    configuration = polaron_configuration(state)
    assert abs(slater_yield(configuration, configuration) - 1.0) < 1.0e-14


def test_bipolaron_adapter_matches_spin_summed_static_rdm() -> None:
    wavefunction = np.zeros((2, 2), dtype=np.float64)
    wavefunction[0, 1] = 1.0 / np.sqrt(2.0)
    wavefunction[1, 0] = 1.0 / np.sqrt(2.0)
    state = BipolaronGroundState(energy=-1.0, wavefunction=wavefunction)

    gamma = bipolaron_one_particle_rdm(state)
    np.testing.assert_allclose(
        gamma,
        state.one_body_density_matrix,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert abs(np.trace(gamma).real - 2.0) < 1.0e-14
    assert abs(np.linalg.norm(bipolaron_pair_state_vector(state)) - 1.0) < 1.0e-14

    onsite = rectangular_pair_channel(2, 1, "onsite")
    nearest_x = rectangular_pair_channel(2, 1, "nearest_x")
    assert product_basis_channel_yield(onsite, wavefunction) < 1.0e-15
    assert abs(product_basis_channel_yield(nearest_x, wavefunction) - 1.0) < 1.0e-14


def test_exciton_adapter_preserves_separate_carrier_rdms_and_frenkel_channel() -> None:
    wavefunction = np.array(
        [[np.sqrt(0.6), 0.0], [0.0, np.sqrt(0.4)]],
        dtype=np.float64,
    )
    state = ExcitonGroundState(energy=-1.0, wavefunction=wavefunction)

    rdms = exciton_one_particle_rdms(state)
    np.testing.assert_allclose(
        rdms.electron,
        state.electron_density_matrix,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        rdms.hole,
        state.hole_density_matrix,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert abs(np.trace(rdms.electron).real - 1.0) < 1.0e-14
    assert abs(np.trace(rdms.hole).real - 1.0) < 1.0e-14
    assert abs(np.linalg.norm(exciton_pair_state_vector(state)) - 1.0) < 1.0e-14

    frenkel = rectangular_pair_channel(2, 1, "onsite", name="frenkel")
    assert abs(product_basis_channel_yield(frenkel, wavefunction) - 1.0) < 1.0e-14
    assert abs(product_basis_channel_yield(frenkel, wavefunction) - state.onsite_probability) < 1.0e-14


def test_rectangular_pair_channels_have_expected_periodic_ranks() -> None:
    assert rectangular_pair_channel(3, 3, "onsite").rank == 9
    assert rectangular_pair_channel(3, 3, "nearest_x").rank == 18
    assert rectangular_pair_channel(3, 3, "nearest_y").rank == 18
    assert rectangular_pair_channel(3, 3, "diagonal").rank == 36


def test_s0_adapter_preserves_particle_number_and_zero_trace_excitation() -> None:
    state = _referenced_s0_state(SpinMultiplicity.SINGLET)
    rdms = spin_adapted_rdms(state)

    assert abs(np.trace(rdms.neutral).real - 4.0) < 1.0e-14
    assert abs(np.trace(rdms.excited).real - 4.0) < 1.0e-14
    assert abs(np.trace(rdms.excitation).real) < 1.0e-14
    np.testing.assert_allclose(
        rdms.excitation,
        rdms.excited - rdms.neutral,
        rtol=0.0,
        atol=1.0e-15,
    )

    occupations = instantaneous_occupations(
        rdms.excited,
        np.eye(4, dtype=np.complex128),
    )
    assert abs(np.sum(occupations) - 4.0) < 1.0e-14


def test_s0_configuration_adapter_matches_established_spin_coefficients() -> None:
    singlet = _referenced_s0_state(SpinMultiplicity.SINGLET)
    triplet = _referenced_s0_state(SpinMultiplicity.TRIPLET)

    neutral = spin_adapted_configuration_expansion(singlet, component="neutral")
    singlet_expansion = spin_adapted_configuration_expansion(singlet)
    triplet_expansion = spin_adapted_configuration_expansion(triplet)

    assert len(neutral.configurations) == 1
    assert abs(configuration_expansion_norm(neutral.configurations, neutral.coefficients) - 1.0) < 1.0e-14
    assert abs(
        configuration_expansion_norm(
            singlet_expansion.configurations,
            singlet_expansion.coefficients,
        )
        - 1.0
    ) < 1.0e-14
    assert abs(
        configuration_expansion_norm(
            triplet_expansion.configurations,
            triplet_expansion.coefficients,
        )
        - 1.0
    ) < 1.0e-14

    np.testing.assert_allclose(
        singlet_expansion.configurations[0],
        triplet_expansion.configurations[0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        singlet_expansion.configurations[1],
        triplet_expansion.configurations[1],
        rtol=0.0,
        atol=0.0,
    )
    assert abs(np.vdot(singlet_expansion.coefficients, triplet_expansion.coefficients)) < 1.0e-14
