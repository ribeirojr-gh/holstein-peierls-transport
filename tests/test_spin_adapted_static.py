import math

import numpy as np

from holstein_peierls.spin_adapted import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    IsotropicControlParameters,
    SpinMultiplicity,
    general_open_shell_energy,
    minimal_open_shell_coefficients,
    s2_eigenvalue,
    shell_fock_matrices,
)
from holstein_peierls.spin_adapted.open_shell import projector_from_orbitals


def test_requested_isotropic_control_maps_to_legacy_parameter_names() -> None:
    control = IsotropicControlParameters()
    assert control.is_isotropic
    assert control.j1 == control.j2 == 0.100
    assert control.alpha1 == control.alpha2 == 3.0
    # K1/K2 are deliberately not collapsed: the user only requested J/alpha
    # simplification for the coding control.
    assert control.k1 == 16.51
    assert control.k2 == 0.51

    polaron = control.to_polaron_parameters(nx=4, ny=4)
    assert polaron.j0x == polaron.j0y == 0.100
    assert polaron.alpha_intra == 3.0
    assert polaron.alpha_interx == polaron.alpha_intery == 3.0

    exciton = control.to_exciton_parameters(nx=4, ny=4)
    assert exciton.electron_j0x == exciton.electron_j0y == 0.100
    assert exciton.hole_j0x == exciton.hole_j0y == 0.100
    assert exciton.electron_alpha_intra == 3.0
    assert exciton.electron_alpha_interx == exciton.electron_alpha_intery == 3.0
    assert exciton.hole_alpha_intra == 3.0
    assert exciton.hole_alpha_interx == exciton.hole_alpha_intery == 3.0


def test_minimal_two_determinant_spin_coefficients_are_normalized_and_orthogonal() -> None:
    singlet = minimal_open_shell_coefficients(SpinMultiplicity.SINGLET)
    triplet = minimal_open_shell_coefficients(SpinMultiplicity.TRIPLET)
    assert np.isclose(np.linalg.norm(singlet), 1.0)
    assert np.isclose(np.linalg.norm(triplet), 1.0)
    assert np.isclose(np.dot(singlet, triplet), 0.0)
    assert s2_eigenvalue(SpinMultiplicity.SINGLET) == 0.0
    assert s2_eigenvalue(SpinMultiplicity.TRIPLET) == 2.0


def test_miranda_state_parameter_matrices_are_encoded_explicitly() -> None:
    assert OPEN_SHELL_SINGLET.occupations == (2, 1, 1)
    np.testing.assert_allclose(OPEN_SHELL_SINGLET.a_matrix, np.ones((3, 3)))
    np.testing.assert_allclose(
        OPEN_SHELL_SINGLET.b_matrix,
        np.asarray(((1, 1, 1), (1, 2, -2), (1, -2, 2)), dtype=float),
    )
    assert OPEN_SHELL_SINGLET.s2 == 0.0

    assert HIGH_SPIN_TRIPLET.occupations == (2, 1)
    np.testing.assert_allclose(HIGH_SPIN_TRIPLET.a_matrix, np.ones((2, 2)))
    np.testing.assert_allclose(
        HIGH_SPIN_TRIPLET.b_matrix,
        np.asarray(((1, 1), (1, 2)), dtype=float),
    )
    assert HIGH_SPIN_TRIPLET.s2 == 2.0


def test_noninteracting_open_shell_singlet_triplet_are_degenerate() -> None:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    valence = np.asarray((inv_sqrt2, inv_sqrt2))[:, None]
    conduction = np.asarray((inv_sqrt2, -inv_sqrt2))[:, None]
    p_valence = projector_from_orbitals(valence)
    p_conduction = projector_from_orbitals(conduction)
    p_empty = np.zeros((2, 2))
    p_open = p_valence + p_conduction
    one_body = np.diag((-0.3, 0.2))
    interaction = np.zeros((2, 2))

    singlet = general_open_shell_energy(
        one_body,
        interaction,
        (p_empty, p_valence, p_conduction),
        OPEN_SHELL_SINGLET,
    )
    triplet = general_open_shell_energy(
        one_body,
        interaction,
        (p_empty, p_open),
        HIGH_SPIN_TRIPLET,
    )
    assert np.isclose(singlet, triplet)


def test_hubbard_dimer_open_shell_exchange_splits_singlet_and_triplet() -> None:
    """Exact algebra gate: positive onsite U raises the singlet by U here."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    valence = np.asarray((inv_sqrt2, inv_sqrt2))[:, None]
    conduction = np.asarray((inv_sqrt2, -inv_sqrt2))[:, None]
    p_valence = projector_from_orbitals(valence)
    p_conduction = projector_from_orbitals(conduction)
    p_empty = np.zeros((2, 2))
    p_open = p_valence + p_conduction
    one_body = np.zeros((2, 2))
    hubbard_u = 0.7
    interaction = hubbard_u * np.eye(2)

    singlet = general_open_shell_energy(
        one_body,
        interaction,
        (p_empty, p_valence, p_conduction),
        OPEN_SHELL_SINGLET,
    )
    triplet = general_open_shell_energy(
        one_body,
        interaction,
        (p_empty, p_open),
        HIGH_SPIN_TRIPLET,
    )
    assert np.isclose(singlet - triplet, hubbard_u, atol=1.0e-12)
    assert singlet > triplet


def test_closed_shell_fock_reduces_to_two_j_minus_k() -> None:
    orbital = np.asarray((0.8, 0.6))[:, None]
    projector = projector_from_orbitals(orbital)
    one_body = np.asarray(((0.2, -0.1), (-0.1, -0.3)))
    interaction = np.asarray(((1.0, 0.4), (0.4, 0.8)))

    (fock,) = shell_fock_matrices(
        one_body,
        interaction,
        (projector,),
        CLOSED_SHELL_SINGLET,
    )
    density = np.diag(projector)
    expected = one_body + 2.0 * np.diag(interaction @ density) - interaction * projector
    np.testing.assert_allclose(fock, expected, atol=1.0e-12)
