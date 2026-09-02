import numpy as np
from scipy.linalg import expm

from holstein_peierls.spin_adapted import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    open_shell_orbital_energy,
    optimize_open_shell_orbitals,
    orbital_rotation_gradient,
)


def _orthogonal_from_seed(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q @ np.diag(signs)


def test_orbital_rotation_gradient_matches_central_difference() -> None:
    n = 4
    orbitals = _orthogonal_from_seed(12, n)
    one_body = np.asarray(
        (
            (-0.30, -0.08, 0.00, -0.02),
            (-0.08, -0.10, -0.05, 0.00),
            (0.00, -0.05, 0.15, -0.06),
            (-0.02, 0.00, -0.06, 0.32),
        )
    )
    interaction = np.asarray(
        (
            (0.70, 0.20, 0.10, 0.20),
            (0.20, 0.70, 0.20, 0.10),
            (0.10, 0.20, 0.70, 0.20),
            (0.20, 0.10, 0.20, 0.70),
        )
    )
    shell_sizes = (1, 1, 1)
    gradient = orbital_rotation_gradient(
        one_body,
        interaction,
        orbitals,
        shell_sizes,
        OPEN_SHELL_SINGLET,
    )

    # Mix the doubly occupied orbital (occupation 2) with the virtual orbital
    # (occupation 0).  This is an allowed Miranda orbital rotation.
    direction = np.zeros((n, n))
    direction[0, 3] = 1.0
    direction[3, 0] = -1.0
    analytic = float(np.trace(gradient @ direction))

    step = 1.0e-6
    plus = orbitals @ expm(step * direction)
    minus = orbitals @ expm(-step * direction)
    e_plus = open_shell_orbital_energy(
        one_body, interaction, plus, shell_sizes, OPEN_SHELL_SINGLET
    )
    e_minus = open_shell_orbital_energy(
        one_body, interaction, minus, shell_sizes, OPEN_SHELL_SINGLET
    )
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    assert np.isclose(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-8)


def test_same_occupation_open_shell_rotation_is_masked() -> None:
    orbitals = _orthogonal_from_seed(3, 4)
    one_body = np.diag((-0.4, -0.1, 0.2, 0.5))
    interaction = 0.6 * np.eye(4)
    gradient = orbital_rotation_gradient(
        one_body,
        interaction,
        orbitals,
        (1, 1, 1),
        OPEN_SHELL_SINGLET,
    )
    # Columns 1 and 2 are the two distinct singly occupied shells.  Their
    # rotation is fixed in the minimal fixed-coefficient ansatz.
    assert gradient[1, 2] == 0.0
    assert gradient[2, 1] == 0.0


def test_closed_shell_orbital_optimizer_decreases_energy_and_converges() -> None:
    orbitals = _orthogonal_from_seed(21, 4)
    one_body = np.asarray(
        (
            (-0.35, -0.10, 0.00, -0.03),
            (-0.10, -0.12, -0.07, 0.00),
            (0.00, -0.07, 0.16, -0.09),
            (-0.03, 0.00, -0.09, 0.34),
        )
    )
    interaction = np.asarray(
        (
            (0.55, 0.15, 0.08, 0.15),
            (0.15, 0.55, 0.15, 0.08),
            (0.08, 0.15, 0.55, 0.15),
            (0.15, 0.08, 0.15, 0.55),
        )
    )
    shell_sizes = (2,)
    initial_energy = open_shell_orbital_energy(
        one_body, interaction, orbitals, shell_sizes, CLOSED_SHELL_SINGLET
    )
    result = optimize_open_shell_orbitals(
        one_body,
        interaction,
        orbitals,
        shell_sizes,
        CLOSED_SHELL_SINGLET,
        gradient_tolerance=1.0e-8,
        max_iterations=500,
    )
    assert result.energy < initial_energy
    assert result.diagnostics.converged
    assert result.diagnostics.final_max_gradient < 1.0e-8
    np.testing.assert_allclose(result.orbitals.T @ result.orbitals, np.eye(4), atol=1e-10)


def test_noninteracting_optimized_singlet_and_triplet_are_degenerate() -> None:
    one_body = np.diag((-0.8, -0.3, 0.2, 0.7))
    interaction = np.zeros((4, 4))
    orbitals = np.eye(4)

    singlet = optimize_open_shell_orbitals(
        one_body,
        interaction,
        orbitals,
        (1, 1, 1),
        OPEN_SHELL_SINGLET,
    )
    triplet = optimize_open_shell_orbitals(
        one_body,
        interaction,
        orbitals,
        (1, 2),
        HIGH_SPIN_TRIPLET,
    )
    assert singlet.diagnostics.converged
    assert triplet.diagnostics.converged
    assert np.isclose(singlet.energy, triplet.energy, atol=1.0e-12)
