import numpy as np
import pytest

from holstein_peierls.molecular_lattice import (
    MolecularBasisSite,
    PeriodicMolecularLattice2D,
)
from holstein_peierls.molecular_pair import (
    MolecularPairInteractionModel,
    PairContactFamily,
    contact_family_mask,
    legacy_rectangular_observable_contacts,
    legacy_rectangular_pair_model,
    molecular_interaction_expectation,
    molecular_pair_interaction_matrix,
    molecular_pair_observables,
)
from holstein_peierls.two_particle.bipolaron import BipolaronGroundState
from holstein_peierls.two_particle.interaction import (
    interaction_expectation,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _normalized_symmetric_wavefunction(n: int, seed: int = 4001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(n, n))
    psi = psi + psi.T
    psi /= np.linalg.norm(psi)
    return psi


@pytest.mark.parametrize(
    "parameters",
    [
        BipolaronParameters(
            nx=6,
            ny=6,
            pair_position=8,
            hubbard_u=0.70,
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=6.0,
            lattice_spacing_y_angstrom=8.0,
            relative_permittivity=4.0,
        ),
        BipolaronParameters(
            nx=6,
            ny=6,
            pair_position=8,
            hubbard_u=0.70,
            nearest_neighbor_v=0.11,
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=6.0,
            lattice_spacing_y_angstrom=8.0,
            relative_permittivity=4.0,
            short_range_shell_overrides=((1, 1, 0.07),),
        ),
        BipolaronParameters(
            nx=6,
            ny=6,
            pair_position=8,
            hubbard_u=0.70,
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=6.0,
            lattice_spacing_y_angstrom=8.0,
            relative_permittivity=4.0,
            short_range_shell_overrides=(
                (1, 0, 0.21),
                (0, 1, 0.17),
                (1, 1, 0.09),
                (2, 0, 0.05),
            ),
        ),
        BipolaronParameters(
            nx=6,
            ny=6,
            pair_position=8,
            hubbard_u=0.70,
            nearest_neighbor_v=0.11,
        ),
    ],
)
def test_legacy_adapter_reproduces_v0_6_interaction_matrix_bitwise(
    parameters: BipolaronParameters,
) -> None:
    reference = pair_interaction_matrix(parameters)
    candidate = molecular_pair_interaction_matrix(
        legacy_rectangular_pair_model(parameters)
    )
    assert np.array_equal(candidate, reference)


def test_legacy_adapter_reproduces_interaction_expectation() -> None:
    p = BipolaronParameters(
        nx=6,
        ny=6,
        pair_position=8,
        hubbard_u=0.70,
        nearest_neighbor_v=0.11,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=6.0,
        lattice_spacing_y_angstrom=8.0,
        relative_permittivity=4.0,
        short_range_shell_overrides=((1, 1, 0.07),),
    )
    psi = _normalized_symmetric_wavefunction(p.n_sites)
    reference = interaction_expectation(psi, p)
    candidate = molecular_interaction_expectation(
        psi, legacy_rectangular_pair_model(p)
    )
    assert np.isclose(candidate, reference, rtol=0.0, atol=2.0e-16)


def test_named_legacy_contact_probabilities_match_rectangular_observables() -> None:
    p = BipolaronParameters(nx=5, ny=4, pair_position=8)
    psi = _normalized_symmetric_wavefunction(p.n_sites, seed=4002)
    ground = BipolaronGroundState(energy=-1.0, wavefunction=psi)
    reference = pair_observables(ground, p)
    candidate = molecular_pair_observables(
        psi,
        legacy_rectangular_pair_model(p).lattice,
        legacy_rectangular_observable_contacts(),
    )
    assert np.isclose(candidate.onsite_probability, reference.onsite_probability)
    assert np.isclose(
        candidate.contact_probabilities["nearest_x"],
        reference.nearest_neighbour_x_probability,
    )
    assert np.isclose(
        candidate.contact_probabilities["nearest_y"],
        reference.nearest_neighbour_y_probability,
    )
    assert np.isclose(candidate.one_body_ipr, reference.one_body_ipr)


def test_skew_two_basis_continuum_and_contact_override_use_physical_geometry() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=3,
        n2=3,
        a1_angstrom=(5.0, 0.0),
        a2_angstrom=(1.0, 4.0),
        basis=(
            MolecularBasisSite("A", (0.0, 0.0)),
            MolecularBasisSite("B", (0.5, 0.5)),
        ),
    )
    contact = PairContactFamily("AB0", 0, 1, ((0, 0),))
    model = MolecularPairInteractionModel(
        lattice=lattice,
        hubbard_u_ev=0.65,
        continuum_coulomb=True,
        relative_permittivity=4.0,
        contact_families=(contact,),
        contact_overrides_ev=(("AB0", 0.12),),
    )
    interaction = molecular_pair_interaction_matrix(model)
    a0 = lattice.site_index(0, 0, 0)
    b0 = lattice.site_index(0, 0, 1)
    a1 = lattice.site_index(0, 1, 0)
    assert interaction[a0, a0] == 0.65
    assert interaction[a0, b0] == 0.12
    distance = lattice.minimum_image_distances_angstrom()[a0, a1]
    expected = 14.3996454784255 / (4.0 * distance)
    assert np.isclose(interaction[a0, a1], expected)


def test_contact_masks_include_all_declared_offsets_and_are_symmetric() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=4,
        n2=4,
        a1_angstrom=(1.0, 0.0),
        a2_angstrom=(0.0, 1.0),
        basis=(MolecularBasisSite("A", (0.0, 0.0)),),
    )
    diagonal = PairContactFamily("diag", 0, 0, ((1, 1), (1, -1)))
    mask = contact_family_mask(lattice, diagonal)
    assert np.array_equal(mask, mask.T)
    assert np.all(np.diag(mask) == 0)
    assert np.all(np.sum(mask, axis=1) == 4)


def test_overlapping_overridden_contact_families_are_rejected() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=4,
        n2=4,
        a1_angstrom=(1.0, 0.0),
        a2_angstrom=(0.0, 1.0),
        basis=(MolecularBasisSite("A", (0.0, 0.0)),),
    )
    first = PairContactFamily("first", 0, 0, ((1, 0),))
    second = PairContactFamily("second", 0, 0, ((1, 0), (0, 1)))
    model = MolecularPairInteractionModel(
        lattice=lattice,
        contact_families=(first, second),
        contact_overrides_ev=(("first", 0.1), ("second", 0.2)),
    )
    with pytest.raises(ValueError, match="overlapping"):
        molecular_pair_interaction_matrix(model)


def test_physical_pair_observables_use_nonorthogonal_minimum_image_distance() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=2,
        n2=2,
        a1_angstrom=(2.0, 0.0),
        a2_angstrom=(1.0, 2.0),
        basis=(MolecularBasisSite("A", (0.0, 0.0)),),
    )
    psi = np.zeros((lattice.n_sites, lattice.n_sites), dtype=float)
    first = lattice.site_index(0, 0, 0)
    second = lattice.site_index(1, 1, 0)
    amplitude = 1.0 / np.sqrt(2.0)
    psi[first, second] = amplitude
    psi[second, first] = amplitude
    obs = molecular_pair_observables(psi, lattice)
    assert np.isclose(obs.mean_separation_angstrom, np.sqrt(5.0))
    assert np.isclose(obs.rms_separation_angstrom, np.sqrt(5.0))
    assert np.isclose(obs.onsite_probability, 0.0)
