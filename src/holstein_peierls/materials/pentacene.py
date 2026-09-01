"""Provenance-rich pentacene records for the material-specific G5 stage.

G5a fixed the crystallographic geometry without promoting a transport model.
G5b adds two deliberately distinct electronic layers:

1. a signed 90 K HOMO tight-binding reference fitted by de Wijs et al. to a DFT
   band structure on the Mattheus low-temperature crystal; and
2. a 293 K *cross-source candidate* that combines the Stehr et al. room-
   temperature coupling magnitudes with the gauge-fixed sign pattern established
   by the signed de Wijs band fit.

The second object is useful for band-topology and scale checks, but it is not
claimed to be a final single-source pentacene parameterization. Coulomb,
Holstein, and Peierls material parameters remain unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..molecular_hamiltonian import MolecularTightBindingModel
from ..molecular_lattice import BondFamily, MolecularBasisSite, PeriodicMolecularLattice2D


@dataclass(frozen=True, slots=True)
class PentaceneCrystalRecord:
    """Traceable bulk pentacene crystallographic record."""

    temperature_k: float
    a_angstrom: float
    b_angstrom: float
    c_angstrom: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    basis_fractional_3d: tuple[tuple[float, float, float], ...]
    source_doi: str
    ccdc_id: str

    def projected_ab_bravais_vectors(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the Cartesian 2D representation of the crystallographic a-b plane."""
        gamma = math.radians(self.gamma_deg)
        a1 = (self.a_angstrom, 0.0)
        a2 = (
            self.b_angstrom * math.cos(gamma),
            self.b_angstrom * math.sin(gamma),
        )
        return a1, a2

    @property
    def projected_basis_fractional(self) -> tuple[tuple[float, float], ...]:
        return tuple((x, y) for x, y, _ in self.basis_fractional_3d)


@dataclass(frozen=True, slots=True)
class TransferIntegralEvidence:
    """Published transfer-integral evidence with explicit promotion status."""

    label: str
    magnitude_mev: float
    direction_miller: tuple[int, int, int]
    source_doi: str
    signed_value_known: bool
    candidate_geometry_group: str


PENTACENE_293K = PentaceneCrystalRecord(
    temperature_k=293.0,
    a_angstrom=6.266,
    b_angstrom=7.775,
    c_angstrom=14.530,
    alpha_deg=76.475,
    beta_deg=87.682,
    gamma_deg=84.684,
    # Mattheus et al. place crystallographic inversion centers on the two
    # molecules at (0,0,0) and (1/2,1/2,0).
    basis_fractional_3d=((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
    source_doi="10.1107/S010827010100703X",
    ccdc_id="170186",
)

PENTACENE_90K = PentaceneCrystalRecord(
    temperature_k=90.0,
    a_angstrom=6.239,
    b_angstrom=7.636,
    c_angstrom=14.330,
    alpha_deg=76.978,
    beta_deg=88.136,
    gamma_deg=84.415,
    basis_fractional_3d=((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
    source_doi="10.1107/S010827010100703X",
    ccdc_id="170185",
)


# Stehr et al. use the 293 K Mattheus crystal and report magnitudes because their
# Marcus rates depend on |V|^2. Their text identifies V1 with [1 -1 0] and V2
# with [1 1 0]. V3/V4 are the two same-basis a-axis contacts in the minimal
# in-plane graph. The A/B assignment of V3 versus V4 is basis-label dependent.
STEHR_2011_HOLE_COUPLINGS = (
    TransferIntegralEvidence(
        "V1",
        90.69,
        (1, -1, 0),
        "10.1103/PhysRevB.83.155208",
        False,
        "diag_minus_AB",
    ),
    TransferIntegralEvidence(
        "V2",
        55.05,
        (1, 1, 0),
        "10.1103/PhysRevB.83.155208",
        False,
        "diag_plus_AB",
    ),
    TransferIntegralEvidence(
        "V3",
        39.68,
        (1, 0, 0),
        "10.1103/PhysRevB.83.155208",
        False,
        "a_axis_same_basis",
    ),
    TransferIntegralEvidence(
        "V4",
        36.62,
        (1, 0, 0),
        "10.1103/PhysRevB.83.155208",
        False,
        "a_axis_same_basis",
    ),
)


# Signed HOMO transfer integrals from the de Wijs et al. single-layer TB fit on
# the 90 K Mattheus structure. This is one particular molecular-orbital gauge:
# flipping the phase of every B-basis HOMO reverses both A-B signs and leaves the
# band eigenvalues invariant.
DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV = (
    ("a_same_basis", 0.031),
    ("diag_plus_AB", -0.056),
    ("diag_minus_AB", 0.091),
)
DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV = 0.042
DEWIJS_2003_SOURCE_DOI = "10.1016/S0379-6779(03)00020-1"


# Cross-source 293 K candidate: magnitudes from Stehr et al. and the signed
# herringbone pattern from de Wijs et al. Each diagonal direction requires two
# opposite A-B bonds in the finite graph. V3/V4 are assigned deterministically to
# A/B; exchanging them changes basis character but not the two-band eigenvalues
# when onsite energies are equal.
PENTACENE_293K_CROSS_SOURCE_HOLE_HOPPINGS_EV = (
    ("a_AA", 0.03968),
    ("a_BB", 0.03662),
    ("diag_plus_AB_forward", -0.05505),
    ("diag_plus_AB_backward", -0.05505),
    ("diag_minus_AB_forward", 0.09069),
    ("diag_minus_AB_backward", 0.09069),
)
PENTACENE_293K_CROSS_SOURCE_STATUS = (
    "candidate only: 293 K magnitudes from Stehr et al.; relative signs from the "
    "de Wijs et al. HOMO band fit; not a single-source material parameterization"
)


UNRESOLVED_PENTACENE_FIELDS = (
    "single_source_signed_hopping_parameterization",
    "bond_resolved_peierls_derivatives",
    "effective_bond_stiffness_matrices",
    "screened_onsite_hubbard_u",
    "screened_short_range_contact_interactions",
    "long_range_dielectric_convention",
    "holstein_reorganization_mapping",
)


def _projected_lattice(
    record: PentaceneCrystalRecord,
    n1: int,
    n2: int,
) -> PeriodicMolecularLattice2D:
    a1, a2 = record.projected_ab_bravais_vectors()
    basis = tuple(
        MolecularBasisSite(label, position)
        for label, position in zip(("A", "B"), record.projected_basis_fractional)
    )
    return PeriodicMolecularLattice2D(
        n1=n1,
        n2=n2,
        a1_angstrom=a1,
        a2_angstrom=a2,
        basis=basis,
    )


def _candidate_transport_lattice(
    record: PentaceneCrystalRecord,
    n1: int,
    n2: int,
) -> PeriodicMolecularLattice2D:
    """Return the minimal connected herringbone graph for one a-b layer.

    The two same-basis a-axis contacts are each represented by one translational
    family; their reverse neighbours are generated automatically because source
    and target have the same basis. For A-B diagonals this is not true: each
    physical direction needs both +d and -d representatives. Consequently the
    graph has six bond families but only four independent hopping parameters.
    """
    geometry = _projected_lattice(record, n1, n2)
    return PeriodicMolecularLattice2D(
        n1=geometry.n1,
        n2=geometry.n2,
        a1_angstrom=geometry.a1_angstrom,
        a2_angstrom=geometry.a2_angstrom,
        basis=geometry.basis,
        bonds=(
            BondFamily("a_AA", 0, 0, (1, 0)),
            BondFamily("a_BB", 1, 1, (1, 0)),
            BondFamily("diag_plus_AB_forward", 0, 1, (0, 0)),
            BondFamily("diag_plus_AB_backward", 0, 1, (-1, -1)),
            BondFamily("diag_minus_AB_forward", 0, 1, (0, -1)),
            BondFamily("diag_minus_AB_backward", 0, 1, (-1, 0)),
        ),
    )


def pentacene_293k_projected_lattice(n1: int, n2: int) -> PeriodicMolecularLattice2D:
    """Return the 293 K bulk pentacene a-b geometry without a transport graph."""
    return _projected_lattice(PENTACENE_293K, n1, n2)


def pentacene_293k_candidate_transport_lattice(
    n1: int, n2: int
) -> PeriodicMolecularLattice2D:
    """Return the connected 293 K six-edge/four-parameter herringbone graph."""
    return _candidate_transport_lattice(PENTACENE_293K, n1, n2)


def pentacene_90k_dewijs_reference_lattice(
    n1: int, n2: int
) -> PeriodicMolecularLattice2D:
    """Return the 90 K geometry used by the de Wijs DFT/TB reference."""
    return _candidate_transport_lattice(PENTACENE_90K, n1, n2)


def pentacene_90k_dewijs_homo_model(
    n1: int, n2: int
) -> MolecularTightBindingModel:
    """Return the three-parameter signed HOMO reference in the de Wijs gauge."""
    lattice = pentacene_90k_dewijs_reference_lattice(n1, n2)
    return MolecularTightBindingModel(
        lattice=lattice,
        bond_transfer_integrals_ev=(
            ("a_AA", 0.031),
            ("a_BB", 0.031),
            ("diag_plus_AB_forward", -0.056),
            ("diag_plus_AB_backward", -0.056),
            ("diag_minus_AB_forward", 0.091),
            ("diag_minus_AB_backward", 0.091),
        ),
        # de Wijs et al. fit an inequivalent-molecule offset e=42 meV. Which
        # molecule is called A is a basis-label choice; this convention puts the
        # lower onsite energy on A.
        onsite_energies_ev=(-DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV, 0.0),
    )


def pentacene_293k_cross_source_hole_model(
    n1: int, n2: int
) -> MolecularTightBindingModel:
    """Return the provenance-labelled 293 K signed *candidate* hole model.

    This object must not yet be used to label a polaron or bipolaron result as a
    final material-specific pentacene prediction. It is intended for G5b band
    validation and for determining whether the signed room-temperature magnitude
    pattern is internally consistent with the herringbone band topology.
    """
    return MolecularTightBindingModel(
        lattice=pentacene_293k_candidate_transport_lattice(n1, n2),
        bond_transfer_integrals_ev=PENTACENE_293K_CROSS_SOURCE_HOLE_HOPPINGS_EV,
        onsite_energies_ev=(0.0, 0.0),
    )
