"""Provenance-rich pentacene records for the material-specific G5 stage.

Only quantities that can be mapped without mixing incompatible conventions are
made executable here. In particular, the 293 K crystal geometry is fixed, but
published Marcus-theory electronic couplings remain unsigned evidence until a
source establishes their sign and one-to-one mapping to the graph families.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    """Published transfer-integral evidence not yet promoted to a signed hopping."""

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


STEHR_2011_HOLE_COUPLINGS = (
    TransferIntegralEvidence(
        "C", 90.69, (1, 1, 0), "10.1103/PhysRevB.83.155208", False, "diagonal_AB"
    ),
    TransferIntegralEvidence(
        "B", 55.05, (1, 1, 0), "10.1103/PhysRevB.83.155208", False, "diagonal_AB"
    ),
    TransferIntegralEvidence(
        "A", 39.68, (1, 0, 0), "10.1103/PhysRevB.83.155208", False, "a_axis_same_basis"
    ),
    TransferIntegralEvidence(
        "A_prime",
        36.62,
        (1, 0, 0),
        "10.1103/PhysRevB.83.155208",
        False,
        "a_axis_same_basis",
    ),
)


UNRESOLVED_PENTACENE_FIELDS = (
    "signed_hopping_assignment",
    "bond_resolved_peierls_derivatives",
    "effective_bond_stiffness_matrices",
    "screened_onsite_hubbard_u",
    "screened_short_range_contact_interactions",
    "long_range_dielectric_convention",
    "holstein_reorganization_mapping",
)


def pentacene_293k_projected_lattice(n1: int, n2: int) -> PeriodicMolecularLattice2D:
    """Return the 293 K bulk pentacene a-b geometry without a transport graph."""
    a1, a2 = PENTACENE_293K.projected_ab_bravais_vectors()
    basis = tuple(
        MolecularBasisSite(label, position)
        for label, position in zip(
            ("A", "B"), PENTACENE_293K.projected_basis_fractional
        )
    )
    return PeriodicMolecularLattice2D(
        n1=n1,
        n2=n2,
        a1_angstrom=a1,
        a2_angstrom=a2,
        basis=basis,
    )


def pentacene_293k_candidate_transport_lattice(
    n1: int, n2: int
) -> PeriodicMolecularLattice2D:
    """Return the minimal four-family in-plane herringbone geometry.

    The labels are deliberately geometric rather than A/A'/B/C electronic
    labels. The two same-basis a-axis paths and two A-B diagonals are the four
    dominant geometric families. Published unsigned Marcus couplings are not
    assigned one-to-one here because their signs and family correspondence must
    be established from a compatible electronic-structure source first.
    """
    geometry = pentacene_293k_projected_lattice(n1, n2)
    return PeriodicMolecularLattice2D(
        n1=geometry.n1,
        n2=geometry.n2,
        a1_angstrom=geometry.a1_angstrom,
        a2_angstrom=geometry.a2_angstrom,
        basis=geometry.basis,
        bonds=(
            BondFamily("a_AA", 0, 0, (1, 0)),
            BondFamily("a_BB", 1, 1, (1, 0)),
            BondFamily("diag_plus_AB", 0, 1, (0, 0)),
            BondFamily("diag_minus_AB", 0, 1, (-1, 0)),
        ),
    )
