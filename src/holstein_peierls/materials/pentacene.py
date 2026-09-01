"""Provenance-rich pentacene records for the material-specific G5 stage.

G5a fixed the crystallographic geometry. G5b established a signed low-temperature
band reference and a cross-source 293 K candidate. G5c adds the strongest
room-temperature single-source signed evidence currently available in the
project: the experimental ARPES tight-binding fit of Neef et al. together with
its independent 295 K MD + FO-DFT transfer-integral statistics.

The Neef work remains a preprint as of September 2026. It is therefore promoted
here as an experimental room-temperature *reference*, but not as the final
peer-reviewed material parameterization. Its dynamical-disorder standard
deviations are evidence records only; they are not silently converted into
Peierls derivatives or elastic constants.
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


@dataclass(frozen=True, slots=True)
class SignedTransferIntegralReference:
    """One signed experimental/computational transfer-integral reference."""

    label: str
    value_mev: float
    uncertainty_mev: float
    geometry_group: str
    source_doi: str
    evidence_type: str
    temperature_description: str
    peer_reviewed: bool


@dataclass(frozen=True, slots=True)
class DynamicalDisorderEvidence:
    """Thermal hopping statistics that must not be confused with Peierls derivatives."""

    label: str
    mean_mev: float
    standard_deviation_mev: float
    geometry_group: str
    temperature_k: float
    method: str
    source_doi: str
    peer_reviewed: bool


PENTACENE_293K = PentaceneCrystalRecord(
    temperature_k=293.0,
    a_angstrom=6.266,
    b_angstrom=7.775,
    c_angstrom=14.530,
    alpha_deg=76.475,
    beta_deg=87.682,
    gamma_deg=84.684,
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
    ccdc_id="170187",
)

STEHR_2011_HOLE_COUPLINGS = (
    TransferIntegralEvidence("V1", 90.69, (1, -1, 0), "10.1103/PhysRevB.83.155208", False, "diag_minus_AB"),
    TransferIntegralEvidence("V2", 55.05, (1, 1, 0), "10.1103/PhysRevB.83.155208", False, "diag_plus_AB"),
    TransferIntegralEvidence("V3", 39.68, (1, 0, 0), "10.1103/PhysRevB.83.155208", False, "a_axis_same_basis"),
    TransferIntegralEvidence("V4", 36.62, (1, 0, 0), "10.1103/PhysRevB.83.155208", False, "a_axis_same_basis"),
)

DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV = (
    ("a_same_basis", 0.031),
    ("diag_plus_AB", -0.056),
    ("diag_minus_AB", 0.091),
)
DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV = 0.042
DEWIJS_2003_SOURCE_DOI = "10.1016/S0379-6779(03)00020-1"

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

NEEF_2024_SOURCE_DOI = "10.48550/arXiv.2412.06030"
NEEF_2024_PEER_REVIEWED = False
NEEF_2024_ARPES_HOLE_HOPPINGS = (
    SignedTransferIntegralReference("t_a", 35.0, 10.0, "a_same_basis", NEEF_2024_SOURCE_DOI, "room-temperature single-crystal ARPES tight-binding fit", "room temperature", NEEF_2024_PEER_REVIEWED),
    SignedTransferIntegralReference("t_plus", 55.0, 5.0, "diag_plus_AB", NEEF_2024_SOURCE_DOI, "room-temperature single-crystal ARPES tight-binding fit", "room temperature", NEEF_2024_PEER_REVIEWED),
    SignedTransferIntegralReference("t_minus", -70.0, 5.0, "diag_minus_AB", NEEF_2024_SOURCE_DOI, "room-temperature single-crystal ARPES tight-binding fit", "room temperature", NEEF_2024_PEER_REVIEWED),
)

NEEF_2024_MD_TEMPERATURE_K = 295.0
NEEF_2024_MD_HOLE_HOPPING_STATISTICS = (
    DynamicalDisorderEvidence("t_a", 32.0, 12.0, "a_same_basis", NEEF_2024_MD_TEMPERATURE_K, "ab-initio-quality MD plus FO-DFT (fragment-orbital DFT) on extracted dimers", NEEF_2024_SOURCE_DOI, NEEF_2024_PEER_REVIEWED),
    DynamicalDisorderEvidence("t_plus", 39.5, 18.0, "diag_plus_AB", NEEF_2024_MD_TEMPERATURE_K, "ab-initio-quality MD plus FO-DFT (fragment-orbital DFT) on extracted dimers", NEEF_2024_SOURCE_DOI, NEEF_2024_PEER_REVIEWED),
    DynamicalDisorderEvidence("t_minus", -78.8, 18.4, "diag_minus_AB", NEEF_2024_MD_TEMPERATURE_K, "ab-initio-quality MD plus FO-DFT (fragment-orbital DFT) on extracted dimers", NEEF_2024_SOURCE_DOI, NEEF_2024_PEER_REVIEWED),
)
NEEF_2024_MEAN_TRANSLATIONAL_FLUCTUATION_ANGSTROM = 0.21

UNRESOLVED_PENTACENE_FIELDS = (
    "peer_reviewed_temperature_matched_signed_hopping_parameterization",
    "bond_resolved_peierls_derivatives",
    "effective_bond_stiffness_matrices",
    "screened_onsite_hubbard_u",
    "screened_short_range_contact_interactions",
    "long_range_dielectric_convention",
    "holstein_reorganization_mapping",
)


def _projected_lattice(record: PentaceneCrystalRecord, n1: int, n2: int) -> PeriodicMolecularLattice2D:
    a1, a2 = record.projected_ab_bravais_vectors()
    basis = tuple(MolecularBasisSite(label, position) for label, position in zip(("A", "B"), record.projected_basis_fractional))
    return PeriodicMolecularLattice2D(n1=n1, n2=n2, a1_angstrom=a1, a2_angstrom=a2, basis=basis)


def _candidate_transport_lattice(record: PentaceneCrystalRecord, n1: int, n2: int) -> PeriodicMolecularLattice2D:
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
    return _projected_lattice(PENTACENE_293K, n1, n2)


def pentacene_293k_candidate_transport_lattice(n1: int, n2: int) -> PeriodicMolecularLattice2D:
    return _candidate_transport_lattice(PENTACENE_293K, n1, n2)


def pentacene_90k_dewijs_reference_lattice(n1: int, n2: int) -> PeriodicMolecularLattice2D:
    return _candidate_transport_lattice(PENTACENE_90K, n1, n2)


def pentacene_90k_dewijs_homo_model(n1: int, n2: int) -> MolecularTightBindingModel:
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
        onsite_energies_ev=(-DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV, 0.0),
    )


def pentacene_293k_cross_source_hole_model(n1: int, n2: int) -> MolecularTightBindingModel:
    return MolecularTightBindingModel(
        lattice=pentacene_293k_candidate_transport_lattice(n1, n2),
        bond_transfer_integrals_ev=PENTACENE_293K_CROSS_SOURCE_HOLE_HOPPINGS_EV,
        onsite_energies_ev=(0.0, 0.0),
    )


def pentacene_room_temperature_neef_arpes_homo_model(n1: int, n2: int) -> MolecularTightBindingModel:
    return MolecularTightBindingModel(
        lattice=pentacene_293k_candidate_transport_lattice(n1, n2),
        bond_transfer_integrals_ev=(
            ("a_AA", 0.035),
            ("a_BB", 0.035),
            ("diag_plus_AB_forward", 0.055),
            ("diag_plus_AB_backward", 0.055),
            ("diag_minus_AB_forward", -0.070),
            ("diag_minus_AB_backward", -0.070),
        ),
        onsite_energies_ev=(0.0, 0.0),
    )
