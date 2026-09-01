"""Peer-reviewed pentacene electron-phonon-coupling evidence for G5d.

The records in this module intentionally preserve the representation used by the
source.  Gnoli et al. report normal-mode / Brillouin-zone deformation-potential
EPC strengths.  Those quantities are useful material benchmarks, but they are
not the local bond derivatives ``g_b = d t_b / d q`` consumed by the G3
molecular Peierls model.  No automatic conversion is provided.
"""

from __future__ import annotations

from dataclasses import dataclass


GNOLI_2025_SOURCE_DOI = "10.1021/acs.jpcc.5c04906"
GNOLI_2025_DATASET_DOI = "10.5281/zenodo.17368135"
GNOLI_2025_EPC_FINITE_DISPLACEMENT_ANGSTROM = 0.0025
GNOLI_2025_REPRESENTATION = "normal_mode_bandwidth_deformation_potential"


@dataclass(frozen=True, slots=True)
class ModalEpcEvidence:
    """One mode-resolved nonlocal EPC benchmark from a periodic calculation.

    ``epc_ev_per_angstrom`` is the source's mode/bandwidth deformation
    potential.  It must not be interpreted as a local molecular-bond transfer-
    integral derivative without an explicit projection through the phonon
    eigenvector and the chosen local-coordinate convention.
    """

    polymorph: str
    q_point_label: str
    frequency_cm_inverse: float
    epc_ev_per_angstrom: float
    dominant_coordinate: str
    dominant_coordinate_fraction: float
    source_doi: str = GNOLI_2025_SOURCE_DOI
    dataset_doi: str = GNOLI_2025_DATASET_DOI
    representation: str = GNOLI_2025_REPRESENTATION
    direct_g3_compatible: bool = False

    def __post_init__(self) -> None:
        if self.frequency_cm_inverse <= 0.0:
            raise ValueError("mode frequency must be positive")
        if self.epc_ev_per_angstrom < 0.0:
            raise ValueError("EPC magnitude must be non-negative")
        if not 0.0 <= self.dominant_coordinate_fraction <= 1.0:
            raise ValueError("coordinate fraction must lie in [0, 1]")
        if self.direct_g3_compatible:
            raise ValueError(
                "modal EPC evidence cannot be marked direct-G3-compatible without "
                "an explicit local-coordinate projection"
            )


# Low-temperature bulk pentacene: the two strongest reported EPC points.
# The source's mode decomposition shows that both are dominated by translation
# along the molecular long inertia axis (TL), with 67% and 46% contributions.
GNOLI_2025_LT_DOMINANT_EPC = (
    ModalEpcEvidence(
        polymorph="LT",
        q_point_label="U",
        frequency_cm_inverse=26.7,
        epc_ev_per_angstrom=0.27,
        dominant_coordinate="translation_long_inertia_axis",
        dominant_coordinate_fraction=0.67,
    ),
    ModalEpcEvidence(
        polymorph="LT",
        q_point_label="X",
        frequency_cm_inverse=39.2,
        epc_ev_per_angstrom=0.24,
        dominant_coordinate="translation_long_inertia_axis",
        dominant_coordinate_fraction=0.46,
    ),
)


def g5d_projection_requirements() -> tuple[str, ...]:
    """Return the missing ingredients before modal EPC can become G3 parameters."""

    return (
        "phonon_eigenvector_in_molecular_local_coordinates",
        "normal_mode_coordinate_normalization",
        "bond_resolved_signed_transfer_integral_response",
        "mapping_from_periodic_mode_to_local_bond_coordinate_differences",
        "consistent_elastic_or_phonon_hessian_representation",
    )
