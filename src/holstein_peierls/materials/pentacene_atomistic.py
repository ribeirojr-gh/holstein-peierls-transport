"""Atomistic 293 K bulk-pentacene reconstruction from Mattheus et al.

G5g reconstructs the two complete C22H14 molecules in the P-1 unit cell from
the published independent fractional coordinates and crystallographic inversion
operations.  The tabulated coordinates are the rounded values printed by the
IUCr article/supporting-data view; the module records that provenance explicitly
and does not claim hidden precision from an unavailable raw-CIF parse.

Code basis convention (matching ``materials.pentacene``):

- basis A: crystallographic inversion center (0, 0, 0), sites C12--C22;
- basis B: crystallographic inversion center (1/2, 1/2, 0), sites C1--C11.

This is intentionally the reverse of the order used in one sentence of the
crystallographic numbering discussion, but it preserves the established G5
A/B lattice convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from ..dimer_finite_difference import RigidDimerReference, RigidMoleculeGeometry
from .pentacene import PENTACENE_293K_CROSS_SOURCE_STATUS

FloatArray = NDArray[np.float64]

MATT_HEUS_2001_SOURCE_DOI = "10.1107/S010827010100703X"
MATT_HEUS_2001_CIF_PATH = "sk1477sup1.cif"
MATT_HEUS_2001_293K_CCDC = "170186"
MATT_HEUS_2001_COORDINATE_STATUS = (
    "published rounded fractional coordinates from the IUCr article/supporting-data view"
)


@dataclass(frozen=True, slots=True)
class TriclinicCell3D:
    """Triclinic crystallographic cell in the conventional Cartesian embedding."""

    a_angstrom: float
    b_angstrom: float
    c_angstrom: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    @property
    def vectors(self) -> FloatArray:
        alpha = math.radians(self.alpha_deg)
        beta = math.radians(self.beta_deg)
        gamma = math.radians(self.gamma_deg)
        sin_gamma = math.sin(gamma)
        if abs(sin_gamma) < 1.0e-14:
            raise ValueError("gamma produces a singular triclinic embedding")
        a = np.array([self.a_angstrom, 0.0, 0.0], dtype=np.float64)
        b = np.array(
            [self.b_angstrom * math.cos(gamma), self.b_angstrom * sin_gamma, 0.0],
            dtype=np.float64,
        )
        cx = self.c_angstrom * math.cos(beta)
        cy = self.c_angstrom * (
            math.cos(alpha) - math.cos(beta) * math.cos(gamma)
        ) / sin_gamma
        cz_squared = self.c_angstrom**2 - cx**2 - cy**2
        if cz_squared <= 0.0:
            raise ValueError("cell angles and lengths do not define a positive volume")
        c = np.array([cx, cy, math.sqrt(cz_squared)], dtype=np.float64)
        return np.column_stack((a, b, c))

    @property
    def volume_angstrom3(self) -> float:
        return float(np.linalg.det(self.vectors))

    def fractional_to_cartesian(self, fractional: FloatArray) -> FloatArray:
        values = np.asarray(fractional, dtype=np.float64)
        if values.shape[-1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("fractional coordinates must be finite (..., 3) values")
        return values @ self.vectors.T

    def translation_cartesian(self, offset: tuple[int, int, int]) -> FloatArray:
        return self.vectors @ np.asarray(offset, dtype=np.float64)


PENTACENE_293K_CELL_3D = TriclinicCell3D(
    a_angstrom=6.266,
    b_angstrom=7.775,
    c_angstrom=14.530,
    alpha_deg=76.475,
    beta_deg=87.682,
    gamma_deg=84.684,
)


@dataclass(frozen=True, slots=True)
class PublishedAtomSite:
    label: str
    symbol: str
    fractional: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.label or self.symbol not in {"C", "H"}:
            raise ValueError("pentacene sites require a label and C/H symbol")
        values = np.asarray(self.fractional, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("fractional atom position must be a finite three-vector")


# Molecule B in the code convention: crystallographic C1--C11 half molecule,
# centered at (1/2, 1/2, 0), with seven independent constrained H positions.
_BASIS_B_INDEPENDENT_SITES = (
    PublishedAtomSite("C1", "C", (0.6994, 0.4035, 0.0149)),
    PublishedAtomSite("C2", "C", (0.5410, 0.3719, 0.0864)),
    PublishedAtomSite("C3", "C", (0.5776, 0.2470, 0.1736)),
    PublishedAtomSite("C4", "C", (0.4200, 0.2212, 0.2443)),
    PublishedAtomSite("C5", "C", (0.4551, 0.0979, 0.3327)),
    PublishedAtomSite("C6", "C", (0.2986, 0.0764, 0.4009)),
    PublishedAtomSite("C7", "C", (0.0967, 0.1741, 0.3851)),
    PublishedAtomSite("C8", "C", (0.0548, 0.2913, 0.3020)),
    PublishedAtomSite("C9", "C", (0.2132, 0.3216, 0.2283)),
    PublishedAtomSite("C10", "C", (0.1755, 0.4436, 0.1432)),
    PublishedAtomSite("C11", "C", (0.6641, 0.5285, -0.0709)),
    PublishedAtomSite("H1", "H", (0.83247, 0.33915, 0.02479)),
    PublishedAtomSite("H3", "H", (0.70958, 0.18106, 0.18368)),
    PublishedAtomSite("H5", "H", (0.58649, 0.03122, 0.34389)),
    PublishedAtomSite("H6", "H", (0.32476, -0.00351, 0.45863)),
    PublishedAtomSite("H7", "H", (-0.00866, 0.15758, 0.43255)),
    PublishedAtomSite("H8", "H", (-0.07976, 0.35329, 0.29283)),
    PublishedAtomSite("H10", "H", (0.04263, 0.50821, 0.13324)),
)

# Molecule A in the code convention: crystallographic C12--C22 half molecule,
# centered at the origin.
_BASIS_A_INDEPENDENT_SITES = (
    PublishedAtomSite("C12", "C", (-0.1814, -0.0442, -0.0408)),
    PublishedAtomSite("C13", "C", (-0.1548, -0.0991, 0.0582)),
    PublishedAtomSite("C14", "C", (-0.3054, -0.1986, 0.1172)),
    PublishedAtomSite("C15", "C", (-0.2797, -0.2530, 0.2129)),
    PublishedAtomSite("C16", "C", (-0.4310, -0.3542, 0.2754)),
    PublishedAtomSite("C17", "C", (-0.4010, -0.4079, 0.3704)),
    PublishedAtomSite("C18", "C", (-0.2131, -0.3636, 0.4114)),
    PublishedAtomSite("C19", "C", (-0.0681, -0.2681, 0.3564)),
    PublishedAtomSite("C20", "C", (-0.0957, -0.2083, 0.2554)),
    PublishedAtomSite("C21", "C", (0.0555, -0.1106, 0.1985)),
    PublishedAtomSite("C22", "C", (-0.0332, 0.0537, -0.1004)),
    PublishedAtomSite("H12", "H", (-0.30151, -0.07431, -0.06712)),
    PublishedAtomSite("H14", "H", (-0.42529, -0.22804, 0.09045)),
    PublishedAtomSite("H16", "H", (-0.55280, -0.38368, 0.25004)),
    PublishedAtomSite("H17", "H", (-0.50152, -0.47306, 0.40892)),
    PublishedAtomSite("H18", "H", (-0.19243, -0.40143, 0.47625)),
    PublishedAtomSite("H19", "H", (0.05188, -0.24014, 0.38370)),
    PublishedAtomSite("H21", "H", (0.17444, -0.08227, 0.22636)),
)

_ATOMIC_MASS_U = {"C": 12.011, "H": 1.008}


@dataclass(frozen=True, slots=True)
class AtomisticPentaceneMolecule:
    """One reconstructed inversion-symmetric pentacene molecule."""

    basis_label: str
    atom_labels: tuple[str, ...]
    fractional_coordinates: tuple[tuple[float, float, float], ...]
    rigid_geometry: RigidMoleculeGeometry
    inversion_center_fractional: tuple[float, float, float]
    principal_moments_u_angstrom2: tuple[float, float, float]
    principal_axes_global: tuple[tuple[float, float, float], ...]

    @property
    def axes(self) -> FloatArray:
        return np.asarray(self.principal_axes_global, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AtomisticPentaceneDimer:
    """One explicitly oriented G5 herringbone bond with atomistic partners."""

    family_label: str
    source_basis: str
    target_basis: str
    target_cell_offset: tuple[int, int, int]
    reference: RigidDimerReference

    @property
    def centroid_displacement_angstrom(self) -> FloatArray:
        return self.reference.molecule_b.pivot - self.reference.molecule_a.pivot


def _deterministic_principal_axes(
    symbols: tuple[str, ...], coordinates: FloatArray, pivot: FloatArray
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    inertia = np.zeros((3, 3), dtype=np.float64)
    for symbol, coordinate in zip(symbols, coordinates):
        relative = coordinate - pivot
        mass = _ATOMIC_MASS_U[symbol]
        inertia += mass * (
            float(relative @ relative) * np.eye(3) - np.outer(relative, relative)
        )
    moments, axes = np.linalg.eigh(inertia)
    order = np.argsort(moments)
    moments = moments[order]
    axes = axes[:, order]

    # Principal eigenvectors are defined only up to sign.  Fix long and short
    # signs by requiring the largest-magnitude Cartesian component to be
    # positive, then define the normal by a right-handed cross product.
    for column in (0, 1):
        vector = axes[:, column]
        pivot_index = int(np.argmax(np.abs(vector)))
        if vector[pivot_index] < 0.0:
            axes[:, column] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] /= np.linalg.norm(axes[:, 2])
    return (
        tuple(float(value) for value in moments),
        tuple(tuple(float(value) for value in row) for row in axes),
    )


def _reconstruct_inversion_molecule(
    basis_label: str,
    independent_sites: tuple[PublishedAtomSite, ...],
    inversion_center: tuple[float, float, float],
) -> AtomisticPentaceneMolecule:
    center = np.asarray(inversion_center, dtype=np.float64)
    original = np.asarray([site.fractional for site in independent_sites], dtype=np.float64)
    generated = 2.0 * center - original
    fractional = np.vstack((original, generated))
    symbols = tuple(site.symbol for site in independent_sites) * 2
    labels = tuple(site.label for site in independent_sites) + tuple(
        f"{site.label}_inv" for site in independent_sites
    )
    cartesian = PENTACENE_293K_CELL_3D.fractional_to_cartesian(fractional)
    pivot = PENTACENE_293K_CELL_3D.fractional_to_cartesian(center)
    moments, axes = _deterministic_principal_axes(symbols, cartesian, pivot)
    geometry = RigidMoleculeGeometry(
        symbols=symbols,
        coordinates_angstrom=tuple(tuple(float(value) for value in row) for row in cartesian),
        pivot_angstrom=tuple(float(value) for value in pivot),
    )
    return AtomisticPentaceneMolecule(
        basis_label=basis_label,
        atom_labels=labels,
        fractional_coordinates=tuple(
            tuple(float(value) for value in row) for row in fractional
        ),
        rigid_geometry=geometry,
        inversion_center_fractional=inversion_center,
        principal_moments_u_angstrom2=moments,
        principal_axes_global=axes,
    )


def pentacene_293k_atomistic_basis_a() -> AtomisticPentaceneMolecule:
    """Return code-basis A, centered at crystallographic (0,0,0)."""
    return _reconstruct_inversion_molecule(
        "A", _BASIS_A_INDEPENDENT_SITES, (0.0, 0.0, 0.0)
    )


def pentacene_293k_atomistic_basis_b() -> AtomisticPentaceneMolecule:
    """Return code-basis B, centered at crystallographic (1/2,1/2,0)."""
    return _reconstruct_inversion_molecule(
        "B", _BASIS_B_INDEPENDENT_SITES, (0.5, 0.5, 0.0)
    )


def _translated_geometry(
    geometry: RigidMoleculeGeometry, offset: tuple[int, int, int]
) -> RigidMoleculeGeometry:
    return geometry.translated(PENTACENE_293K_CELL_3D.translation_cartesian(offset))


_DIMER_DEFINITIONS = {
    "a_AA": ("A", "A", (1, 0, 0)),
    "a_BB": ("B", "B", (1, 0, 0)),
    "diag_plus_AB_forward": ("A", "B", (0, 0, 0)),
    "diag_plus_AB_backward": ("A", "B", (-1, -1, 0)),
    "diag_minus_AB_forward": ("A", "B", (0, -1, 0)),
    "diag_minus_AB_backward": ("A", "B", (-1, 0, 0)),
}
PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES = tuple(_DIMER_DEFINITIONS)


def pentacene_293k_atomistic_dimer(family_label: str) -> AtomisticPentaceneDimer:
    """Return an atomistic dimer for one explicit oriented G5 bond family.

    The local l/s/n frame is the deterministic principal-axis frame of the
    fixed/source molecule.  The target molecule is the partner perturbed by G5e.
    This is an explicit coordinate convention compatible with Neef's projection
    onto the principal axes of one molecule; it is not claimed as a unique gauge.
    """
    if family_label not in _DIMER_DEFINITIONS:
        raise ValueError(f"unknown pentacene atomistic dimer family: {family_label}")
    source_basis, target_basis, offset = _DIMER_DEFINITIONS[family_label]
    basis = {
        "A": pentacene_293k_atomistic_basis_a(),
        "B": pentacene_293k_atomistic_basis_b(),
    }
    source = basis[source_basis]
    target = basis[target_basis]
    target_geometry = _translated_geometry(target.rigid_geometry, offset)
    reference = RigidDimerReference(
        molecule_a=source.rigid_geometry,
        molecule_b=target_geometry,
        local_axes_global=source.principal_axes_global,
    )
    return AtomisticPentaceneDimer(
        family_label=family_label,
        source_basis=source_basis,
        target_basis=target_basis,
        target_cell_offset=offset,
        reference=reference,
    )


def pentacene_293k_atomistic_dimer_set() -> tuple[AtomisticPentaceneDimer, ...]:
    """Return all six explicit oriented herringbone dimers in graph order."""
    return tuple(
        pentacene_293k_atomistic_dimer(label)
        for label in PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES
    )
