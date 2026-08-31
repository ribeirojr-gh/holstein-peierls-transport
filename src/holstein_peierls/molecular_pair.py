"""Correlated-pair geometry, interactions, and observables on molecular lattices.

Phase G4 keeps this layer parallel to the validated rectangular two-particle
solver. It generalizes continuum Coulomb distances and explicit short-range
replacements to arbitrary molecular bases and named contact families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .molecular_lattice import PeriodicMolecularLattice2D, rectangular_legacy_lattice
from .two_particle.interaction import COULOMB_PREFACTOR_EV_ANGSTROM
from .two_particle.parameters import BipolaronParameters

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PairContactFamily:
    """Symmetry-equivalent distinct-site contacts in a molecular crystal.

    Each offset translates ``target_basis`` relative to ``source_basis`` in
    primitive-cell coordinates. Several offsets may share one label when they
    are intentionally assigned the same effective interaction, e.g. the two
    diagonal orientations of a one-basis rectangular shell.
    """

    label: str
    source_basis: int
    target_basis: int
    cell_offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("contact-family label must be non-empty")
        if self.source_basis < 0 or self.target_basis < 0:
            raise ValueError("contact-family basis indices must be non-negative")
        if not self.cell_offsets:
            raise ValueError("contact family must contain at least one cell offset")
        if len(set(self.cell_offsets)) != len(self.cell_offsets):
            raise ValueError("contact-family cell offsets must be unique")
        for offset in self.cell_offsets:
            if len(offset) != 2 or not all(
                isinstance(value, (int, np.integer)) for value in offset
            ):
                raise ValueError("contact-family cell offsets must contain integers")
            if (
                self.source_basis == self.target_basis
                and tuple(int(value) for value in offset) == (0, 0)
            ):
                raise ValueError("contact family cannot include the onsite pair")


@dataclass(frozen=True, slots=True)
class MolecularPairInteractionModel:
    """Static two-carrier repulsion on a periodic molecular lattice."""

    lattice: PeriodicMolecularLattice2D
    hubbard_u_ev: float = 0.0
    continuum_coulomb: bool = False
    relative_permittivity: float | None = None
    contact_families: tuple[PairContactFamily, ...] = ()
    contact_overrides_ev: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.hubbard_u_ev):
            raise ValueError("hubbard_u_ev must be finite")
        if self.continuum_coulomb:
            if (
                self.relative_permittivity is None
                or not np.isfinite(self.relative_permittivity)
                or self.relative_permittivity <= 0.0
            ):
                raise ValueError(
                    "continuum Coulomb requires a finite positive relative_permittivity"
                )
        elif self.relative_permittivity is not None:
            raise ValueError(
                "relative_permittivity is meaningful only when continuum_coulomb=True"
            )

        labels = [family.label for family in self.contact_families]
        if len(set(labels)) != len(labels):
            raise ValueError("contact-family labels must be unique")
        for family in self.contact_families:
            if (
                family.source_basis >= self.lattice.n_basis
                or family.target_basis >= self.lattice.n_basis
            ):
                raise ValueError(
                    f"contact family {family.label!r} references a basis index outside "
                    f"0..{self.lattice.n_basis - 1}"
                )

        override_labels = [label for label, _ in self.contact_overrides_ev]
        if len(set(override_labels)) != len(override_labels):
            raise ValueError("contact override labels must be unique")
        unknown = set(override_labels) - set(labels)
        if unknown:
            raise ValueError(
                f"contact overrides reference unknown families: {sorted(unknown)}"
            )
        for label, value in self.contact_overrides_ev:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"contact override for {label!r} must be finite and non-negative"
                )

    @property
    def contact_overrides(self) -> dict[str, float]:
        return dict(self.contact_overrides_ev)


def contact_family_mask(
    lattice: PeriodicMolecularLattice2D,
    family: PairContactFamily,
) -> BoolArray:
    """Return a symmetric ordered-pair mask for one named contact family."""
    if family.source_basis >= lattice.n_basis or family.target_basis >= lattice.n_basis:
        raise ValueError("contact family references a basis index outside the lattice")

    mask = np.zeros((lattice.n_sites, lattice.n_sites), dtype=bool)
    for delta_x, delta_y in family.cell_offsets:
        for cell_y in range(lattice.n2):
            for cell_x in range(lattice.n1):
                source = lattice.site_index(cell_y, cell_x, family.source_basis)
                target = lattice.site_index(
                    (cell_y + delta_y) % lattice.n2,
                    (cell_x + delta_x) % lattice.n1,
                    family.target_basis,
                )
                if source == target:
                    raise ValueError(
                        f"contact family {family.label!r} collapses to onsite in this "
                        "finite supercell"
                    )
                mask[source, target] = True
                mask[target, source] = True
    return mask


def molecular_pair_interaction_matrix(
    model: MolecularPairInteractionModel,
) -> FloatArray:
    """Return the static pair interaction matrix for the molecular lattice.

    The continuum value is the baseline for all distinct sites when enabled.
    Named contact overrides *replace* that baseline on their masks and are never
    added to it, preserving the no-double-counting convention established in
    ``v0.6.0a1``.
    """
    n = model.lattice.n_sites
    interaction = np.zeros((n, n), dtype=np.float64)
    if model.continuum_coulomb:
        assert model.relative_permittivity is not None
        distance = model.lattice.minimum_image_distances_angstrom()
        offsite = distance > 0.0
        interaction[offsite] = (
            COULOMB_PREFACTOR_EV_ANGSTROM
            / model.relative_permittivity
            / distance[offsite]
        )

    used = np.zeros((n, n), dtype=bool)
    family_by_label = {family.label: family for family in model.contact_families}
    for label, value in model.contact_overrides_ev:
        mask = contact_family_mask(model.lattice, family_by_label[label])
        overlap = mask & used
        if np.any(overlap):
            raise ValueError(
                f"overlapping overridden contact families are ambiguous: {label!r}"
            )
        interaction[mask] = value
        used |= mask

    np.fill_diagonal(interaction, model.hubbard_u_ev)
    interaction.setflags(write=False)
    return interaction


def molecular_interaction_expectation(
    wavefunction: NDArray[np.floating | np.complexfloating],
    model: MolecularPairInteractionModel,
) -> float:
    """Return ``<Psi|V|Psi>`` for a normalized ordered-pair wavefunction."""
    psi = np.asarray(wavefunction)
    expected = (model.lattice.n_sites, model.lattice.n_sites)
    if psi.shape != expected:
        raise ValueError("wavefunction shape does not match molecular pair lattice")
    probability = np.square(np.abs(psi))
    return float(np.sum(probability * molecular_pair_interaction_matrix(model)))


@dataclass(frozen=True, slots=True)
class MolecularPairObservables:
    """Topology-independent and named-contact diagnostics for a pair state."""

    onsite_probability: float
    contact_probabilities: dict[str, float]
    mean_separation_angstrom: float
    rms_separation_angstrom: float
    one_body_ipr: float


def molecular_pair_observables(
    wavefunction: NDArray[np.floating | np.complexfloating],
    lattice: PeriodicMolecularLattice2D,
    contact_families: tuple[PairContactFamily, ...] = (),
) -> MolecularPairObservables:
    """Calculate physical-distance and named-contact observables."""
    psi = np.asarray(wavefunction)
    expected = (lattice.n_sites, lattice.n_sites)
    if psi.shape != expected:
        raise ValueError("wavefunction shape does not match molecular lattice")
    probability = np.square(np.abs(psi)).astype(np.float64, copy=False)
    normalization = float(np.sum(probability))
    if not np.isclose(normalization, 1.0, atol=2.0e-10):
        raise ValueError("two-particle probability is not normalized")

    onsite = float(np.trace(probability))
    distance = lattice.minimum_image_distances_angstrom()
    mean = float(np.sum(probability * distance))
    rms = float(np.sqrt(np.sum(probability * np.square(distance))))

    contacts: dict[str, float] = {}
    seen_labels: set[str] = set()
    for family in contact_families:
        if family.label in seen_labels:
            raise ValueError("observable contact-family labels must be unique")
        seen_labels.add(family.label)
        mask = contact_family_mask(lattice, family)
        contacts[family.label] = float(np.sum(probability[mask]))

    # Trace-two spin-summed site density from the normalized ordered-pair state.
    site_density = 2.0 * np.sum(probability, axis=1)
    normalized_density = site_density / 2.0
    one_body_ipr = float(np.sum(np.square(normalized_density)))

    return MolecularPairObservables(
        onsite_probability=onsite,
        contact_probabilities=contacts,
        mean_separation_angstrom=mean,
        rms_separation_angstrom=rms,
        one_body_ipr=one_body_ipr,
    )


def _legacy_shell_contact_family(dx: int, dy: int) -> PairContactFamily:
    """Translate one absolute rectangular shell into a named contact family."""
    if dx == 0 and dy == 0:
        raise ValueError("legacy onsite shell is controlled by hubbard_u")
    if dx > 0 and dy > 0:
        offsets = ((dx, dy), (dx, -dy))
    else:
        offsets = ((dx, dy),)
    return PairContactFamily(
        label=f"legacy_shell_{dx}_{dy}",
        source_basis=0,
        target_basis=0,
        cell_offsets=offsets,
    )


def legacy_rectangular_pair_model(
    parameters: BipolaronParameters,
) -> MolecularPairInteractionModel:
    """Adapt the validated rectangular interaction parameters to the G4 model."""
    if parameters.long_range_coulomb:
        assert parameters.lattice_spacing_x_angstrom is not None
        assert parameters.lattice_spacing_y_angstrom is not None
        sx = parameters.lattice_spacing_x_angstrom
        sy = parameters.lattice_spacing_y_angstrom
        epsilon = parameters.relative_permittivity
    else:
        # Physical spacings are irrelevant when the continuum is disabled.
        sx = sy = 1.0
        epsilon = None

    lattice = rectangular_legacy_lattice(parameters.nx, parameters.ny, sx, sy)
    families: list[PairContactFamily] = []
    overrides: list[tuple[str, float]] = []

    if parameters.nearest_neighbor_v != 0.0:
        family = PairContactFamily(
            label="legacy_v1_cardinal",
            source_basis=0,
            target_basis=0,
            cell_offsets=((1, 0), (0, 1)),
        )
        families.append(family)
        overrides.append((family.label, parameters.nearest_neighbor_v))

    for dx, dy, value in parameters.short_range_shell_overrides:
        family = _legacy_shell_contact_family(dx, dy)
        families.append(family)
        overrides.append((family.label, value))

    return MolecularPairInteractionModel(
        lattice=lattice,
        hubbard_u_ev=parameters.hubbard_u,
        continuum_coulomb=parameters.long_range_coulomb,
        relative_permittivity=epsilon,
        contact_families=tuple(families),
        contact_overrides_ev=tuple(overrides),
    )


def legacy_rectangular_observable_contacts() -> tuple[PairContactFamily, ...]:
    """Return named contacts corresponding to legacy x/y/diagonal observables."""
    return (
        PairContactFamily("nearest_x", 0, 0, ((1, 0),)),
        PairContactFamily("nearest_y", 0, 0, ((0, 1),)),
        PairContactFamily("diagonal", 0, 0, ((1, 1), (1, -1))),
    )
