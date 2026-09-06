"""Adiabatic translation-path diagnostics for a one-polaron lattice state.

IP0a constructs a deliberately simple path between a relaxed lattice distortion
and an exactly translated copy of that distortion.  At every image the
one-particle electronic ground state is re-solved, while the classical lattice
coordinates are held on the linear interpolation path.

The resulting barrier is therefore an *upper-bound diagnostic path barrier*, not
a minimum-energy-path (MEP) or a material activation energy.  Its purpose is to
identify which Holstein/Peierls and elastic terms make translation expensive
before implementing a constrained/NEB calculation in IP0b.

All paths use periodic boundaries and the same C-order site mapping as the
legacy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .electronic import GroundState, SolverName, solve_ground_state
from .hamiltonian import bond_transfer_integrals
from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]
Direction = Literal["+x", "-x", "+y", "-y"]


@dataclass(frozen=True, slots=True)
class TranslationEnergyDecomposition:
    """Physical energy terms for one adiabatic path image, in eV."""

    intramolecular_elastic: float
    intermolecular_elastic_x: float
    intermolecular_elastic_y: float
    holstein_coupling: float
    bare_transfer_x: float
    bare_transfer_y: float
    peierls_coupling_x: float
    peierls_coupling_y: float

    @property
    def lattice(self) -> float:
        return float(
            self.intramolecular_elastic
            + self.intermolecular_elastic_x
            + self.intermolecular_elastic_y
        )

    @property
    def electronic(self) -> float:
        return float(
            self.holstein_coupling
            + self.bare_transfer_x
            + self.bare_transfer_y
            + self.peierls_coupling_x
            + self.peierls_coupling_y
        )

    @property
    def total(self) -> float:
        return float(self.lattice + self.electronic)


@dataclass(frozen=True, slots=True)
class TranslationPathImage:
    fraction: float
    energy: TranslationEnergyDecomposition
    electronic_eigenvalue_eV: float
    norm_error: float
    ipr: float
    participation_number: float
    source_population: float
    target_population: float
    charge_difference: float
    maximum_population: float
    maximum_population_site: int


@dataclass(frozen=True, slots=True)
class FrozenTranslationProfile:
    direction: Direction
    source_site: int
    target_site: int
    images: tuple[TranslationPathImage, ...]
    barrier_eV: float
    barrier_image_index: int
    endpoint_energy_mismatch_eV: float
    maximum_mirror_energy_mismatch_eV: float
    maximum_decomposition_error_eV: float


def translated_site_index(
    site: int,
    parameters: StaticPolaronParameters,
    direction: Direction,
) -> int:
    """Return the nearest-neighbour site index with periodic wrapping."""
    index = int(site)
    if not 0 <= index < parameters.n_sites:
        raise ValueError("site is outside the lattice")
    y, x = divmod(index, parameters.nx)
    if direction == "+x":
        x = (x + 1) % parameters.nx
    elif direction == "-x":
        x = (x - 1) % parameters.nx
    elif direction == "+y":
        y = (y + 1) % parameters.ny
    elif direction == "-y":
        y = (y - 1) % parameters.ny
    else:
        raise ValueError(f"unknown direction: {direction}")
    return int(x + parameters.nx * y)


def translate_lattice_state(state: LatticeState, direction: Direction) -> LatticeState:
    """Translate every classical field by exactly one periodic lattice site."""
    state.validate()
    if direction == "+x":
        shift = (0, 1)
    elif direction == "-x":
        shift = (0, -1)
    elif direction == "+y":
        shift = (1, 0)
    elif direction == "-y":
        shift = (-1, 0)
    else:
        raise ValueError(f"unknown direction: {direction}")
    return LatticeState(
        np.roll(state.u, shift=shift, axis=(0, 1)).copy(),
        np.roll(state.vx, shift=shift, axis=(0, 1)).copy(),
        np.roll(state.vy, shift=shift, axis=(0, 1)).copy(),
    )


def interpolate_lattice_states(
    start: LatticeState,
    end: LatticeState,
    fraction: float,
) -> LatticeState:
    """Linearly interpolate the three classical lattice-coordinate fields."""
    start.validate()
    end.validate()
    if start.shape != end.shape:
        raise ValueError("start and end lattice shapes must match")
    value = float(fraction)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return LatticeState(
        np.asarray((1.0 - value) * start.u + value * end.u, dtype=np.float64),
        np.asarray((1.0 - value) * start.vx + value * end.vx, dtype=np.float64),
        np.asarray((1.0 - value) * start.vy + value * end.vy, dtype=np.float64),
    )


def physical_energy_decomposition(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: NDArray[np.floating] | NDArray[np.complexfloating],
) -> TranslationEnergyDecomposition:
    """Decompose the physical Holstein-Peierls energy for an arbitrary state.

    The electronic state is normalized internally only for evaluating the
    expectation value; the supplied array is never modified.  The sum of all
    eight returned terms equals ``<psi|H|psi> + E_lattice``.
    """
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    psi_flat = np.asarray(electronic_state, dtype=np.complex128).reshape(-1)
    if psi_flat.size != parameters.n_sites or not np.all(np.isfinite(psi_flat)):
        raise ValueError("electronic state dimension/content is invalid")
    norm2 = float(np.vdot(psi_flat, psi_flat).real)
    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise ValueError("electronic state must have finite non-zero norm")
    psi = (psi_flat / np.sqrt(norm2)).reshape(state.shape, order="C")

    dx_v = np.roll(state.vx, -1, axis=1) - state.vx
    dy_v = np.roll(state.vy, -1, axis=0) - state.vy
    right = np.roll(psi, -1, axis=1)
    down = np.roll(psi, -1, axis=0)

    intra_elastic = 0.5 * parameters.k1 * float(np.sum(state.u * state.u))
    inter_x = 0.5 * parameters.k2 * float(np.sum(dx_v * dx_v))
    inter_y = 0.5 * parameters.k2 * float(np.sum(dy_v * dy_v))

    population = np.abs(psi) ** 2
    holstein = parameters.alpha_intra * float(np.sum(state.u * population))

    bond_x = 2.0 * np.real(np.conj(psi) * right)
    bond_y = 2.0 * np.real(np.conj(psi) * down)
    bare_x = -parameters.j0x * float(np.sum(bond_x))
    bare_y = -parameters.j0y * float(np.sum(bond_y))
    peierls_x = parameters.alpha_interx * float(np.sum(dx_v * bond_x))
    peierls_y = parameters.alpha_intery * float(np.sum(dy_v * bond_y))

    return TranslationEnergyDecomposition(
        intramolecular_elastic=float(intra_elastic),
        intermolecular_elastic_x=float(inter_x),
        intermolecular_elastic_y=float(inter_y),
        holstein_coupling=float(holstein),
        bare_transfer_x=float(bare_x),
        bare_transfer_y=float(bare_y),
        peierls_coupling_x=float(peierls_x),
        peierls_coupling_y=float(peierls_y),
    )


def _image_from_ground_state(
    fraction: float,
    state: LatticeState,
    parameters: StaticPolaronParameters,
    ground_state: GroundState,
    source_site: int,
    target_site: int,
) -> tuple[TranslationPathImage, float]:
    psi = np.asarray(ground_state.wavefunction, dtype=np.complex128)
    norm = float(np.linalg.norm(psi))
    population = np.abs(psi) ** 2 / float(np.vdot(psi, psi).real)
    ipr = float(np.sum(population * population))
    participation = float(1.0 / ipr)
    decomposition = physical_energy_decomposition(state, parameters, psi)
    decomposition_error = abs(decomposition.electronic - float(ground_state.energy))
    source_population = float(population[source_site])
    target_population = float(population[target_site])
    maximum_site = int(np.argmax(population))
    image = TranslationPathImage(
        fraction=float(fraction),
        energy=decomposition,
        electronic_eigenvalue_eV=float(ground_state.energy),
        norm_error=float(abs(norm - 1.0)),
        ipr=ipr,
        participation_number=participation,
        source_population=source_population,
        target_population=target_population,
        charge_difference=float(source_population - target_population),
        maximum_population=float(population[maximum_site]),
        maximum_population_site=maximum_site,
    )
    return image, float(decomposition_error)


def frozen_translation_profile(
    relaxed_state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    direction: Direction = "+x",
    image_count: int = 11,
    solver: SolverName = "sparse",
) -> FrozenTranslationProfile:
    """Return the adiabatic electronic energy along a frozen lattice path.

    ``relaxed_state`` is treated as the source endpoint.  The target endpoint is
    an exact one-site translation of the same lattice distortion.  Classical
    coordinates are linearly interpolated; only the electronic ground state is
    relaxed at each image.
    """
    relaxed_state.validate()
    if relaxed_state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    count = int(image_count)
    if count < 3 or count % 2 == 0:
        raise ValueError("image_count must be an odd integer >= 3")

    source = parameters.polaron_index
    target = translated_site_index(source, parameters, direction)
    translated = translate_lattice_state(relaxed_state, direction)
    fractions = np.linspace(0.0, 1.0, count)

    images: list[TranslationPathImage] = []
    max_decomposition_error = 0.0
    previous_wavefunction: FloatArray | None = None
    for fraction in fractions:
        state = interpolate_lattice_states(relaxed_state, translated, float(fraction))
        ground = solve_ground_state(
            state,
            parameters,
            solver=solver,
            initial_wavefunction=previous_wavefunction,
        )
        previous_wavefunction = np.asarray(ground.wavefunction, dtype=np.float64)
        image, error = _image_from_ground_state(
            float(fraction), state, parameters, ground, source, target
        )
        images.append(image)
        max_decomposition_error = max(max_decomposition_error, error)

    energies = np.asarray([image.energy.total for image in images], dtype=np.float64)
    endpoint_reference = 0.5 * float(energies[0] + energies[-1])
    barrier_index = int(np.argmax(energies))
    mirror_error = float(np.max(np.abs(energies - energies[::-1])))
    return FrozenTranslationProfile(
        direction=direction,
        source_site=source,
        target_site=target,
        images=tuple(images),
        barrier_eV=float(energies[barrier_index] - endpoint_reference),
        barrier_image_index=barrier_index,
        endpoint_energy_mismatch_eV=float(abs(energies[0] - energies[-1])),
        maximum_mirror_energy_mismatch_eV=mirror_error,
        maximum_decomposition_error_eV=float(max_decomposition_error),
    )
