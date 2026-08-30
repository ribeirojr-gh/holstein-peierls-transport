"""Parameters for the experimental static singlet bipolaron solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..parameters import StaticPolaronParameters


@dataclass(frozen=True, slots=True)
class BipolaronParameters:
    """Adiabatic Holstein-Peierls-Hubbard parameters for two equal carriers.

    Energies are in eV and displacements in angstrom, matching the validated
    single-polaron implementation. ``gradient_convergence_criterion`` is in
    eV/angstrom. The interaction sector contains onsite ``hubbard_u`` and an
    optional positive nearest-neighbour repulsion ``nearest_neighbor_v``.
    The Holstein-only reference solver ignores ``k2`` and the intermolecular
    coupling constants; the Peierls extension activates them.
    """

    nx: int = 20
    ny: int = 20
    k1: float = 16.51
    k2: float = 0.51
    j0x: float = 0.100
    j0y: float = 0.015
    alpha_intra: float = 3.0
    alpha_interx: float = 0.4
    alpha_intery: float = 0.4
    hubbard_u: float = 0.0
    nearest_neighbor_v: float = 0.0
    pair_position: int = 205
    max_iterations: int = 2000
    update_start: float = 1.0e-3
    update_max: float = 1.0e-2
    update_min: float = 2.0**-52
    acceleration_factor: float = 1.2
    deceleration_factor: float = 0.5
    convergence_criterion: float = 1.0e-8
    gradient_convergence_criterion: float = 1.0e-6
    eigensolver_tolerance: float = 1.0e-11
    eigensolver_max_iterations: int = 20_000

    @property
    def n_sites(self) -> int:
        return self.nx * self.ny

    @property
    def pair_index(self) -> int:
        index = self.pair_position - 1
        if not 0 <= index < self.n_sites:
            raise ValueError(
                f"pair_position={self.pair_position} is outside a "
                f"{self.nx}x{self.ny} lattice"
            )
        return index

    @property
    def atomic_holstein_pairing_scale(self) -> float:
        """Return A^2/K1, the atomic-limit Holstein pairing scale in eV."""
        return self.alpha_intra**2 / self.k1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_polaron_parameters(
        cls,
        parameters: StaticPolaronParameters,
        *,
        hubbard_u: float = 0.0,
        nearest_neighbor_v: float = 0.0,
    ) -> "BipolaronParameters":
        """Create a two-particle parameter set from a validated polaron input."""
        return cls(
            nx=parameters.nx,
            ny=parameters.ny,
            k1=parameters.k1,
            k2=parameters.k2,
            j0x=parameters.j0x,
            j0y=parameters.j0y,
            alpha_intra=parameters.alpha_intra,
            alpha_interx=parameters.alpha_interx,
            alpha_intery=parameters.alpha_intery,
            hubbard_u=hubbard_u,
            nearest_neighbor_v=nearest_neighbor_v,
            pair_position=parameters.polaron_position,
            max_iterations=parameters.max_iterations,
            update_start=parameters.update_start,
            update_max=parameters.update_max,
            update_min=parameters.update_min,
            acceleration_factor=parameters.acceleration_factor,
            deceleration_factor=parameters.deceleration_factor,
            convergence_criterion=parameters.convergence_criterion,
        )
