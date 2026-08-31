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
    eV/angstrom.

    The interaction sector contains onsite ``hubbard_u`` and an optional
    positive nearest-neighbour repulsion ``nearest_neighbor_v``. The screened
    long-range Coulomb tail is explicitly opt-in through
    ``long_range_coulomb``. When enabled, both physical lattice spacings and
    ``relative_permittivity`` must be supplied. The continuum tail is applied
    only for distinct sites; onsite ``hubbard_u`` remains an independent
    short-range parameter. A nonzero ``nearest_neighbor_v`` replaces, rather
    than adds to, the continuum value on the four cardinal nearest neighbours.

    ``short_range_shell_overrides`` generalizes that replacement rule without
    assigning material-specific values. Each entry is ``(dx, dy, value_eV)``,
    where ``dx`` and ``dy`` are absolute minimum-image lattice offsets. For
    example, ``(1, 0, Vx)`` replaces the continuum interaction on the two x
    neighbours and ``(1, 1, Vdiag)`` replaces it on the four first diagonals.
    The onsite shell ``(0, 0)`` is forbidden because ``hubbard_u`` remains the
    independent onsite interaction. These explicit shell overrides are enabled
    only together with the long-range tail in this first implementation.

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
    long_range_coulomb: bool = False
    lattice_spacing_x_angstrom: float | None = None
    lattice_spacing_y_angstrom: float | None = None
    relative_permittivity: float | None = None
    short_range_shell_overrides: tuple[tuple[int, int, float], ...] = ()
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

    def __post_init__(self) -> None:
        if self.nearest_neighbor_v < 0.0:
            raise ValueError("nearest_neighbor_v must be non-negative")

        if self.short_range_shell_overrides and not self.long_range_coulomb:
            raise ValueError(
                "short_range_shell_overrides require long_range_coulomb=True"
            )

        seen_shells: set[tuple[int, int]] = set()
        max_dx = self.nx // 2
        max_dy = self.ny // 2
        for override in self.short_range_shell_overrides:
            if len(override) != 3:
                raise ValueError(
                    "each short-range shell override must be (dx, dy, value_eV)"
                )
            dx, dy, value = override
            if dx < 0 or dy < 0:
                raise ValueError("short-range shell offsets must be non-negative")
            if dx == 0 and dy == 0:
                raise ValueError(
                    "the onsite shell (0, 0) is controlled only by hubbard_u"
                )
            if dx > max_dx or dy > max_dy:
                raise ValueError(
                    "short-range shell offset is outside the minimum-image range "
                    f"for a {self.nx}x{self.ny} lattice: ({dx}, {dy})"
                )
            if value < 0.0:
                raise ValueError("short-range shell interaction values must be non-negative")
            shell = (dx, dy)
            if shell in seen_shells:
                raise ValueError(f"duplicate short-range shell override: {shell}")
            seen_shells.add(shell)

        if self.nearest_neighbor_v != 0.0 and (
            (1, 0) in seen_shells or (0, 1) in seen_shells
        ):
            raise ValueError(
                "nearest_neighbor_v cannot be combined with explicit (1, 0) or "
                "(0, 1) shell overrides; use one representation for cardinal "
                "nearest neighbours"
            )

        if not self.long_range_coulomb:
            return

        required = {
            "lattice_spacing_x_angstrom": self.lattice_spacing_x_angstrom,
            "lattice_spacing_y_angstrom": self.lattice_spacing_y_angstrom,
            "relative_permittivity": self.relative_permittivity,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            names = ", ".join(missing)
            raise ValueError(
                "long_range_coulomb requires explicit positive values for " + names
            )
        nonpositive = [
            name for name, value in required.items() if value is not None and value <= 0.0
        ]
        if nonpositive:
            names = ", ".join(nonpositive)
            raise ValueError("long-range Coulomb parameters must be positive: " + names)

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
        long_range_coulomb: bool = False,
        lattice_spacing_x_angstrom: float | None = None,
        lattice_spacing_y_angstrom: float | None = None,
        relative_permittivity: float | None = None,
        short_range_shell_overrides: tuple[tuple[int, int, float], ...] = (),
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
            long_range_coulomb=long_range_coulomb,
            lattice_spacing_x_angstrom=lattice_spacing_x_angstrom,
            lattice_spacing_y_angstrom=lattice_spacing_y_angstrom,
            relative_permittivity=relative_permittivity,
            short_range_shell_overrides=short_range_shell_overrides,
            pair_position=parameters.polaron_position,
            max_iterations=parameters.max_iterations,
            update_start=parameters.update_start,
            update_max=parameters.update_max,
            update_min=parameters.update_min,
            acceleration_factor=parameters.acceleration_factor,
            deceleration_factor=parameters.deceleration_factor,
            convergence_criterion=parameters.convergence_criterion,
        )
