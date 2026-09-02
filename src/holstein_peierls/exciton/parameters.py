"""Parameters for the generic static electron-hole exciton model.

The exciton layer deliberately keeps electron and hole one-particle parameters
independent.  The default/reference constructor can copy the validated polaron
parameter set to both carriers, but equality is not built into the solver.
Energies are in eV and displacements in angstrom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..parameters import StaticPolaronParameters


@dataclass(frozen=True, slots=True)
class ExcitonParameters:
    """Adiabatic Holstein-Peierls parameters for one electron and one hole.

    ``onsite_attraction`` and ``nearest_neighbor_attraction`` are positive
    magnitudes; the electronic Hamiltonian applies them with a minus sign.
    ``long_range_coulomb`` likewise represents an attractive ``-1/r`` tail.

    Electron and hole hopping/e-ph parameters are independent by design.  The
    reference control model uses the same validated values for both carriers,
    but later studies may change them without modifying the solver.
    """

    nx: int = 20
    ny: int = 20
    k1: float = 16.51
    k2: float = 0.51

    electron_j0x: float = 0.100
    electron_j0y: float = 0.015
    hole_j0x: float = 0.100
    hole_j0y: float = 0.015

    electron_alpha_intra: float = 3.0
    electron_alpha_interx: float = 0.4
    electron_alpha_intery: float = 0.4
    hole_alpha_intra: float = 3.0
    hole_alpha_interx: float = 0.4
    hole_alpha_intery: float = 0.4

    onsite_attraction: float = 0.0
    nearest_neighbor_attraction: float = 0.0
    long_range_coulomb: bool = False
    lattice_spacing_x_angstrom: float | None = None
    lattice_spacing_y_angstrom: float | None = None
    relative_permittivity: float | None = None
    short_range_shell_attractions: tuple[tuple[int, int, float], ...] = ()

    exciton_position: int = 205
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
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if self.k1 <= 0.0 or self.k2 <= 0.0:
            raise ValueError("elastic force constants must be positive")
        if self.onsite_attraction < 0.0:
            raise ValueError("onsite_attraction is a non-negative magnitude")
        if self.nearest_neighbor_attraction < 0.0:
            raise ValueError(
                "nearest_neighbor_attraction is a non-negative magnitude"
            )

        if self.short_range_shell_attractions and not self.long_range_coulomb:
            raise ValueError(
                "short_range_shell_attractions require long_range_coulomb=True"
            )

        seen: set[tuple[int, int]] = set()
        max_dx = self.nx // 2
        max_dy = self.ny // 2
        for override in self.short_range_shell_attractions:
            if len(override) != 3:
                raise ValueError(
                    "each shell attraction must be (dx, dy, magnitude_eV)"
                )
            dx, dy, magnitude = override
            if dx < 0 or dy < 0:
                raise ValueError("shell offsets must be non-negative")
            if dx == 0 and dy == 0:
                raise ValueError(
                    "the onsite shell is controlled only by onsite_attraction"
                )
            if dx > max_dx or dy > max_dy:
                raise ValueError("shell offset is outside the minimum-image range")
            if magnitude < 0.0:
                raise ValueError("shell attraction magnitude must be non-negative")
            shell = (dx, dy)
            if shell in seen:
                raise ValueError(f"duplicate short-range shell attraction: {shell}")
            seen.add(shell)

        if self.nearest_neighbor_attraction != 0.0 and (
            (1, 0) in seen or (0, 1) in seen
        ):
            raise ValueError(
                "nearest_neighbor_attraction cannot coexist with explicit cardinal "
                "shell attractions"
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
            raise ValueError(
                "long_range_coulomb requires explicit positive values for "
                + ", ".join(missing)
            )
        nonpositive = [
            name for name, value in required.items() if value is not None and value <= 0.0
        ]
        if nonpositive:
            raise ValueError(
                "long-range Coulomb parameters must be positive: "
                + ", ".join(nonpositive)
            )

    @property
    def n_sites(self) -> int:
        return self.nx * self.ny

    @property
    def exciton_index(self) -> int:
        index = self.exciton_position - 1
        if not 0 <= index < self.n_sites:
            raise ValueError(
                f"exciton_position={self.exciton_position} is outside a "
                f"{self.nx}x{self.ny} lattice"
            )
        return index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_reference_polaron(
        cls,
        parameters: StaticPolaronParameters,
        *,
        onsite_attraction: float = 0.0,
        nearest_neighbor_attraction: float = 0.0,
        long_range_coulomb: bool = False,
        lattice_spacing_x_angstrom: float | None = None,
        lattice_spacing_y_angstrom: float | None = None,
        relative_permittivity: float | None = None,
        short_range_shell_attractions: tuple[tuple[int, int, float], ...] = (),
    ) -> "ExcitonParameters":
        """Build the equal-carrier control model from validated polaron values."""
        return cls(
            nx=parameters.nx,
            ny=parameters.ny,
            k1=parameters.k1,
            k2=parameters.k2,
            electron_j0x=parameters.j0x,
            electron_j0y=parameters.j0y,
            hole_j0x=parameters.j0x,
            hole_j0y=parameters.j0y,
            electron_alpha_intra=parameters.alpha_intra,
            electron_alpha_interx=parameters.alpha_interx,
            electron_alpha_intery=parameters.alpha_intery,
            hole_alpha_intra=parameters.alpha_intra,
            hole_alpha_interx=parameters.alpha_interx,
            hole_alpha_intery=parameters.alpha_intery,
            onsite_attraction=onsite_attraction,
            nearest_neighbor_attraction=nearest_neighbor_attraction,
            long_range_coulomb=long_range_coulomb,
            lattice_spacing_x_angstrom=lattice_spacing_x_angstrom,
            lattice_spacing_y_angstrom=lattice_spacing_y_angstrom,
            relative_permittivity=relative_permittivity,
            short_range_shell_attractions=short_range_shell_attractions,
            exciton_position=parameters.polaron_position,
            max_iterations=parameters.max_iterations,
            update_start=parameters.update_start,
            update_max=parameters.update_max,
            update_min=parameters.update_min,
            acceleration_factor=parameters.acceleration_factor,
            deceleration_factor=parameters.deceleration_factor,
            convergence_criterion=parameters.convergence_criterion,
        )
