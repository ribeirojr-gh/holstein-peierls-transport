"""Canonical S0 controls for fully relaxed singlet/triplet calculations.

The spin-adapted many-electron model is developed on the deliberately simple
isotropic parameter set selected for the framework implementation.  This module
keeps three conventions explicit:

1. the neutral pi system is half filled (one electron per molecular site);
2. a HOMO->LUMO excitation therefore leaves ``N/2-1`` closed orbitals plus two
   frontier electrons for an even ``N``-site lattice;
3. small deterministic lattice seeds are used only to break the translational
   symmetry of the perfectly isotropic control.  They are not material data.

The x- and y-bond seeds are related by the square-lattice symmetry and are used
as an important regression: after relaxation their energies must agree within
numerical tolerance if both converge to symmetry-equivalent minima.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .excitation_reference import (
    StaticReferencedExcitationResult,
    relax_referenced_excitation,
)
from .spin import SpinMultiplicity


class IsotropicRelaxationSeed(str, Enum):
    """Small symmetry-breaking structural seeds for the S0 control."""

    ONSITE = "onsite"
    BOND_X = "bond_x"
    BOND_Y = "bond_y"


@dataclass(frozen=True, slots=True)
class SpinRelaxationBranchResult:
    """One spin/seed branch of the fully relaxed S0 control."""

    multiplicity: SpinMultiplicity
    seed: IsotropicRelaxationSeed
    result: StaticReferencedExcitationResult


def half_filled_n_closed(n_sites: int) -> int:
    """Return ``n_closed`` for one neutral HOMO->LUMO excitation at half filling.

    The neutral reference has one pi electron per site.  For an even number of
    sites this gives ``N/2`` doubly occupied spatial orbitals.  The excited
    state keeps the lower ``N/2-1`` orbitals closed and places two electrons in
    the two frontier orbitals, spin coupled as either a singlet or triplet.
    """
    if n_sites < 2:
        raise ValueError("at least two sites are required")
    if n_sites % 2 != 0:
        raise ValueError("the half-filled closed-shell reference requires even N")
    return n_sites // 2 - 1


def isotropic_relaxation_seed(
    parameters: StaticPolaronParameters,
    seed: IsotropicRelaxationSeed | str,
    *,
    amplitude: float = 1.0e-3,
) -> LatticeState:
    """Return a deterministic zero-background symmetry-breaking lattice seed."""
    if amplitude <= 0.0:
        raise ValueError("seed amplitude must be positive")
    try:
        seed_kind = IsotropicRelaxationSeed(seed)
    except ValueError as exc:
        raise ValueError(f"unsupported isotropic relaxation seed: {seed}") from exc

    shape = (parameters.ny, parameters.nx)
    u = np.zeros(shape, dtype=np.float64)
    vx = np.zeros(shape, dtype=np.float64)
    vy = np.zeros(shape, dtype=np.float64)
    cy = parameters.ny // 2
    cx = parameters.nx // 2

    if seed_kind is IsotropicRelaxationSeed.ONSITE:
        # A local compression is compensated over the four nearest neighbours.
        # The zero mean avoids adding a physically irrelevant uniform Holstein
        # displacement to a neutral excitation whose Delta-gamma has zero trace.
        u[cy, cx] = -amplitude
        neighbours = (
            (cy, (cx + 1) % parameters.nx),
            (cy, (cx - 1) % parameters.nx),
            ((cy + 1) % parameters.ny, cx),
            ((cy - 1) % parameters.ny, cx),
        )
        unique_neighbours = sorted(set(neighbours))
        share = amplitude / len(unique_neighbours)
        for y, x in unique_neighbours:
            u[y, x] += share
    elif seed_kind is IsotropicRelaxationSeed.BOND_X:
        # Strengthen one +x bond without introducing a net rigid translation.
        vx[cy, cx] = +0.5 * amplitude
        vx[cy, (cx + 1) % parameters.nx] = -0.5 * amplitude
    else:
        # The exact 90-degree partner of BOND_X in the isotropic square control.
        vy[cy, cx] = +0.5 * amplitude
        vy[(cy + 1) % parameters.ny, cx] = -0.5 * amplitude

    return LatticeState(u=u, vx=vx, vy=vy)


def relax_isotropic_spin_branch(
    parameters: StaticPolaronParameters,
    interaction: np.ndarray,
    *,
    multiplicity: SpinMultiplicity,
    seed: IsotropicRelaxationSeed | str,
    seed_amplitude: float = 1.0e-3,
    orbital_gradient_tolerance: float = 1.0e-8,
    orbital_max_iterations: int = 800,
    gradient_convergence_criterion: float = 1.0e-6,
) -> SpinRelaxationBranchResult:
    """Relax one half-filled singlet/triplet branch from a canonical seed."""
    if parameters.nx != parameters.ny:
        raise ValueError("the canonical isotropic relaxation benchmark requires nx=ny")
    if not np.isclose(parameters.j0x, parameters.j0y):
        raise ValueError("the canonical relaxation benchmark requires Jx=Jy")
    if not np.isclose(parameters.alpha_interx, parameters.alpha_intery):
        raise ValueError("the canonical relaxation benchmark requires alpha_x=alpha_y")

    seed_kind = IsotropicRelaxationSeed(seed)
    initial_lattice = isotropic_relaxation_seed(
        parameters,
        seed_kind,
        amplitude=seed_amplitude,
    )
    result = relax_referenced_excitation(
        parameters,
        interaction,
        n_closed=half_filled_n_closed(parameters.n_sites),
        multiplicity=multiplicity,
        initial_lattice=initial_lattice,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=orbital_max_iterations,
        gradient_convergence_criterion=gradient_convergence_criterion,
    )
    return SpinRelaxationBranchResult(
        multiplicity=multiplicity,
        seed=seed_kind,
        result=result,
    )
