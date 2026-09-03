"""Uniform electric-field coupling for frozen-lattice D1 dynamics.

The archived Holstein-Peierls dynamics introduces a static electric field through
an explicitly time-dependent vector potential,

    A(t) = -c E (t - t0),

so a forward hopping over lattice spacing ``a`` acquires the Peierls phase

    exp(i phi),   phi(t) = -a E (t - t0) / hbar.

Here energies are in eV, time in fs, lattice spacings in angstrom, and electric
fields in V/angstrom.  In these units the elementary-charge conversion is
implicit because one electron crossing one volt changes its energy by one eV.
The reverse hopping always receives the complex-conjugate phase, preserving
Hermiticity exactly up to floating-point roundoff.

D1 deliberately keeps the classical lattice frozen.  This module changes only
the electronic hopping phases and leaves all validated static Hamiltonian
builders untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ..hamiltonian import bond_transfer_integrals, build_dense_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .frozen import HBAR_EV_FS

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class UniformElectricField2D:
    """Static in-plane electric field represented in the vector-potential gauge.

    Parameters
    ----------
    ex_v_per_angstrom, ey_v_per_angstrom
        Cartesian electric-field components in V/angstrom.
    ax_angstrom, ay_angstrom
        Molecular-lattice spacings associated with one +x/+y electronic hop.
    time_origin_fs
        Time at which the vector potential, and therefore the Peierls phase,
        is defined to vanish.
    """

    ex_v_per_angstrom: float = 0.0
    ey_v_per_angstrom: float = 0.0
    ax_angstrom: float = 3.0
    ay_angstrom: float = 3.0
    time_origin_fs: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.ex_v_per_angstrom,
            self.ey_v_per_angstrom,
            self.ax_angstrom,
            self.ay_angstrom,
            self.time_origin_fs,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("field parameters must be finite")
        if self.ax_angstrom <= 0.0 or self.ay_angstrom <= 0.0:
            raise ValueError("lattice spacings must be positive")

    @classmethod
    def from_millivolt_per_angstrom(
        cls,
        magnitude_mv_per_angstrom: float,
        angle_radians: float = 0.0,
        *,
        ax_angstrom: float = 3.0,
        ay_angstrom: float = 3.0,
        time_origin_fs: float = 0.0,
    ) -> "UniformElectricField2D":
        """Construct a field from the legacy-style magnitude and in-plane angle."""
        magnitude = float(magnitude_mv_per_angstrom)
        angle = float(angle_radians)
        if not np.isfinite(magnitude) or not np.isfinite(angle):
            raise ValueError("field magnitude and angle must be finite")
        magnitude_v = 1.0e-3 * magnitude
        return cls(
            ex_v_per_angstrom=magnitude_v * float(np.cos(angle)),
            ey_v_per_angstrom=magnitude_v * float(np.sin(angle)),
            ax_angstrom=ax_angstrom,
            ay_angstrom=ay_angstrom,
            time_origin_fs=time_origin_fs,
        )

    @property
    def is_zero(self) -> bool:
        return self.ex_v_per_angstrom == 0.0 and self.ey_v_per_angstrom == 0.0

    def phases(self, time_fs: float) -> tuple[float, float]:
        """Return the dimensionless Peierls phases for forward +x/+y hopping."""
        time = float(time_fs)
        if not np.isfinite(time):
            raise ValueError("time_fs must be finite")
        elapsed = time - self.time_origin_fs
        phi_x = -(self.ax_angstrom * self.ex_v_per_angstrom / HBAR_EV_FS) * elapsed
        phi_y = -(self.ay_angstrom * self.ey_v_per_angstrom / HBAR_EV_FS) * elapsed
        return float(phi_x), float(phi_y)

    def phase_rates_per_fs(self) -> tuple[float, float]:
        """Return ``d phi_x/dt`` and ``d phi_y/dt`` in rad/fs."""
        return (
            float(-self.ax_angstrom * self.ex_v_per_angstrom / HBAR_EV_FS),
            float(-self.ay_angstrom * self.ey_v_per_angstrom / HBAR_EV_FS),
        )


def _site_neighbours(
    parameters: StaticPolaronParameters,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    right = np.roll(sites, shift=-1, axis=1)
    down = np.roll(sites, shift=-1, axis=0)
    return sites.ravel(), right.ravel(), down.ravel()


def _validate_field_cell(
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
) -> None:
    """Reject ambiguous two-site periodic multibonds for a nonzero phased axis."""
    if field.ex_v_per_angstrom != 0.0 and parameters.nx < 3:
        raise ValueError("nonzero x field requires nx >= 3 under periodic boundaries")
    if field.ey_v_per_angstrom != 0.0 and parameters.ny < 3:
        raise ValueError("nonzero y field requires ny >= 3 under periodic boundaries")


def build_dense_field_hamiltonian(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    time_fs: float,
) -> ComplexArray:
    """Construct the frozen-lattice Hermitian Hamiltonian ``H(t)`` with field.

    At zero field this is exactly the validated static dense Hamiltonian cast to
    complex dtype.  For a nonzero field, forward +x/+y hoppings receive
    ``exp(i phi_x/y)`` and reverse hoppings their complex conjugates.
    """
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    _validate_field_cell(parameters, field)

    if field.is_zero:
        return np.asarray(build_dense_hamiltonian(state, parameters), dtype=np.complex128)

    n = parameters.n_sites
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    diagonal = parameters.alpha_intra * state.u.reshape(-1, order="C")
    np.fill_diagonal(hamiltonian, diagonal)

    tx, ty = bond_transfer_integrals(state, parameters)
    sites, right, down = _site_neighbours(parameters)
    phi_x, phi_y = field.phases(time_fs)
    phase_x = np.exp(1.0j * phi_x)
    phase_y = np.exp(1.0j * phi_y)
    tx_forward = tx.ravel(order="C") * phase_x
    ty_forward = ty.ravel(order="C") * phase_y

    hamiltonian[sites, right] = tx_forward
    hamiltonian[right, sites] = np.conj(tx_forward)
    hamiltonian[sites, down] = ty_forward
    hamiltonian[down, sites] = np.conj(ty_forward)
    return hamiltonian


def build_sparse_field_hamiltonian(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    time_fs: float,
) -> csr_matrix:
    """Construct the D1 field Hamiltonian directly in sparse CSR form."""
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    _validate_field_cell(parameters, field)

    if field.is_zero or parameters.nx < 3 or parameters.ny < 3:
        return csr_matrix(build_dense_field_hamiltonian(state, parameters, field, time_fs))

    sites, right, down = _site_neighbours(parameters)
    tx, ty = bond_transfer_integrals(state, parameters)
    diagonal = parameters.alpha_intra * state.u.ravel(order="C")
    phi_x, phi_y = field.phases(time_fs)
    tx_forward = tx.ravel(order="C") * np.exp(1.0j * phi_x)
    ty_forward = ty.ravel(order="C") * np.exp(1.0j * phi_y)

    rows = np.concatenate((sites, sites, right, sites, down))
    cols = np.concatenate((sites, right, sites, down, sites))
    data = np.concatenate(
        (
            diagonal.astype(np.complex128),
            tx_forward,
            np.conj(tx_forward),
            ty_forward,
            np.conj(ty_forward),
        )
    )
    return coo_matrix(
        (data, (rows, cols)),
        shape=(parameters.n_sites, parameters.n_sites),
        dtype=np.complex128,
    ).tocsr()
