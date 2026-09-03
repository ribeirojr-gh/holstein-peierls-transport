from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_dense_field_hamiltonian,
    build_sparse_field_hamiltonian,
)
from holstein_peierls.dynamics.frozen import HBAR_EV_FS
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted.isotropic import IsotropicControlParameters


def _problem(size: int = 4):
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=size, ny=size)
    lattice = LatticeState.zeros(size, size)
    return parameters, lattice


def test_legacy_style_field_helper_and_phase_rate() -> None:
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        2.0,
        0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    assert field.ex_v_per_angstrom == pytest.approx(0.002)
    assert field.ey_v_per_angstrom == pytest.approx(0.0, abs=1.0e-18)
    expected_rate = -(3.0 * 0.002) / HBAR_EV_FS
    rate_x, rate_y = field.phase_rates_per_fs()
    assert rate_x == pytest.approx(expected_rate)
    assert rate_y == pytest.approx(0.0)
    phase_x, phase_y = field.phases(100.0)
    assert phase_x == pytest.approx(100.0 * expected_rate)
    assert phase_y == pytest.approx(0.0)


def test_zero_field_reduces_exactly_to_static_hamiltonian() -> None:
    parameters, lattice = _problem()
    static = build_dense_hamiltonian(lattice, parameters)
    field = UniformElectricField2D()
    for time_fs in (0.0, 0.1, 10.0):
        dynamic = build_dense_field_hamiltonian(lattice, parameters, field, time_fs)
        assert np.array_equal(dynamic.real, static)
        assert np.count_nonzero(dynamic.imag) == 0


def test_field_hamiltonian_is_hermitian_and_t0_is_static() -> None:
    parameters, lattice = _problem()
    field = UniformElectricField2D.from_millivolt_per_angstrom(2.0, np.pi / 5.0)
    static = build_dense_hamiltonian(lattice, parameters).astype(np.complex128)
    at_zero = build_dense_field_hamiltonian(lattice, parameters, field, 0.0)
    assert np.array_equal(at_zero, static)
    at_time = build_dense_field_hamiltonian(lattice, parameters, field, 37.0)
    assert np.allclose(at_time, at_time.conj().T, rtol=0.0, atol=1.0e-15)


def test_x_field_phases_only_x_bonds() -> None:
    parameters, lattice = _problem()
    field = UniformElectricField2D(ex_v_per_angstrom=0.01, ey_v_per_angstrom=0.0)
    time_fs = 5.0
    dynamic = build_dense_field_hamiltonian(lattice, parameters, field, time_fs)
    static = build_dense_hamiltonian(lattice, parameters)
    phi_x, _ = field.phases(time_fs)

    # C-order mapping on the 4x4 cell: site 0 -> +x is 1, site 0 -> +y is 4.
    assert dynamic[0, 1] == pytest.approx(static[0, 1] * np.exp(1.0j * phi_x))
    assert dynamic[1, 0] == pytest.approx(np.conj(dynamic[0, 1]))
    assert dynamic[0, 4] == pytest.approx(static[0, 4])
    assert dynamic[4, 0] == pytest.approx(static[4, 0])


def test_field_reversal_complex_conjugates_hamiltonian() -> None:
    parameters, lattice = _problem()
    plus = UniformElectricField2D(ex_v_per_angstrom=0.013, ey_v_per_angstrom=-0.007)
    minus = UniformElectricField2D(ex_v_per_angstrom=-0.013, ey_v_per_angstrom=0.007)
    h_plus = build_dense_field_hamiltonian(lattice, parameters, plus, 2.3)
    h_minus = build_dense_field_hamiltonian(lattice, parameters, minus, 2.3)
    assert np.allclose(h_minus, h_plus.conj(), rtol=0.0, atol=1.0e-15)


def test_sparse_and_dense_field_hamiltonians_agree() -> None:
    parameters, lattice = _problem()
    field = UniformElectricField2D(ex_v_per_angstrom=0.01, ey_v_per_angstrom=0.006)
    dense = build_dense_field_hamiltonian(lattice, parameters, field, 4.2)
    sparse = build_sparse_field_hamiltonian(lattice, parameters, field, 4.2)
    assert np.allclose(sparse.toarray(), dense, rtol=0.0, atol=1.0e-15)


def test_nonzero_field_rejects_ambiguous_two_site_periodic_axis() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=2, ny=4)
    lattice = LatticeState.zeros(4, 2)
    field = UniformElectricField2D(ex_v_per_angstrom=0.01)
    with pytest.raises(ValueError, match="nx >= 3"):
        build_dense_field_hamiltonian(lattice, parameters, field, 1.0)
