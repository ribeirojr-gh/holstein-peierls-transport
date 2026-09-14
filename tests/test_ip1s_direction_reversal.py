import numpy as np
import pytest

from holstein_peierls.dynamics.direction_reversal import (
    direction_reversal_fourier_errors,
    reverse_non_special_vx_velocity,
    vx_special_mask,
    vx_velocity_kinetic_energy_eV,
)
from holstein_peierls.dynamics.mode_memory import traveling_vx_energy_split
from holstein_peierls.lattice import LatticeState
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.parameters import StaticPolaronParameters


def test_direction_reversal_is_involution_and_preserves_vx_kinetic_energy():
    parameters = StaticPolaronParameters(nx=6, ny=5)
    rng = np.random.default_rng(1234)
    velocity = rng.normal(size=(parameters.ny, parameters.nx)) * 1.0e-4
    reversed_velocity = reverse_non_special_vx_velocity(velocity, parameters)
    restored = reverse_non_special_vx_velocity(reversed_velocity, parameters)
    assert np.allclose(restored, velocity, rtol=0.0, atol=1.0e-15)
    assert vx_velocity_kinetic_energy_eV(reversed_velocity, parameters) == pytest.approx(
        vx_velocity_kinetic_energy_eV(velocity, parameters), rel=0.0, abs=1.0e-15
    )


def test_direction_reversal_keeps_special_and_flips_non_special_fft_sectors():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    rng = np.random.default_rng(7)
    velocity = rng.normal(size=(parameters.ny, parameters.nx)) * 2.0e-4
    reversed_velocity = reverse_non_special_vx_velocity(velocity, parameters)
    errors = direction_reversal_fourier_errors(velocity, reversed_velocity, parameters)
    assert errors["special_sector_unchanged_max_abs_A_per_fs"] < 1.0e-15
    assert errors["non_special_sector_sign_reversal_max_abs_A_per_fs"] < 1.0e-15

    special = vx_special_mask(parameters.nx)
    assert special[0]
    assert special[parameters.nx // 2]
    assert np.count_nonzero(special) == 2


def test_direction_reversal_swaps_traveling_vx_energies_at_fixed_coordinate():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    rng = np.random.default_rng(21)
    equilibrium = np.zeros((parameters.ny, parameters.nx))
    vx = rng.normal(size=equilibrium.shape) * 2.0e-3
    velocity_vx = rng.normal(size=equilibrium.shape) * 2.0e-4
    lattice = LatticeState(
        np.zeros_like(vx),
        vx.copy(),
        np.zeros_like(vx),
    )
    velocity = LatticeVelocity(
        np.zeros_like(vx),
        velocity_vx.copy(),
        np.zeros_like(vx),
    )
    before = traveling_vx_energy_split(
        lattice,
        velocity,
        LatticeState(np.zeros_like(vx), equilibrium, np.zeros_like(vx)),
        parameters,
        carrier_dx_sites=-1,
    )
    reversed_vx = reverse_non_special_vx_velocity(velocity_vx, parameters)
    after = traveling_vx_energy_split(
        lattice,
        LatticeVelocity(np.zeros_like(vx), reversed_vx, np.zeros_like(vx)),
        LatticeState(np.zeros_like(vx), equilibrium, np.zeros_like(vx)),
        parameters,
        carrier_dx_sites=-1,
    )
    assert after["retrograde_eV"] == pytest.approx(before["comoving_eV"], rel=0.0, abs=1.0e-12)
    assert after["comoving_eV"] == pytest.approx(before["retrograde_eV"], rel=0.0, abs=1.0e-12)
    assert after["nondirectional_special_eV"] == pytest.approx(
        before["nondirectional_special_eV"], rel=0.0, abs=1.0e-12
    )
