import numpy as np
import pytest

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.dynamics.ensemble_preparation import (
    ensemble_velocity_kinetic_energy_eV,
    peierls_fourier_support,
    peierls_velocity_member,
)


def _params():
    return StaticPolaronParameters(nx=40, ny=40, j0x=0.1, j0y=0.1, polaron_position=821)


def test_member_energy_matches_target():
    parameters = _params()
    member = peierls_velocity_member(parameters, 3, target_energy_eV=3.0e-5)
    assert ensemble_velocity_kinetic_energy_eV(member, parameters) == pytest.approx(
        3.0e-5, rel=0.0, abs=2.0e-18
    )
    assert np.max(np.abs(member.velocity.u)) == 0.0


def test_opposite_pairs_are_bitwise_negatives():
    parameters = _params()
    for i in range(16):
        a = peierls_velocity_member(parameters, i, target_energy_eV=1.0e-5)
        b = peierls_velocity_member(parameters, i + 16, target_energy_eV=1.0e-5)
        assert a.pair_id == b.pair_id == i
        assert a.pair_sign == 1
        assert b.pair_sign == -1
        assert np.array_equal(a.velocity.u, b.velocity.u)
        assert np.array_equal(a.velocity.vx, -b.velocity.vx)
        assert np.array_equal(a.velocity.vy, -b.velocity.vy)


def test_complete_ensemble_has_zero_mean_velocity():
    parameters = _params()
    members = [peierls_velocity_member(parameters, i) for i in range(32)]
    vx = np.mean(np.stack([m.velocity.vx for m in members]), axis=0)
    vy = np.mean(np.stack([m.velocity.vy for m in members]), axis=0)
    assert np.max(np.abs(vx)) < 1.0e-20
    assert np.max(np.abs(vy)) < 1.0e-20


def test_zero_modes_are_absent_and_support_is_low_q():
    parameters = _params()
    member = peierls_velocity_member(parameters, 7, max_mode_index=4)
    assert abs(float(np.mean(member.velocity.vx))) < 1.0e-20
    assert abs(float(np.mean(member.velocity.vy))) < 1.0e-20
    support = peierls_fourier_support(member.velocity, threshold=1.0e-12)
    allowed = {1, 2, 3, 4, 36, 37, 38, 39}
    for ky, kx in support["vx_indices"]:
        assert ky == 0
        assert kx in allowed
    for ky, kx in support["vy_indices"]:
        assert kx == 0
        assert ky in allowed


def test_base_pairs_are_distinct_before_sign_pairing():
    parameters = _params()
    flattened = []
    for i in range(16):
        member = peierls_velocity_member(parameters, i)
        flattened.append(np.concatenate([member.velocity.vx.ravel(), member.velocity.vy.ravel()]))
    for i in range(16):
        for j in range(i + 1, 16):
            assert np.linalg.norm(flattened[i] - flattened[j]) > 1.0e-7


def test_candidate_energies_scale_velocity_without_changing_pattern():
    parameters = _params()
    a = peierls_velocity_member(parameters, 5, target_energy_eV=1.0e-5)
    b = peierls_velocity_member(parameters, 5, target_energy_eV=1.0e-4)
    assert np.allclose(b.velocity.vx, np.sqrt(10.0) * a.velocity.vx, rtol=1e-13, atol=1e-22)
    assert np.allclose(b.velocity.vy, np.sqrt(10.0) * a.velocity.vy, rtol=1e-13, atol=1e-22)


@pytest.mark.parametrize("bad_size", [1, 3, 31])
def test_rejects_bad_ensemble_size(bad_size):
    with pytest.raises(ValueError):
        peierls_velocity_member(_params(), 0, ensemble_size=bad_size)


def test_rejects_invalid_member_and_energy():
    parameters = _params()
    with pytest.raises(ValueError):
        peierls_velocity_member(parameters, 32)
    with pytest.raises(ValueError):
        peierls_velocity_member(parameters, 0, target_energy_eV=0.0)
    with pytest.raises(ValueError):
        peierls_velocity_member(parameters, 0, vx_energy_fraction=1.0)
