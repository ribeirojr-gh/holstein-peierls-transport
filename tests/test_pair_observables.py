import numpy as np

from holstein_peierls.two_particle.bipolaron import BipolaronGroundState
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _ground(psi: np.ndarray) -> BipolaronGroundState:
    psi = np.asarray(psi, dtype=float)
    psi /= np.linalg.norm(psi)
    return BipolaronGroundState(energy=0.0, wavefunction=psi)


def test_onsite_pair_has_zero_separation_and_unit_onsite_probability() -> None:
    p = BipolaronParameters(nx=4, ny=4, pair_position=6)
    psi = np.zeros((p.n_sites, p.n_sites))
    psi[5, 5] = 1.0
    obs = pair_observables(_ground(psi), p)
    assert np.isclose(obs.onsite_probability, 1.0)
    assert np.isclose(obs.nearest_neighbour_probability, 0.0)
    assert np.isclose(obs.mean_separation, 0.0)
    assert np.isclose(obs.rms_separation, 0.0)


def test_periodic_boundary_pair_is_nearest_neighbour() -> None:
    p = BipolaronParameters(nx=4, ny=4, pair_position=1)
    psi = np.zeros((p.n_sites, p.n_sites))
    left = 0
    right_across_boundary = 3
    psi[left, right_across_boundary] = 1.0 / np.sqrt(2.0)
    psi[right_across_boundary, left] = 1.0 / np.sqrt(2.0)
    obs = pair_observables(_ground(psi), p)
    assert np.isclose(obs.onsite_probability, 0.0)
    assert np.isclose(obs.nearest_neighbour_probability, 1.0)
    assert np.isclose(obs.mean_separation, 1.0)
    assert np.isclose(obs.rms_separation, 1.0)
    assert np.isclose(obs.radial_probability_by_r2[1], 1.0)


def test_pair_radial_probabilities_sum_to_one() -> None:
    p = BipolaronParameters(nx=5, ny=4, pair_position=8)
    rng = np.random.default_rng(93)
    raw = rng.normal(size=(p.n_sites, p.n_sites))
    psi = 0.5 * (raw + raw.T)
    obs = pair_observables(_ground(psi), p)
    assert np.isclose(sum(obs.radial_probability_by_r2.values()), 1.0, atol=2e-12)
    assert 1.0 / p.n_sites <= obs.one_body_ipr <= 1.0
