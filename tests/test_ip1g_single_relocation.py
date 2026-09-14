"""Tests for the IP1g controlled electronic-relocation helper."""

from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.single_relocation import translate_electronic_state


def test_translation_moves_delta_state_through_periodic_boundary() -> None:
    nx = ny = 6
    source_x, source_y = 5, 2
    psi = np.zeros(nx * ny, dtype=np.complex128)
    psi[source_y * nx + source_x] = 1.0
    shifted = translate_electronic_state(psi, nx, ny, dx_sites=1)
    target = source_y * nx + 0
    assert int(np.argmax(np.abs(shifted) ** 2)) == target


def test_translation_preserves_norm_and_is_invertible() -> None:
    nx, ny = 7, 5
    rng = np.random.default_rng(19)
    psi = rng.normal(size=nx * ny) + 1j * rng.normal(size=nx * ny)
    shifted = translate_electronic_state(psi, nx, ny, dx_sites=2, dy_sites=-1)
    restored = translate_electronic_state(shifted, nx, ny, dx_sites=-2, dy_sites=1)
    assert np.vdot(shifted, shifted).real == pytest.approx(np.vdot(psi, psi).real)
    assert np.allclose(restored, psi)


def test_translation_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError):
        translate_electronic_state(np.ones(8), 3, 3, dx_sites=1)
