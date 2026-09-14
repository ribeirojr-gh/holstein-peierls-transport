"""Traveling-wave attribution of harmonic x-Peierls energy current.

IP1r decomposes the vx excitation on a frozen electronic surface into
laboratory +/-x traveling waves, nondirectional special qx sectors, and the
remaining bilinear cross/interference current.  The decomposition is algebraic
and reconstructs the complete harmonic x-current exactly up to roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..parameters import StaticPolaronParameters
from .ehrenfest import lattice_masses_fs
from .mode_memory import harmonic_wave_numbers

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class TravelingFields:
    plus_displacement: FloatArray
    plus_velocity: FloatArray
    minus_displacement: FloatArray
    minus_velocity: FloatArray
    special_displacement: FloatArray
    special_velocity: FloatArray


def traveling_vx_fields(
    vx: ArrayLike,
    velocity_vx: ArrayLike,
    equilibrium_vx: ArrayLike,
    parameters: StaticPolaronParameters,
) -> TravelingFields:
    """Reconstruct real +/-x traveling and special vx excitation fields."""
    coordinate = np.asarray(vx, dtype=np.float64)
    velocity = np.asarray(velocity_vx, dtype=np.float64)
    equilibrium = np.asarray(equilibrium_vx, dtype=np.float64)
    shape = (parameters.ny, parameters.nx)
    if coordinate.shape != shape or velocity.shape != shape or equilibrium.shape != shape:
        raise ValueError("vx, velocity_vx and equilibrium_vx must match lattice shape")
    if not all(np.all(np.isfinite(a)) for a in (coordinate, velocity, equilibrium)):
        raise ValueError("traveling-field inputs must be finite")

    displacement = coordinate - equilibrium
    qfield = np.fft.fft2(displacement, norm="ortho")
    vfield = np.fft.fft2(velocity, norm="ortho")
    qx = harmonic_wave_numbers(parameters.nx)
    mass_v = lattice_masses_fs(parameters)[1]
    omega = 2.0 * np.sqrt(parameters.k2 / mass_v) * np.abs(np.sin(0.5 * qx))

    qp = np.zeros_like(qfield, dtype=np.complex128)
    vp = np.zeros_like(vfield, dtype=np.complex128)
    qm = np.zeros_like(qfield, dtype=np.complex128)
    vm = np.zeros_like(vfield, dtype=np.complex128)
    qs = np.zeros_like(qfield, dtype=np.complex128)
    vs = np.zeros_like(vfield, dtype=np.complex128)

    special = (np.abs(qx) <= 1.0e-14) | (np.abs(np.abs(qx) - np.pi) <= 1.0e-14)
    qs[:, special] = qfield[:, special]
    vs[:, special] = vfield[:, special]

    positive = (qx > 1.0e-14) & (qx < np.pi - 1.0e-14)
    for ix in np.flatnonzero(positive):
        w = float(omega[ix])
        a_plus = 0.5 * (qfield[:, ix] + 1.0j * vfield[:, ix] / w)
        a_minus = 0.5 * (qfield[:, ix] - 1.0j * vfield[:, ix] / w)
        qp[:, ix] = a_plus
        vp[:, ix] = -1.0j * w * a_plus
        qm[:, ix] = a_minus
        vm[:, ix] = +1.0j * w * a_minus

        neg_ix = (-int(ix)) % parameters.nx
        for iy in range(parameters.ny):
            neg_iy = (-iy) % parameters.ny
            qp[neg_iy, neg_ix] = np.conj(a_plus[iy])
            vp[neg_iy, neg_ix] = np.conj(-1.0j * w * a_plus[iy])
            qm[neg_iy, neg_ix] = np.conj(a_minus[iy])
            vm[neg_iy, neg_ix] = np.conj(+1.0j * w * a_minus[iy])

    return TravelingFields(
        np.asarray(np.fft.ifft2(qp, norm="ortho").real, dtype=np.float64),
        np.asarray(np.fft.ifft2(vp, norm="ortho").real, dtype=np.float64),
        np.asarray(np.fft.ifft2(qm, norm="ortho").real, dtype=np.float64),
        np.asarray(np.fft.ifft2(vm, norm="ortho").real, dtype=np.float64),
        np.asarray(np.fft.ifft2(qs, norm="ortho").real, dtype=np.float64),
        np.asarray(np.fft.ifft2(vs, norm="ortho").real, dtype=np.float64),
    )


def harmonic_vx_current(
    equilibrium_vx: ArrayLike,
    displacement_vx: ArrayLike,
    velocity_vx: ArrayLike,
    k2_eV_per_A2: float,
) -> FloatArray:
    """Return jx for one vx excitation component on a common static equilibrium."""
    eq = np.asarray(equilibrium_vx, dtype=np.float64)
    disp = np.asarray(displacement_vx, dtype=np.float64)
    vel = np.asarray(velocity_vx, dtype=np.float64)
    if eq.shape != disp.shape or eq.shape != vel.shape or eq.ndim != 2:
        raise ValueError("equilibrium, displacement and velocity must share a 2D shape")
    if not all(np.all(np.isfinite(a)) for a in (eq, disp, vel)):
        raise ValueError("current inputs must be finite")
    actual = eq + disp
    strain = np.roll(actual, -1, axis=1) - actual
    return np.asarray(
        -0.5 * float(k2_eV_per_A2) * (vel + np.roll(vel, -1, axis=1)) * strain,
        dtype=np.float64,
    )


def decompose_vx_current(
    vx: ArrayLike,
    velocity_vx: ArrayLike,
    equilibrium_vx: ArrayLike,
    parameters: StaticPolaronParameters,
    *,
    carrier_dx_sites: int,
) -> dict[str, FloatArray]:
    """Return full, retrograde, co-moving, special and cross x-current fields."""
    dx = int(carrier_dx_sites)
    if dx not in (-1, 1):
        raise ValueError("carrier_dx_sites must be +/-1")
    coordinate = np.asarray(vx, dtype=np.float64)
    velocity = np.asarray(velocity_vx, dtype=np.float64)
    equilibrium = np.asarray(equilibrium_vx, dtype=np.float64)
    fields = traveling_vx_fields(coordinate, velocity, equilibrium, parameters)
    full = harmonic_vx_current(
        equilibrium,
        coordinate - equilibrium,
        velocity,
        parameters.k2,
    )
    plus = harmonic_vx_current(
        equilibrium,
        fields.plus_displacement,
        fields.plus_velocity,
        parameters.k2,
    )
    minus = harmonic_vx_current(
        equilibrium,
        fields.minus_displacement,
        fields.minus_velocity,
        parameters.k2,
    )
    special = harmonic_vx_current(
        equilibrium,
        fields.special_displacement,
        fields.special_velocity,
        parameters.k2,
    )
    if dx < 0:
        retrograde, comoving = plus, minus
    else:
        retrograde, comoving = minus, plus
    cross = full - retrograde - comoving - special
    return {
        "full": np.asarray(full, dtype=np.float64),
        "retrograde": np.asarray(retrograde, dtype=np.float64),
        "comoving": np.asarray(comoving, dtype=np.float64),
        "special": np.asarray(special, dtype=np.float64),
        "cross": np.asarray(cross, dtype=np.float64),
    }


def projection_attribution(component: ArrayLike, full: ArrayLike) -> float:
    """Return additive least-squares projection of one current vector component."""
    comp = np.asarray(component, dtype=np.float64)
    ref = np.asarray(full, dtype=np.float64)
    if comp.shape != ref.shape or comp.size == 0:
        raise ValueError("component and full arrays must have identical non-empty shape")
    if not np.all(np.isfinite(comp)) or not np.all(np.isfinite(ref)):
        raise ValueError("projection arrays must be finite")
    denominator = float(np.sum(ref * ref))
    return 0.0 if denominator <= 1.0e-30 else float(np.sum(comp * ref) / denominator)


def squared_norm_ratio(component: ArrayLike, full: ArrayLike) -> float:
    comp = np.asarray(component, dtype=np.float64)
    ref = np.asarray(full, dtype=np.float64)
    if comp.shape != ref.shape or comp.size == 0:
        raise ValueError("component and full arrays must have identical non-empty shape")
    denominator = float(np.sum(ref * ref))
    return 0.0 if denominator <= 1.0e-30 else float(np.sum(comp * comp) / denominator)


def row_cosine_similarity(component: ArrayLike, full: ArrayLike) -> FloatArray:
    comp = np.asarray(component, dtype=np.float64)
    ref = np.asarray(full, dtype=np.float64)
    if comp.shape != ref.shape or comp.ndim != 2:
        raise ValueError("component and full must have identical shape (time, vector)")
    numerator = np.sum(comp * ref, axis=1)
    denominator = np.linalg.norm(comp, axis=1) * np.linalg.norm(ref, axis=1)
    result = np.zeros(comp.shape[0], dtype=np.float64)
    valid = denominator > 1.0e-30
    result[valid] = numerator[valid] / denominator[valid]
    return result
