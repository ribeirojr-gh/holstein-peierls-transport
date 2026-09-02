"""Scalable channel definitions in an ordered product basis.

The correlated bipolaron and distinguishable electron-hole exciton are stored
as ordered pair matrices rather than Slater determinants.  Physical spatial
channels such as onsite/Frenkel or nearest-neighbour charge-transfer states are
therefore most naturally represented by orthogonal projectors onto selected
canonical product-basis states ``|i,j>``.

Materializing those projectors would cost O(N^4) memory.  O0 instead stores only
the selected flattened basis indices and evaluates the mathematically identical
projector probability by summing the corresponding amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]
PairChannelKind = Literal["onsite", "nearest_x", "nearest_y", "diagonal"]


@dataclass(frozen=True, slots=True)
class ProductBasisChannel:
    """Orthogonal projector onto selected canonical states of a product basis."""

    name: str
    basis_shape: tuple[int, int]
    flat_indices: IntArray

    def __post_init__(self) -> None:
        first, second = self.basis_shape
        if first <= 0 or second <= 0:
            raise ValueError("basis_shape dimensions must be positive")
        indices = np.asarray(self.flat_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("a product-basis channel must contain at least one basis state")
        dimension = first * second
        if np.any(indices < 0) or np.any(indices >= dimension):
            raise ValueError("product-basis channel index lies outside the basis")
        unique = np.unique(indices)
        object.__setattr__(self, "flat_indices", unique)

    @property
    def dimension(self) -> int:
        return self.basis_shape[0] * self.basis_shape[1]

    @property
    def rank(self) -> int:
        return int(self.flat_indices.size)


def product_basis_channel_from_pairs(
    basis_shape: tuple[int, int],
    pairs: ArrayLike,
    *,
    name: str,
) -> ProductBasisChannel:
    """Build a channel from ordered canonical-basis index pairs ``(i,j)``."""
    first, second = basis_shape
    if first <= 0 or second <= 0:
        raise ValueError("basis_shape dimensions must be positive")
    pair_array = np.asarray(pairs, dtype=np.int64)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2 or pair_array.shape[0] == 0:
        raise ValueError("pairs must have shape (n_pairs, 2) with at least one pair")
    if np.any(pair_array[:, 0] < 0) or np.any(pair_array[:, 0] >= first):
        raise ValueError("first product-basis index lies outside the basis")
    if np.any(pair_array[:, 1] < 0) or np.any(pair_array[:, 1] >= second):
        raise ValueError("second product-basis index lies outside the basis")
    flat = pair_array[:, 0] * second + pair_array[:, 1]
    return ProductBasisChannel(
        name=name,
        basis_shape=basis_shape,
        flat_indices=np.asarray(flat, dtype=np.int64),
    )


def product_basis_channel_from_mask(
    mask: ArrayLike,
    *,
    name: str,
) -> ProductBasisChannel:
    """Build a channel from a two-dimensional boolean product-basis mask."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional product-basis array")
    flat = np.flatnonzero(array.ravel(order="C")).astype(np.int64)
    return ProductBasisChannel(
        name=name,
        basis_shape=(array.shape[0], array.shape[1]),
        flat_indices=flat,
    )


def rectangular_pair_channel(
    nx: int,
    ny: int,
    kind: PairChannelKind,
    *,
    name: str | None = None,
) -> ProductBasisChannel:
    """Return a periodic rectangular onsite/axial/diagonal ordered-pair channel.

    Distances are minimum-image distances in lattice-site units.  The channel
    includes both ordered orientations, as required by the stored bipolaron and
    distinguishable exciton wavefunctions.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    n_sites = nx * ny
    indices = np.arange(n_sites, dtype=np.int64)
    y = indices // nx
    x = indices % nx
    dx_raw = np.abs(x[:, None] - x[None, :])
    dy_raw = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx_raw, nx - dx_raw)
    dy = np.minimum(dy_raw, ny - dy_raw)

    if kind == "onsite":
        mask = (dx == 0) & (dy == 0)
    elif kind == "nearest_x":
        mask = (dx == 1) & (dy == 0)
    elif kind == "nearest_y":
        mask = (dx == 0) & (dy == 1)
    elif kind == "diagonal":
        mask = (dx == 1) & (dy == 1)
    else:
        raise ValueError(f"unknown rectangular pair channel: {kind}")

    return product_basis_channel_from_mask(
        mask,
        name=kind if name is None else name,
    )


def minimum_separation_pair_channel(
    nx: int,
    ny: int,
    minimum_distance: float,
    *,
    name: str = "separated",
) -> ProductBasisChannel:
    """Return the channel with periodic pair distance >= ``minimum_distance``."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    n_sites = nx * ny
    indices = np.arange(n_sites, dtype=np.int64)
    y = indices // nx
    x = indices % nx
    dx_raw = np.abs(x[:, None] - x[None, :])
    dy_raw = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx_raw, nx - dx_raw).astype(np.float64)
    dy = np.minimum(dy_raw, ny - dy_raw).astype(np.float64)
    distance = np.sqrt(dx * dx + dy * dy)
    return product_basis_channel_from_mask(
        distance >= minimum_distance,
        name=name,
    )


def product_basis_channel_yield(
    channel: ProductBasisChannel,
    state: ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> float:
    """Return the normalized probability of ``state`` in ``channel``.

    This evaluates ``<Psi|P_C|Psi>/<Psi|Psi>`` for the orthogonal projector
    onto the selected canonical product-basis states without constructing
    ``P_C`` explicitly.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    array = np.asarray(state, dtype=np.complex128)
    if array.shape == channel.basis_shape:
        vector = array.ravel(order="C")
    elif array.ndim == 1 and array.size == channel.dimension:
        vector = array
    else:
        raise ValueError("state shape does not match the channel product basis")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have a finite positive norm")
    selected = vector[channel.flat_indices]
    value = float(np.vdot(selected, selected).real / norm)
    if value < -tolerance or value > 1.0 + tolerance:
        raise FloatingPointError("product-basis channel yield lies outside [0,1]")
    return float(np.clip(value, 0.0, 1.0))
