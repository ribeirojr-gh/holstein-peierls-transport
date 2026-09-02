"""Projection observables for stationary and dynamical carrier models.

O0 keeps diagnostics independent of the time propagator. Occupation numbers
are defined from the one-particle reduced density matrix, while reaction/state
yields are defined as many-electron projection probabilities.
"""

from .occupations import instantaneous_occupations, occupations_from_orbitals
from .yields import (
    channel_yield,
    configuration_expansion_overlap,
    configuration_expansion_norm,
    slater_gram_matrix,
    slater_overlap,
    slater_yield,
)

__all__ = [
    "channel_yield",
    "configuration_expansion_norm",
    "configuration_expansion_overlap",
    "instantaneous_occupations",
    "occupations_from_orbitals",
    "slater_gram_matrix",
    "slater_overlap",
    "slater_yield",
]
