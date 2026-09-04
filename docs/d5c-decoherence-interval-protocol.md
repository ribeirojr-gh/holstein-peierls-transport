# D5c decoherence-interval sensitivity protocol

## Purpose

D5b validated the instantaneous-decoherence implementation at one numerical-control interval, `t_d = 100 fs`.  At that interval IDC-BM produced the closest equilibrium diagnostics, but the decoherence time is a phenomenological parameter and cannot be promoted from a single control.

D5c therefore compares the two energy-relaxing candidates, IDC-BM and IDC-MA, over

`50, 100, 180, 250, 500 fs`.

The sweep uses the same 20x20, 300 K, projected-zero-mode Holstein-Peierls control and four lattice seeds as D5a/D5b.  Each sensitivity trajectory is 6 ps with a 2 ps burn-in.  This gate is shorter than the final 10 ps validation so that several intervals can be compared at tractable cost.  A selected candidate must later return to a 10 ps closure run.

## Metrics fixed before the sweep

No weighted scalar score is used.  The following diagnostics are examined independently:

1. mean pre-collapse heating coordinate;
2. late-window pre-collapse heating coordinate;
3. pre-collapse heating-coordinate slope versus time;
4. total-variation distance of the instantaneous adiabatic populations to the canonical reference;
5. absolute mismatch between propagated and canonical ground-manifold populations;
6. expected post-collapse heating coordinate;
7. electronic-environment energy-exchange rate;
8. kinetic lattice temperature;
9. generalized energy-balance residual;
10. electronic norm error.

`beta_eff/beta_bath` is retained only as a secondary nonlinear diagnostic.  D5b showed that the mean of snapshot-wise inverse temperatures can differ substantially from one even when the ensemble mean energy is close to canonical, so it must not be used alone for scheme selection.

## Selection logic

A production candidate should satisfy all of the following qualitatively across a nontrivial interval range rather than only at one tuned point:

- heating coordinate remains close to zero after burn-in;
- no statistically clear positive late-time heating drift;
- canonical TV distance and ground-manifold mismatch remain small;
- lattice temperature remains compatible with the validated 300 K D4 bath;
- generalized energy accounting remains at the validated integration-error scale;
- electronic norm remains at roundoff scale;
- behavior is not hypersensitive to small changes in `t_d`.

If BM is strongly interval-sensitive while MA gives comparably good equilibrium diagnostics over a broader interval range, MA may be preferred despite a slightly larger error at a single `t_d`.  Conversely, if BM remains consistently closer to the canonical reference across the sweep, it becomes the stronger candidate.

D5c does not yet validate a material-specific decoherence time.  Any eventual `t_d` used for production must remain an explicit model parameter unless independently calibrated or derived.
