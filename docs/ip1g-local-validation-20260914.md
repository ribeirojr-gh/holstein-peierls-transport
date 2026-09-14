# IP1g local validation — 2026-09-14

## Numerical closure

The controlled 40x40 single-relocation experiment completed on branch `isotropic-polaron-barrier` at commit `b6153fdd661e911b93b942499cda0e8e2eb358ff`.

- 416/416 full tests passed.
- The deterministic 5 ps trajectories for `J0y/J0x = 1.0` and `0.15` both completed.
- Static relaxations converged.
- Maximum total-energy residuals were `3.81e-7 eV` (isotropic) and `2.94e-7 eV` (anisotropic), well inside the size-aware numerical gate.
- Electronic norm and intermolecular zero-mode checks passed.
- The PBC recurrence preflight passed: the 5 ps experiment finishes well before the 40-site stationary-carrier harmonic full-wrap scale.

IP1g is therefore numerically closed.

## What the original IP1g summary does and does not show

The imposed one-site electronic relocation is a clean deterministic re-dressing impulse. It removes finite-temperature background, Langevin damping, IDC events and overlapping natural hops. It is not a natural hopping trajectory.

The original 0:500 fs summary did **not** provide a clean demonstration of retrograde propagation. The fitted backward-excess centroid moved toward increasing `s` in both conditions and the fit quality was weak (`R2 ~ 0.25` isotropic, `~0.44` anisotropic). The fitted slopes, about 2.06 and 2.37 sites/ps, also exceed the harmonic maximum group velocity (~1.844 sites/ps), proving that this centroid is not a defensible packet propagation velocity.

A second issue was identified after validation: `wake_window_metrics` forms its old "backward/forward outward flux" by summing the current over every longitudinal bin in a half-space. That quantity is a distributed-current diagnostic, **not the energy flux through a boundary**. It remains useful for provenance but must not be used as the decisive radiation-direction observable.

## Consequence

No claim of retrograde phonon group velocity is made from the original IP1g aggregate fields. The saved longitudinal-current profiles are sufficient for a better posthoc analysis without rerunning dynamics.

IP1h therefore evaluates the current through fixed carrier-centered boundaries and infers packet propagation delays by cross-correlating the outward current at successive boundaries. This gives a causal space-time propagation diagnostic whose inferred speed can be checked directly against the harmonic maximum group velocity.
