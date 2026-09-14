# IP1f local validation — carrier-centered phonon wake

Date: 2026-09-14

## Numerical status

The corrected IP1f implementation was validated locally on WSL2 with Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1. Single-thread BLAS/OpenMP/MKL settings were used.

- full test suite: **413 passed**;
- 40x40 stationary-carrier recurrence preflight: PASS;
- all 8 requested 40x40 trajectories completed;
- all static relaxations converged;
- exact IDC event counts: PASS;
- lattice temperature: PASS;
- size-aware generalized energy balance: PASS;
- electronic norm: PASS;
- projected intermolecular zero modes: PASS;
- finite wake diagnostics: PASS.

The run therefore closes the *numerical implementation* of IP1f. Physical wake signatures remain diagnostic and are not numerical acceptance gates.

## Protocol

- cell: 40x40;
- T = 300 K;
- zero external field;
- dt = 0.2 fs;
- total time = 15 ps;
- burn-in = 5 ps;
- sample interval = 2 fs;
- IDC/Boltzmann interval = 180 fs;
- anisotropy ratios J0y/J0x = 1.0 and 0.15;
- intermolecular damping gamma_v = 0.01 and 0.002 fs^-1;
- two lattice seeds per condition.

Every persistent nearest-neighbour electronic relocation is rotated so that the carrier event points toward +s. Retrograde lattice-energy flow is therefore toward -s.

## Main aggregate results

| gamma_v [fs^-1] | J0y/J0x | complete events | wake asymmetry 0–500 fs | retrograde flux bias 0–500 fs | retrograde flux-positive fraction | pooled centroid slope [sites/ps] | pooled R^2 | transverse intermolecular fraction 0–500 fs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 1.00 | 23 | +0.035 | +0.054 | 0.435 | -1.692 | 0.382 | 0.519 |
| 0.010 | 0.15 | 28 | -0.027 | -0.129 | 0.393 | -1.264 | 0.138 | 0.478 |
| 0.002 | 1.00 | 29 | +0.033 | +0.171 | 0.621 | -0.590 | 0.111 | 0.474 |
| 0.002 | 0.15 | 30 | +0.039 | +0.000 | 0.600 | +2.257 | 0.488 | 0.464 |

The weakly damped isotropic condition shows the most suggestive early-time backward-energy-flow signature: the mean outward backward flux is positive and exceeds the forward outward flux during 0–500 fs, with a positive retrograde flux bias (+0.171). The corresponding anisotropic control has essentially zero mean early flux bias.

However, the centroid-based propagation result is **not robust enough to claim a backward phonon group velocity**. In the weakly damped isotropic condition the pooled centroid fit is negative, but individual-event slopes have a positive mean (+0.078 sites/ps), a positive median (+0.121 sites/ps), only 44.8% are negative, and the median event-level R^2 is only 0.143. Strong-damping centroid slopes are not interpreted as quasiparticle group velocities.

The transverse intermolecular energy fraction does not show a large isotropic enhancement in the weak-damping control (0.474 isotropic versus 0.464 anisotropic in 0–500 fs). Thus IP1f does **not** establish the proposed chain `larger transverse dressing -> larger emitted transverse wake -> larger drag`.

## Important confounders discovered by IP1f

### IDC contamination is unavoidable in this protocol

The IDC interval is 180 fs while the cleanliness criterion was ±100 fs around the hop. Since any event lies at most 90 fs from one member of a perfectly periodic 180 fs IDC sequence, the requirement is impossible by construction. Consequently **0 IDC-clean events** were obtained in all four conditions. This is a protocol-design issue, not a numerical failure.

### Long wake windows overlap other electronic relocations

No complete event had both neighbouring nearest-neighbour events farther than 1 ps. Typical nearest-neighbour event gaps are only a few hundred femtoseconds. Therefore a 2 ps post-event wake window generally contains lattice memory from additional electronic relocations.

These two facts prevent a strong causal interpretation of the current event-conditioned centroid fits.

## Scientific conclusion

IP1f numerically validates the carrier-centered local-energy and harmonic Peierls energy-current observables and finds a **suggestive early retrograde lattice-energy-current asymmetry in the weakly damped isotropic system**. It does not yet prove a well-defined backward-propagating phonon packet or a material phonon group velocity.

The next stage must remove IDC and event-overlap ambiguity before any such claim. A controlled single-relocation lattice-radiation launch is the preferred diagnostic: impose one known charge relocation on a relaxed polaron, propagate deterministic zero-field dynamics without thermostat or IDC, and measure the emitted longitudinal/transverse lattice-energy flux on a large PBC cell. This is explicitly a re-dressing/radiation impulse experiment, not a natural hopping-rate or mobility calculation.
