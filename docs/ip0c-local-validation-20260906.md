# IP0c local validation — failed geometric reaction coordinate

Date: 2026-09-06

## Status

**Execution/regression tests: PASS. IP0c benchmark: FAIL. IP0c physical path definition: SUPERSEDED.**

The local ZIP `ip0c-local-validation.zip` contains run `20260906T101927Z` from Python 3.12.3 / NumPy 2.5.2 / SciPy 1.18.1 on WSL2 with `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Git provenance is `unknown`, as expected for the downloaded ZIP workflow.

- focused tests: PASS (25 tests);
- complete suite: PASS (**357 passed**);
- benchmark return code: 1;
- static relaxations: PASS;
- endpoint translational energy invariance: PASS, max mismatch `7.22e-16 eV`;
- fixed geometric reaction-coordinate preservation: PASS, max error `2.55e-15`;
- relaxation never above the frozen image: PASS, max excess `3.33e-16 eV`;
- finite/non-negative reported relaxed barriers: PASS;
- all constrained images converged: **FAIL**;
- projected-gradient convergence: **FAIL**, maximum `1.456e-2 eV/A`.

The non-converged images included `J0y/J0x=0.50,+x,s=2/3`, `0.50,+y,s=1/2`, and `1.00,+y,s=5/6`; several additional images reached the optimizer iteration limit even when their final projected gradient satisfied the looser per-image diagnostic.

## Raw relaxed-path numbers

| J0y/J0x | dir | frozen barrier (meV) | geometric-relaxed barrier (meV) |
|---:|:---:|---:|---:|
| 0.15 | +x | 13.046 | 4.261 |
| 0.15 | +y | 101.522 | 0.000023 |
| 0.50 | +x | 11.311 | 1.704 |
| 0.50 | +y | 51.242 | 2.098 |
| 1.00 | +x | 12.014 | 3.638 |
| 1.00 | +y | 12.014 | 3.638 |

These reduced values are **not accepted as physical translation barriers**.

## Why this is more than an optimizer-tolerance problem

IP0c fixed only the global projection of the lattice coordinates onto the endpoint displacement vector. Orthogonal relaxation can satisfy that scalar geometric constraint while the electronic polaron remains in the original localization basin. The charge diagnostics expose this explicitly.

For the strongly anisotropic `+y` path (`J0y/J0x=0.15`), the geometrically constrained midpoint `s=0.5` has essentially the endpoint energy, but the charge is still on the source site: source population `0.47256`, target population `0.00142`. Only between `s=0.5` and `s=2/3` does the electronic localization switch to the translated site. The almost-zero `+y` barrier is therefore a bypass in the chosen coordinate, not evidence for barrierless physical hopping.

The same qualitative issue appears in other cases: away from the central image the optimizer frequently returns a state almost indistinguishable in energy and localization from one endpoint while satisfying the global lattice projection. Increasing `max_iterations` would not repair this conceptual decoupling.

At the isotropic midpoint, by contrast, the charge is genuinely shared (`~0.36154/0.36154`) and the geometric-relaxed profile gives `3.638 meV`. This is interesting but cannot validate the coordinate by itself.

## Decision

IP0c is preserved as a negative methodological result and is **not rerun** with merely looser tolerances or more iterations.

The next static diagnostic, IP0d, replaces the geometric translation coordinate by an explicit electronic charge-transfer order parameter. The operator is built from the difference between the relaxed endpoint charge density and its one-site-translated counterpart. At each frozen lattice image, a Lagrange-multiplier bias is solved so that the complete charge cloud follows a prescribed endpoint-to-endpoint progression. Physical energies are always reported with the bias removed.

This is intended to answer a narrower, well-posed question before any further lattice relaxation: what energy is required to move the **charge localization itself**, rather than merely to move a global projection of the lattice distortion?

No IP0c value is an MEP barrier, activation energy, hopping rate, or mobility.