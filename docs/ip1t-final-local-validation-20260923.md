# IP1t — final local validation (2026-09-23)

## Status

**Numerical validation: PASS.**

Validated local artifact:
- branch: `isotropic-polaron-barrier`
- commit: `89382384fc26c2dffc237e5f590192035bc811c4`
- full pytest: **467 passed**
- recurrence preflight: PASS
- numerical failed gates: none
- unrelated local entry: `?? git-run.sh`

## Common trajectory

The driven isotropic 40x40 trajectory reproduces:
- first natural hop: `820 -> 819 (-x)`, start **2826 fs**, accepted **2874 fs**
- first common released recrossing: `819 -> 820 (+x)`, start **3024 fs**, accepted **3072 fs**

The IP1t counterfactual is applied only after the common recrossing is accepted, so both branches begin from the same electronic/lattice state before the Peierls-direction operation.

## Intervention quality

The non-special `vx` velocity Fourier sectors are reversed while coordinates, electronic state, `u`/`vy` velocities, and special `vx` sectors remain unchanged.

Numerical errors:
- special-sector mismatch: `1.70e-21 A/fs`
- non-special sign-reversal mismatch: `3.41e-21 A/fs`
- `vx` kinetic-energy error: `4.34e-19 eV`
- total matter-energy error: **0 eV**
- retrograde/comoving exchange error: `4.34e-19 eV`

Before intervention:
- retrograde `0.0010857492934 eV`
- co-moving `0.0012717857224 eV`

After intervention:
- retrograde `0.0012717857224 eV`
- co-moving `0.0010857492934 eV`

## Post-branch numerical quality

Both zero-power branches complete 2 ps.

Native:
- external work: **0 eV**
- maximum matter-energy drift: `7.21e-9 eV`
- norm error: `7.48e-14`

Direction-reversed:
- external work: **0 eV**
- maximum matter-energy drift: `6.46e-9 eV`
- norm error: `6.24e-14`

The 6 ps planned window is safely below the harmonic PBC wrap time of **21.693 ps**.

## Preregistered commitment decision

The primary commitment window is 1 ps after the branch at 3072 fs.

Native first x event:
- `820 -> 819 (-x)`
- start **3344 fs**
- accepted **3392 fs**

Reversed first x event:
- `820 -> 819 (-x)`
- start **3404 fs**
- accepted **3452 fs**

Both branches execute the same direct return. The transition-start difference is **60 fs**, below the preregistered 100 fs threshold.

Therefore:
- `return_commitment_changed = FALSE`
- `electronic_state_diverged = TRUE`
- `directional_memory_controls_recrossing_commitment = FALSE`

The primary IP1t causal classification is formally **FAIL** and is not relaxed post hoc.

## Population divergence

Despite the negative primary commitment gate:
- maximum population L1 in the first 2 ps: **0.339238**
- first `L1 >= 0.10`: **+534 fs**
- first `L1 >= 0.25`: **+1498 fs**

Thus the intervention causes substantial later electronic-state sensitivity without changing the immediate return classification.

## Secondary event-history observation

After the common direct return:
- native branch: no additional persistent x event during the observed continuation
- reversed branch: `819 -> 820 (+x)`, start **3702 fs**, accepted **3750 fs**

This is not part of the IP1t primary gate and is hypothesis-generating only.

## Lattice-current response

The Peierls-direction reversal remains strong:
- early (0–0.5 ps) native/reversed trailing-current cosine mean: **-0.98223**
- late (>=1 ps) cosine mean: **-0.97435**

Late mean trailing amplitudes:
- native: `1.69e-6 eV/fs`
- reversed: `5.20e-6 eV/fs`

## Scientific closure

IP1t shows that reversing the direction-resolved `vx` phase-space content at fixed coordinates and fixed energy does **not** satisfy the preregistered criterion for controlling the immediate `820 -> 819` return. The return occurs in both branches with only a 60 fs timing shift.

However, the same intervention generates strong population divergence and a different later re-escape history. Therefore:
1. no validated claim that the retrograde Peierls memory controls the immediate return commitment;
2. continued evidence that the direction-resolved Peierls phase space influences longer-time electronic evolution;
3. no hopping rate, probability, mobility, activation energy, material lifetime, or transport coefficient is inferred.

## Next checkpoint

IP1u will branch only after the common native `820 -> 819` return has been accepted and will prospectively test whether Peierls direction controls the stability of the returned state, specifically the next `819 -> 820` re-escape.
