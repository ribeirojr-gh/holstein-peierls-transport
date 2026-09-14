# IP1s — final local validation (2026-09-14)

## Status

**Numerical validation: PASS.**

Validated local artifact:

- branch: `isotropic-polaron-barrier`
- commit: `b59212111b0ad344bccad787a2d72e5792c91396`
- full pytest: **461 passed**
- recurrence preflight: PASS
- numerical failed gates: none

Unrelated local working-tree entries in the submitted artifact were `.git-run.sh.swp` and `git-run.sh`; they are not part of the validated branch result.

## Common natural hop and intervention

The driven isotropic 40x40 trajectory freshly reproduced the first persistent natural event

- `820 -> 819` (`-x`)
- transition start: **2826 fs**
- accepted / branch time: **2874 fs**.

At the branch point the non-special `vx` velocity Fourier sectors were reversed while coordinates, electronic state, `u`/`vy` velocities and special `vx` sectors were retained.

The intervention is numerically energy preserving:

- maximum special-sector mismatch: `2.02e-21 A/fs`
- maximum non-special sign-reversal mismatch: `5.36e-21 A/fs`
- `vx` kinetic-energy error: `4.34e-19 eV`
- total matter-energy error: **0 eV**
- retrograde/comoving traveling-energy exchange error: `2.17e-19 eV`.

The direction-resolved `vx` energies are exchanged exactly to roundoff:

- before: retrograde `0.00106227978 eV`, co-moving `0.00104726689 eV`
- after: retrograde `0.00104726689 eV`, co-moving `0.00106227978 eV`.

Thus any branch difference is not attributable to an injected energy change or a changed `vx` modal-energy spectrum.

## Post-switch numerical quality

Both zero-power held-phase branches completed 5 ps.

Native released branch:

- accumulated external work: **0 eV**
- maximum matter-energy drift: `1.06757e-8 eV`
- maximum electronic norm error: `1.18e-13`
- maximum projected zero-mode mean: `9.99e-18`.

Direction-reversed branch:

- accumulated external work: **0 eV**
- maximum matter-energy drift: `8.02562e-9 eV`
- maximum electronic norm error: `2.30e-13`
- maximum projected zero-mode mean: `2.66e-17`.

All numerical gates pass the established size-aware tolerance (`2e-4 eV`).

## Preregistered primary physical decision

The IP1s preregistration defined `event_sequence_changed` only from the **first persistent x event within 2 ps** after the branch:

- one branch has an event and the other does not; or
- first-event directions differ; or
- first-event transition-start times differ by at least 100 fs.

Observed first events:

- native: `819 -> 820 (+x)`, start **3024 fs**, accepted 3072 fs;
- reversed: `819 -> 820 (+x)`, start **2988 fs**, accepted 3036 fs.

The start-time difference is only **36 fs**, so the preregistered `event_sequence_changed` gate is **FALSE**.

Electronic population divergence is nevertheless strong:

- maximum population L1 distance in first 2 ps: **0.497651**
- first `L1 >= 0.10`: **+380 fs**
- first `L1 >= 0.25`: **+994 fs**.

Therefore:

- `electronic_state_diverged = TRUE`
- `event_sequence_changed = FALSE`
- preregistered `directional_memory_changes_electronic_trajectory = FALSE`.

**The primary IP1s causal classification formally FAILS and is not relaxed post hoc.**

## Important secondary observation

The complete event lists contain a physically important secondary difference that was not part of the primary gate:

Native branch:

1. `819 -> 820 (+x)`, start 3024 fs, accepted 3072 fs;
2. `820 -> 819 (-x)`, start 3344 fs, accepted 3392 fs.

Direction-reversed branch:

1. `819 -> 820 (+x)`, start 2988 fs, accepted 3036 fs;
2. no second persistent x event during the 5 ps continuation.

Thus direction reversal does not satisfy the preregistered criterion for altering the **first** post-switch relocation, but it is associated with a large later electronic-state divergence and disappearance of the native second recrossing.

This secondary result is hypothesis-generating only. It must not be used to retroactively flip the IP1s primary decision.

## Lattice-current response

The intervention strongly reverses the early trailing-current structure:

- early native/reversed trailing-vector cosine mean: **-0.89854**.

At late times the branches remain structurally different:

- late cosine mean: **-0.24227**
- native late mean trailing amplitude: `2.47e-7 eV/fs`
- reversed late mean trailing amplitude: `1.48e-6 eV/fs`.

This confirms that the energy-preserving velocity operation substantially changes the Peierls phase-space direction as intended.

## Scientific closure

IP1s establishes two separate facts that must not be conflated:

1. **Primary preregistered result:** reversing the non-special `vx` traveling direction does **not** meet the predeclared criterion for changing the first post-switch persistent relocation.
2. **Secondary result:** the same intervention causes strong population divergence and changes the later recrossing history (native returns `-x`; reversed does not during the observed window).

Accordingly, IP1s does **not** justify a claim that the retrograde memory controls the first subsequent hop, but it also does not support the stronger statement that the wake is dynamically irrelevant.

No hopping rate, probability, mobility, activation energy, material lifetime, or transport coefficient is inferred.

## Next checkpoint

The secondary recrossing result motivates a new prospectively defined control, IP1t, branching **after the common first post-release `+x` recrossing has been accepted**. That test will ask whether reversing `vx` traveling direction at this new common state changes the immediate return/commitment dynamics, without contamination from the earlier first-event timing difference.
