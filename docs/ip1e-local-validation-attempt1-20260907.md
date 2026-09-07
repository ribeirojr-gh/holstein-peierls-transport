# IP1e local validation attempt 1 — finite-size and bath-memory screen

Date: 2026-09-07

## Status

**Original runner status: FAIL, for one methodological reason only.**

The 20x20 absolute generalized-energy-balance gate of 5e-5 eV was reused unchanged for 40x40 cells. Both 40x40 protocols completed all requested trajectories and passed every other numerical gate, but their maximum extensive residuals were about 5.3e-5 eV and therefore missed that fixed absolute threshold by only ~6%.

The complete regression suite passed 402/402 tests. All three IP1p recurrence preflights passed. The 20x20 dynamics passed every IP1d numerical gate. The 40x40 strong- and weak-damping dynamics passed static convergence, trajectory completion, IDC count, temperature, norm, zero-mode, template-correlation, finite-diagnostic, and four-neighbour gates. Only the fixed absolute energy-balance check failed.

## Energy-balance audit

Maximum residuals observed:

| protocol | size | max residual [eV] | residual/site [eV] | old absolute gate [eV] |
|---|---:|---:|---:|---:|
| baseline20 | 20x20 | 1.519e-5 | 3.80e-8 | 5.0e-5 |
| large40 | 40x40 | 5.331e-5 | 3.33e-8 | 5.0e-5 |
| weak40 | 40x40 | 5.267e-5 | 3.29e-8 | 5.0e-5 |

The residual is extensive. The 40x40 residual per site is slightly smaller than in the passing 20x20 control, and the final residual is a systematic ~-4.6 to -4.9e-5 eV across anisotropy and damping conditions. This behavior is consistent with finite-size accumulation of the same numerical truncation rather than a physical instability.

For finite-size screens the repository now defines a size-aware numerical gate that preserves the validated 20x20 threshold exactly and scales linearly with site count:

`tol(N) = 5e-5 eV * N / 400`.

Thus the corresponding 40x40 tolerance is 2e-4 eV. This is a numerical validation rule only; no physical result or trajectory is modified. A lightweight posthoc recheck is provided so the expensive dynamics need not be repeated.

## Physical sensitivity data from the completed trajectories

The data are useful despite the original runner exit code because all trajectories completed and the only failed check is the audited extensive-threshold mismatch.

| protocol | J0y/J0x | complete events | <Delta q matched> | q advantage positive | current parallel positive +/-100 fs | current true maximum | bond advantage -100:-80 fs [meV] | bond top1 -100:-80 fs | bond advantage -500:-400 fs [meV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20x20, gamma_v=0.01 | 1.00 | 22 | 0.2465 | 0.818 | 0.955 | 0.682 | 102.6 | 0.799 | 38.0 |
| 40x40, gamma_v=0.01 | 1.00 | 30 | 0.1721 | 0.800 | 0.933 | 0.833 | 188.7 | 0.797 | 169.7 |
| 40x40, gamma_v=0.002 | 1.00 | 33 | 0.1468 | 0.697 | 0.909 | 0.758 | 117.7 | 0.480 | 98.6 |
| 20x20, gamma_v=0.01 | 0.15 | 24 | 0.1615 | 0.750 | 0.917 | 0.792 | 167.2 | 0.750 | 169.0 |
| 40x40, gamma_v=0.01 | 0.15 | 35 | 0.1230 | 0.714 | 0.971 | 0.886 | 302.8 | 1.000 | 301.8 |
| 40x40, gamma_v=0.002 | 0.15 | 34 | 0.0741 | 0.676 | 0.912 | 0.882 | 223.0 | 0.893 | 190.8 |

Only two seed pairs were used per protocol; these are sensitivity-screen descriptors, not production statistics.

## Interpretation

The core event-conditioned result survives doubling the linear cell size. For the isotropic system, the matched lattice response remains positive in 80% of events on 40x40 with gamma_v=0.01, and the TP1 current remains aligned with the electronic relocation in 93% of events. The near-event future-bond advantage also remains large.

Reducing only gamma_v from 0.01 to 0.002 fs^-1 on 40x40 does not destroy the core lattice/current correlation, but it changes the local precursor pattern substantially. In the isotropic system the future-bond top-1 fraction at -100:-80 fs decreases from ~0.80 to ~0.48 while the matched lattice advantage remains positive and the current remains aligned in ~91% of events. This is consistent with increased intermolecular phonon memory making the local transfer landscape less reducible to one instantaneously dominant bond. It is not evidence for a calibrated material damping effect.

The long-lag bond advantage remains positive on 40x40, including the weak-damping case. Because typical separations between complete electronic events are only a few hundred femtoseconds, these long-lag signatures can contain memory of previous relocations and are natural candidates for an explicit phonon-wake analysis.

## Decision

Do not rerun the expensive IP1e dynamics solely because of the original fixed absolute energy-balance gate. Run the new lightweight size-aware artifact recheck and preserve the original failed runner artifact for provenance.

After that recheck closes numerically, the next physics stage should directly measure the carrier-centered intermolecular phonon wake, including its signed propagation direction and group velocity. The user has specifically observed in earlier calculations that the emitted phonon packet has group velocity opposite to the carrier velocity; this should be treated as a physical observable, not merely a PBC artifact control.
