# IP1b local validation — dressed-polaron hopping

Date: 2026-09-06

## Status

**Numerical validation: PASS / CLOSED.**

**Physical interpretation: dressed lattice following is observed for a subset of persistent electronic residence changes, but a thermally activated dressed-polaron hopping rate is NOT established.**

The local WSL2 validation artifact reported:

- focused tests: 36/36 PASS;
- complete test suite: 380/380 PASS;
- 16/16 requested 20x20 trajectories completed;
- all nine IP1b numerical gates PASS;
- maximum generalized zero-field energy-balance residual: 2.62e-5 eV;
- maximum electronic norm error: 2.31e-13;
- maximum projected zero-mode magnitude: 2.66e-17.

The benchmark used zero field, BAOAB + CF4-Lanczos, projected intermolecular zero modes, IDC-BM with td=180 fs as a numerical control, dt=0.2 fs, 20 ps trajectories with 2 ps burn-in, four independent seeds per condition.

## Primary 50 fs persistence result

| J0y/J0x | T [K] | electronic NN | lattice NN | exact source->target match within 500 fs | matched fraction | instantaneous template/electronic-center agreement | future hop bond strongest in pre-hop window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 100 | 88 | 76 | 61 | 0.693 | 0.918 | 0.824 |
| 1.00 | 300 | 119 | 72 | 58 | 0.487 | 0.739 | 0.663 |
| 1.00 | 500 | 151 | 57 | 37 | 0.245 | 0.557 | 0.615 |
| 0.15 | 300 | 140 | 52 | 19 | 0.136 | 0.380 | 0.811 |

The 100 fs persistence control gives isotropic matched fractions within 500 fs of 0.678, 0.537, and 0.333 at 100, 300, and 500 K respectively. Thus the qualitative decline of lattice/electronic co-translation with temperature is not an artifact of the 50 fs persistence choice.

## Interpretation

IP1b confirms that some electronic residence changes are accompanied by translation of a lattice-distortion template. However, the exact discrete-event matching criterion reveals that the monotonic increase in electronic transition counts found in IP1a is not a monotonic increase in dressed-polaron hopping.

For the isotropic system the number of primary matched events in the common 72 ps aggregate observation time is 61, 58, and 37 at 100, 300, and 500 K. The corresponding exact-match dressed-event count therefore does not show simple Arrhenius activation over this range. At higher temperature the electronic center changes more frequently while the static-distortion template and electronic center agree less often.

This behavior is consistent with at least two possibilities that IP1b alone cannot distinguish:

1. a real crossover from strongly dressed, lattice-following motion at low temperature toward more weakly dressed / transiently localized electronic motion at higher temperature;
2. an overly strict discrete template-center event definition, especially when thermal distortion broadens the lattice pattern or when the lattice reorganizes continuously without producing the same discrete source->target event as the electronic tracker.

The low exact-match fraction of the anisotropic 300 K control (0.136) is a warning against interpreting the exact discrete matching fraction as a complete measure of physical transport. The anisotropic system is the established mobile control, yet its electronic and static-template centers do not often execute identical discrete transitions within the matching window. This motivates an event-conditioned continuous lattice-reorganization coordinate rather than merely relaxing the matching window.

## Transient bond anisotropy

The future electronic-hop bond is the strongest of the four local instantaneous transfer magnitudes for a large fraction of the 100 fs pre-hop samples: 0.824, 0.663, and 0.615 for the isotropic 100, 300, and 500 K conditions, compared with a symmetry/null reference of about 0.25 for four exchangeable bonds. Mean future-bond ranks are 1.318, 1.517, and 1.607 respectively.

This is strong evidence of correlation between local Peierls transfer fluctuations and the eventual direction of electronic relocation, but IP1b does not yet establish temporal causality. The structural fluctuation may precede the electronic event, or it may be part of coupled charge-lattice feedback developing very near the event.

## Decision

Do not fit an activation energy or mobility from IP1a/IP1b electronic event counts.

IP1c will use each accepted electronic nearest-neighbor event as an anchor and measure continuously around it:

- source-vs-target lattice-template correlation, rather than requiring a separately detected lattice hop;
- component-resolved lattice response from u, x-bond, and y-bond distortions;
- lag-resolved future-bond rank before the event against non-event baseline windows;
- independently integrated TP1 probability-current displacement through the event.

Only after this event-conditioned test establishes what constitutes a lattice-following translation should a dressed-hop rate versus temperature be defined.
