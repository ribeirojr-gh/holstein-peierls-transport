# IP1d local validation — matched counterfactual precursors

Date: 2026-09-07

## Status

**Numerical validation: PASS / CLOSED.**

**Physical interpretation: direction-specific structural precursor evidence is present, but the 500 fs precursor window is often not isolated from earlier electronic events. No hopping rate, activation energy, diffusion coefficient, or mobility is inferred.**

The local WSL2 validation artifact reported:

- focused test suite: PASS;
- complete test suite: 402/402 PASS;
- 16/16 requested 20x20 trajectories completed;
- all ten IP1d numerical gates PASS;
- phonon-recurrence preflight PASS;
- maximum generalized zero-field energy-balance residual: 2.65e-5 eV;
- maximum electronic norm error: 2.28e-13;
- maximum projected zero-mode magnitude: 1.33e-17;
- benchmark wall time: 1364.95 s.

The benchmark used 20x20 PBC, zero field, BAOAB + CF4-Lanczos, projected intermolecular zero modes, IDC-BM td=180 fs as a numerical control, dt=0.2 fs, 10 ps trajectories with 2 ps burn-in, four independent seed pairs per condition. The 10 ps duration was chosen after the IP1p recurrence audit.

## Phonon PBC recurrence guard

For the current intermolecular harmonic parameters and a=3 A, IP1p reports a maximum harmonic group velocity of 1.843909 sites/ps (5.531727 A/ps), giving a stationary-carrier full-wrap time of 10.846523 ps in both x and y for the 20x20 cell. The planned 10 ps run and all complete +/-500 fs event windows finish before that stationary-carrier wrap estimate.

At gamma_v=0.01 fs^-1, the harmonic intermolecular modes are overdamped in the present Langevin model. This suppresses coherent phonon memory, but gamma remains a numerical bath parameter rather than a material-calibrated phonon lifetime. A moving carrier can also change the actual re-encounter time, so the stationary-wrap estimate is a conservative diagnostic rather than a complete finite-size proof.

## Primary matched results

| J0y/J0x | T [K] | complete events | mean matched lattice advantage | positive lattice advantage | true lattice target max | current true max +/-100 fs | bond advantage -100:-80 fs [meV] | true bond top-1 -100:-80 fs | bond advantage -500:-400 fs [meV] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 100 | 38 | 0.0773 | 0.579 | 0.289 | 0.711 | 80.05 | 0.745 | 70.70 |
| 1.00 | 300 | 43 | 0.2043 | 0.744 | 0.488 | 0.814 | 149.26 | 0.752 | 140.49 |
| 1.00 | 500 | 48 | 0.1691 | 0.792 | 0.375 | 0.708 | 140.48 | 0.637 | 170.25 |
| 0.15 | 300 | 49 | 0.1496 | 0.694 | 0.408 | 0.776 | 176.23 | 0.771 | 181.66 |

The accepted electronic direction is therefore favored over the three simultaneous counterfactual neighbors in both the continuous lattice response and, more strongly, the local transfer-integral field. This removes reliance on the naive 0.25 symmetry reference used in earlier screening.

The seed-level screening intervals support a positive mean lattice matched advantage for all four conditions. For the isotropic cases the seed-level 95% screening intervals are approximately [0.0288,0.1149] at 100 K, [0.0099,0.4108] at 300 K, and [0.0788,0.2661] at 500 K. The 100 K statistic uses only three seeds with complete event windows and should be interpreted accordingly.

The -100:-80 fs matched transfer-bond advantage is positive on the pooled data at all conditions. Its seed-level interval excludes zero for isotropic 300 K and 500 K and for the anisotropic 300 K control; the 100 K interval remains broad and includes zero.

## Current and lattice continuity

The TP1 current remains strongly aligned with the accepted electronic relocation on the short +/-100 fs window. The fraction with positive parallel current is 0.921, 0.977, and 0.833 for isotropic 100, 300, and 500 K, and 0.959 for the anisotropic 300 K control. The stricter matched-current maximum fractions are lower, as expected because opposite-direction projections are algebraically related, but remain well above a random four-direction interpretation.

The continuous lattice response is not a rigid translated static template in every event. The true target has the largest Delta Q_L among the four candidate neighbors in only 0.289, 0.488, 0.375, and 0.408 of events for isotropic 100, 300, 500 K and anisotropic 300 K respectively. This reinforces the IP1b/IP1c conclusion that rigid static-template translation is too strict a definition of physical carrier motion at finite temperature.

Near-event negative-to-positive lattice sign reversal fractions are 0.579, 0.767, 0.583, and 0.102 for the same four conditions. The anisotropic mobile control again shows that absence of a binary template reversal cannot be used to reject transport.

## Important new limitation: event-window overlap

A retrospective check of the accepted complete events shows that nearest-neighbor electronic events are frequently separated by less than the 500 fs precursor window. Median within-trajectory gaps between complete events are approximately 246 fs (isotropic 100 K), 430 fs (isotropic 300 K), 458 fs (isotropic 500 K), and 374 fs (anisotropic 300 K). The fraction of within-trajectory event gaps below 500 fs is about 0.77, 0.54, 0.52, and 0.64 respectively.

Therefore, the strong matched bond advantage seen already at -500:-400 fs cannot automatically be interpreted as a single-event causal precursor. In many cases that interval may belong to the relaxation tail or lattice wake of a previous electronic relocation. This is especially relevant given the independently noted backward-emitted dynamic phonons.

## Interpretation

IP1d establishes that the direction ultimately chosen by the electronic relocation is not equivalent to the other three neighboring directions in the preceding lattice configuration. The future bond is directionally favored hundreds of femtoseconds before the registered residence transition, and the continuous lattice response shifts preferentially toward the true target on average.

However, the data do not yet separate three possibilities:

1. a genuinely isolated thermal fluctuation that prepares the next hop;
2. a persistent local anisotropic environment that determines a sequence of correlated hops;
3. the lattice/phonon wake of an earlier relocation, including possible backward-emitted vibrational energy.

The third possibility is physically important and must be distinguished from both a true precursor and any later PBC self-collision.

## Decision

IP1d is numerically closed, but kinetics remain deferred.

The next stage should use isolated-event conditioning and explicit carrier-centered lattice-wake diagnostics. Events used for causal precursor inference should require a quiet interval before the transition, while a separate analysis should retain clustered events to study memory and wake propagation. Long-time transport rates should not be extracted until finite-size and damping sensitivity are checked on larger cells and weaker, properly equilibrated baths.
