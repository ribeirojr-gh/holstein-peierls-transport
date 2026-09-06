# IP1c local validation — event-conditioned dressed-polaron translation

Date: 2026-09-06

## Status

**Numerical validation: PASS / CLOSED.**

**Physical interpretation: persistent electronic nearest-neighbour relocations carry a correctly directed probability current and are preceded by a strong direction-selective Peierls-transfer signal.  The lattice distortion moves toward the electronic target on average, but a complete source-to-target static-template reversal occurs only for a subset of events.  IP1c therefore supports a coupled thermal precursor mechanism but does not yet define a unique dressed-polaron hopping rate.**

The local WSL2 validation artifact reported:

- focused tests: 52/52 PASS;
- complete test suite: 388/388 PASS;
- 16/16 requested 20x20 trajectories completed;
- all nine IP1c numerical gates PASS;
- maximum generalized zero-field energy-balance residual: 2.62e-5 eV;
- maximum electronic norm error: 2.31e-13;
- maximum projected zero-mode magnitude: 2.66e-17;
- benchmark wall time: about 3139 s (52.3 min).

The benchmark used zero field, BAOAB + CF4-Lanczos, projected intermolecular zero modes, IDC-BM with td=180 fs as a numerical control, dt=0.2 fs, 20 ps trajectories with 2 ps burn-in, and four independent seed pairs per condition.  Git provenance in the downloaded ZIP is `unknown`, which is accepted for the current local-validation workflow.

## Aggregate event-conditioned result

The primary electronic residence persistence is 50 fs.  Event windows are +/-500 fs around the start of the persistent electronic transition.

| J0y/J0x | T [K] | e-NN events | complete windows | lattice sign reversal | mean Delta q_L | q_L increase fraction | positive parallel current, +/-100 fs | median parallel current, +/-100 fs [A] | future bond strongest at -100:-80 fs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 100 | 88 | 84 | 0.357 | 0.0859 | 0.66 | 0.952 | 0.594 | 0.849 |
| 1.00 | 300 | 119 | 110 | 0.300 | 0.1556 | 0.77 | 0.973 | 1.129 | 0.743 |
| 1.00 | 500 | 151 | 142 | 0.232 | 0.1170 | 0.70 | 0.901 | 1.259 | 0.682 |
| 0.15 | 300 | 140 | 131 | 0.160 | 0.1320 | 0.71 | 0.977 | 1.077 | 0.849 |

The exact TP1 current therefore confirms that the persistent residence-site changes are genuine probability redistribution in the accepted direction rather than merely relabeling which component happens to be largest.  At +/-100 fs the positive-parallel-current fraction is about 90--97%.  The median directed displacement is smaller than one full 3 A lattice spacing, which is expected for an extended wavefunction and overlapping/incomplete events; it must not be interpreted as a one-particle classical jump length.

The current diagnostic is a validation of the electronic event detector, not an independent causal mechanism: by continuity, a persistent density relocation must be supported by bond current.

## Continuous lattice response

For each event i -> j the source/target lattice coordinate is

Q_L(t) = [A_j(t)-A_i(t)] / [|A_j(t)|+|A_i(t)|],

using the energy-weighted static-polaron distortion template.  A strict source-to-target sign reversal requires the pre-event mean to be negative and the post-event mean positive.

Only 23--36% of isotropic events satisfy that strict reversal criterion, and the anisotropic mobile control gives only 16%.  Consequently, static-template sign reversal is too restrictive to serve as the definition of a physical hop.

A more general result is nevertheless robust: the mean target-minus-source change Delta q_L is positive in every condition, and roughly 66--77% of complete event windows have Delta q_L > 0.  Thus the lattice distortion is statistically shifted toward the eventual electronic target even when it does not become more target-like than source-like in the strict binary sense.

The u component is the largest positive mean source-to-target correlation change in all conditions.  In the isotropic 500 K run, the transverse Peierls component is larger than the longitudinal mean response, consistent with the IP0b observation that transverse intermolecular deformation contributes strongly to isotropic collective reorganization.  These component means remain mechanistic diagnostics, not a decomposition of an activation barrier.

## Transient local anisotropy precursor

The future-hop bond is already the strongest of the four local instantaneous transfer magnitudes well before the electronic transition begins.  In the earliest reported bin, -100 to -80 fs, the fractions are:

- isotropic 100 K: 0.849;
- isotropic 300 K: 0.743;
- isotropic 500 K: 0.682;
- anisotropic 300 K control: 0.849.

For the isotropic system these values are far above the naive four-exchangeable-bond reference of 0.25 and remain high throughout the final 100 fs before the event.  This is the strongest IP1c evidence for the proposed sequence

thermal lattice fluctuation -> transient local transfer anisotropy -> preferred bond -> electronic relocation.

However, the samples inside one event and multiple events within one trajectory are correlated.  The independent statistical unit is the trajectory seed, not each 2 fs frame.  With only four seeds per condition, IP1c is a mechanistic screen rather than a precision statistical estimate.

## Important limitation of the crossing-lag diagnostic

IP1c stored the **first** negative-to-nonnegative Q_L crossing anywhere in the full +/-500 fs event window.  Thermal Q_L can cross zero several times, so a negative median value of this first-crossing lag cannot by itself establish that the mechanically relevant lattice crossing precedes the electronic event.  Some first crossings occur hundreds of femtoseconds before the event and may be unrelated fluctuations.

The next stage must replace this with a nearest/sustained crossing analysis and matched null controls before making a temporal-causality claim.

## Temperature interpretation

The electronic nearest-neighbour event count increases with temperature, while the strict static-template sign-reversal fraction decreases.  This is consistent with a progressively less rigidly dressed electronic dynamics, but it is not yet a thermally activated dressed-polaron rate.

The present evidence therefore supports the following qualitative picture:

1. finite-temperature Peierls fluctuations locally break the mean x-y isotropy;
2. the bond that becomes strongest is strongly correlated with the subsequent electronic relocation direction;
3. the exact probability current follows that direction;
4. the surrounding lattice usually shifts toward the target, but often without executing a clean binary translation of the zero-temperature static distortion template;
5. increasing temperature increases electronic activity while reducing the usefulness of a rigid static-template definition of the quasiparticle center.

This is compatible with a crossover between strongly dressed hopping and transiently localized / partially undressed motion.  It does not yet establish an activation energy, diffusion coefficient, or mobility.

## Decision

IP1c is numerically closed.  Do not fit Arrhenius kinetics, diffusion, or mobility from IP1a--IP1c event counts.

IP1d will use matched within-event counterfactual neighbours and improved temporal controls.  For each real electronic event, the actual target direction will be compared with the other three nearest-neighbour directions at the same source and time.  The next validation will measure:

- true-target versus counterfactual Delta q_L;
- true-target versus counterfactual pre-event transfer advantage over a longer negative-lag range;
- nearest/sustained Q_L crossing rather than the first crossing in the full window;
- event-level template-correlation strength so that high-temperature loss of template signal can be separated from real loss of lattice following;
- TP1 current direction as an event-detector validation;
- seed-level, rather than frame-level, uncertainty summaries.

A dressed-event kinetic definition will be introduced only after these matched controls establish a physically defensible threshold or continuous score.