# IP1d — matched counterfactual precursor controls

## Motivation

IP1c numerically closed and showed three robust features around persistent nearest-neighbour electronic residence changes: (i) TP1 current is directed along the accepted relocation, (ii) the lattice source-target coordinate shifts toward the target on average, and (iii) the future transfer bond is frequently already the strongest local bond before the electronic event begins.

Two methodological limitations remain before a causal dressed-hopping interpretation is defensible.

First, the commonly quoted 0.25 reference for the strongest of four bonds is only a naive symmetry baseline.  Samples within one trajectory are correlated and the future direction is selected by the coupled dynamics.  Each real event therefore needs a matched counterfactual comparison against the three alternative nearest-neighbour targets at the same source and time.

Second, IP1c stored the first negative-to-nonnegative lattice-coordinate crossing in the full +/-500 fs window.  Thermal fluctuations can cross zero repeatedly, so that quantity is not a reliable causal lag.  IP1d uses the crossing closest to the electronic transition and reports source/target template signal strength explicitly.

## Frozen dynamics

The dynamics are intentionally unchanged from IP1c:

- one-polaron Holstein-Peierls model;
- 20x20 PBC;
- zero electric field;
- BAOAB lattice bath;
- gamma_u = gamma_v = 0.01 fs^-1;
- dt = 0.2 fs;
- CF4-Lanczos, m=6;
- projected intermolecular zero modes;
- IDC-BM, td=180 fs as a numerical control, not a material calibration;
- 2 fs diagnostic sampling;
- 50 fs electronic residence persistence;
- 20 ps total trajectory, 2 ps burn-in;
- four independent lattice/decoherence seed pairs per condition.

Conditions remain isotropic J0y/J0x=1 at 100, 300 and 500 K plus the anisotropic J0y/J0x=0.15 mobile control at 300 K.

## Matched directions

For every accepted electronic nearest-neighbour event from source i in true direction d*, define the four periodic candidate targets

j_d = i + d,  d in {+x,-x,+y,-y}.

All diagnostics are evaluated for all four candidates using the same trajectory frames.  The three directions d != d* are counterfactuals: what would the lattice/current precursor have looked like had the electron moved to another neighbour?

For any directional scalar X_d define

Delta X_match = X_d* - mean_{d != d*} X_d.

The true direction is also assigned a rank among all four candidates.  These matched quantities, rather than a frame-count null of 0.25, are the primary IP1d mechanistic controls.

## Lattice source-target response

The same static-template coordinate as IP1c is used for every candidate neighbour,

Q_L,d(t) = [A_{j_d}(t)-A_i(t)] / [|A_{j_d}(t)|+|A_i(t)|].

Pre and post means remain [-500,-100] fs and [+100,+500] fs.  For every direction record

Delta Q_L,d = <Q_L,d>_post - <Q_L,d>_pre.

Primary matched lattice diagnostics are

- true-target Delta Q_L;
- mean counterfactual Delta Q_L;
- matched advantage Delta Q_L,true - <Delta Q_L,counterfactual>;
- true-target rank among four candidate Delta Q_L values;
- fraction of events in which the true target has the largest Delta Q_L.

The absolute template signal D_d(t)=|A_i|+|A_{j_d}| is also recorded.  This distinguishes a real loss of lattice following at high temperature from a loss of static-template signal-to-background.

## Crossing lag

All negative-to-nonnegative crossings of Q_L,true are detected by linear interpolation.  The reported event lag is the crossing nearest t=0, not the first crossing in the full event window.  A near-event reversal diagnostic also uses local pre/post means away from the transition core; it remains a physical diagnostic, never a numerical gate.

## Long-lag transfer precursor

The local transfer magnitudes around the still-confirmed source site are analyzed farther back in time than IP1c.  Bins are

- [-500,-400] fs,
- [-400,-300] fs,
- [-300,-200] fs,
- [-200,-150] fs,
- [-150,-100] fs,
- [-100,-80] fs,
- [-80,-60] fs,
- [-60,-40] fs,
- [-40,-20] fs,
- [-20,0] fs.

For each bin and each frame with the electronic tracker still confirmed at the event source, record

- true future-bond magnitude;
- mean magnitude of the other three bonds;
- matched bond advantage;
- true bond rank;
- true-bond top-1 fraction;
- mean top-1 fraction of the three counterfactual directions.

A precursor is more convincing if the matched advantage becomes positive before the transition core and grows toward t=0.

## TP1 current matched control

The same PBC-safe TP1 integrated displacement is measured over +/-100, +/-250 and +/-500 fs.  The displacement vector is projected onto all four candidate directions.  The accepted direction is compared with the other three by matched advantage and rank.

This is primarily a detector/continuity control: the density relocation and bond current are not statistically independent observables.

## Statistics

Frame and event samples are temporally correlated.  IP1d therefore reports both pooled descriptive diagnostics and seed-level summaries.  Student-t intervals across the four independent seeds are screening intervals only; no precision population parameter is claimed from N=4.

No physical threshold is frozen in IP1d.  In particular, no event is labeled a `dressed hop` solely because it satisfies one arbitrary Delta Q_L or template-correlation cutoff.  The matched-null distributions will determine whether a defensible continuous score or threshold exists for a later kinetic stage.

## Numerical gates

Numerical PASS requires:

1. all static relaxations converge;
2. all requested trajectories complete;
3. exact IDC event counts;
4. lattice temperatures remain within the established stochastic tolerance;
5. generalized zero-field energy-balance residual <5e-5 eV;
6. electronic norm error <1e-10;
7. projected zero modes <1e-12;
8. static template self-correlation is unity to 1e-10;
9. all produced matched diagnostics are finite;
10. every complete event has exactly four periodic candidate neighbour directions and the true target matches the accepted electronic event.

No nonzero event count, target advantage, crossing fraction, current fraction, or bond-predictability value is a numerical PASS gate.

## Interpretation rule

IP1d may establish direction-specific structural precursor evidence, but it still does not infer an activation energy, diffusion coefficient or mobility.  Kinetics begin only after the event population is physically defined and its sensitivity to IDC/decoherence controls is checked.