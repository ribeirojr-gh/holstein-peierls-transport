# IP1c — event-conditioned dressed-polaron translation

## Motivation

IP1b numerically closed but showed that requiring the independently discretized electronic and lattice-template trackers to emit the exact same source->target event is too restrictive. This is clearest in the established anisotropic mobile control, where the 50 fs exact-match fraction within 500 fs is only 0.136.

IP1c therefore uses each persistent electronic nearest-neighbour transition as the event anchor and asks continuously whether the lattice and quantum probability current support the same translation.

## Frozen dynamics

- one-polaron Holstein–Peierls model;
- 20x20 PBC;
- zero electric field;
- BAOAB lattice bath;
- gamma_u = gamma_v = 0.01 fs^-1;
- dt = 0.2 fs;
- CF4-Lanczos, m=6;
- projected intermolecular zero modes;
- IDC-BM, td=180 fs as a numerical control, not material calibration;
- 2 fs diagnostic sampling;
- 50 fs persistent electronic residence criterion;
- 20 ps total trajectory, 2 ps burn-in;
- four independent lattice/decoherence seed pairs per condition.

Conditions: isotropic J0y/J0x=1 at 100, 300 and 500 K, plus anisotropic J0y/J0x=0.15 at 300 K.

## Continuous lattice translation coordinate

The static distortion template uses the already validated energy-weighted fields

q_u = sqrt(K1) u,
q_x = sqrt(K2) Delta_x vx,
q_y = sqrt(K2) Delta_y vy.

For each sampled lattice, periodic FFT cross-correlations with every translation of the static template are retained. For an electronic event source i -> target j, let A_i(t) and A_j(t) be the total template amplitudes centered on i and j. Define

Q_L(t) = [A_j(t)-A_i(t)] / [|A_j(t)|+|A_i(t)|].

Q_L<0 means the instantaneous distortion resembles the source-centered template more strongly; Q_L>0 means target preference.

Each event is analyzed over +/-500 fs. To avoid the transition core, pre and post means use [-500,-100] fs and [+100,+500] fs. A source-to-target lattice sign reversal is Q_L,pre<0 and Q_L,post>0. This is a physical diagnostic, not a numerical gate.

Component-resolved target-minus-source changes are recorded independently for u, x-bond and y-bond features. For each hop, the Peierls response is also reorganized into longitudinal and transverse bond components.

## Independent probability-current displacement

The exact TP1 nearest-neighbour probability current is evaluated before and after every 0.2 fs coherent BAOAB propagation step and trapezoidally integrated. IDC collapses are not assigned an artificial instantaneous current displacement.

For every electronic hop, current-integrated displacement is measured over +/-100, +/-250 and +/-500 fs and projected parallel/perpendicular to the accepted hop direction. Positive parallel displacement supports the electronic relocation. No requirement that it equal exactly one lattice spacing is imposed because the wavefunction is extended and multiple events may overlap at long windows.

## Temporal precursor test

For each future hop direction, the four local transfer magnitudes around the still-confirmed source site are ranked in 20 fs bins:

- [-100,-80] fs,
- [-80,-60] fs,
- [-60,-40] fs,
- [-40,-20] fs,
- [-20,0] fs.

The principal diagnostic is the fraction of samples in which the future hop bond has rank 1. In an isotropic exchangeable four-bond reference, the symmetry baseline is approximately 0.25. A rise already at negative lag, especially well before the transition core, would support transient lattice anisotropy as a structural precursor rather than merely a simultaneous charge-lattice feedback signature.

## Numerical gates

Numerical PASS requires only:

1. all static relaxations converge;
2. all requested trajectories complete;
3. exact IDC event counts;
4. lattice temperatures remain within the established broad stochastic tolerance;
5. generalized zero-field energy-balance residual <5e-5 eV;
6. electronic norm error <1e-10;
7. projected zero modes <1e-12;
8. static template self-correlation is unity to 1e-10;
9. all event-conditioned diagnostics that are produced are finite.

No nonzero event count, sign-reversal fraction, current-displacement threshold or bond-predictability threshold is a numerical gate.

## Interpretation rule

Do not fit an activation energy, diffusion coefficient or mobility from IP1c. If the continuous lattice coordinate and current confirm a robust subset of source-to-target events, that subset can define a candidate dressed-hop population for a later kinetic stage. If they do not, the electronic residence changes remain transient-localization/flicker diagnostics rather than polaron hops.
