# IP1f — carrier-centered phonon-wake characterization

## Physical question

The user reports a repeatable feature of earlier Holstein–Peierls calculations: when a polaron propagates, a dynamical lattice packet is emitted whose **group velocity is opposite to the carrier propagation direction**. This is a stronger statement than simply observing residual vibration behind the carrier. It implies directional transport of vibrational energy away from the moving polaron and is a candidate microscopic source of polaronic drag.

IP1f therefore promotes the phonon wake from a finite-size nuisance to a central observable. The periodic-boundary recurrence remains guarded, but the purpose of the calculation is to measure the emitted lattice energy and its direction before any recurrence can occur.

The stage asks:

1. after an accepted electronic nearest-neighbour relocation, is excess lattice energy preferentially deposited behind the carrier?
2. is the **intermolecular lattice-energy current** directed backward relative to the carrier event?
3. does the centroid of the backward excess-energy packet move toward increasingly negative carrier-centered coordinates?
4. how much of the emitted excess is carried by the Peierls mode longitudinal to the hop versus the Peierls mode transverse to it?
5. are these conclusions robust to the finite-size and bath-memory controls already motivated by IP1e?

IP1f does not infer a hopping rate, activation energy, diffusion coefficient, mobility, or material phonon lifetime.

## Why energy current is the primary directional observable

The intermolecular lattice energy is

\[
V_v = \frac{K_2}{2}\sum_{x,y}\left[(v_x(x+1,y)-v_x(x,y))^2 +(v_y(x,y+1)-v_y(x,y))^2\right].
\]

Thus the HP intermolecular sector is a set of `vx` harmonic chains propagating along x and `vy` harmonic chains propagating along y. For a harmonic nearest-neighbour chain, a consistent bond energy current is

\[
j_{i\to i+1}=-\frac{K_2}{2}(\dot v_i+\dot v_{i+1})(v_{i+1}-v_i).
\]

IP1f evaluates this expression separately as `jx` for `vx` and `jy` for `vy`. For every accepted carrier event, the coordinate system is aligned so the carrier moves in the +s direction. The projected current is

\[
j_{\parallel}=d_x j_x+d_y j_y,
\]

where `(dx,dy)` is the nearest-neighbour carrier displacement. A negative projected current is therefore retrograde relative to that event.

This energy-current diagnostic is more direct than inferring direction solely from where a broad finite-temperature energy maximum happens to appear.

## Local lattice-energy decomposition

The passive local energy diagnostic is

\[
e_u=\frac{M_u}{2}\dot u^2+\frac{K_1}{2}u^2,
\]

\[
e_{v_x}=\frac{M_v}{2}\dot v_x^2+\frac{K_2}{2}[v_x(x+1)-v_x(x)]^2,
\]

\[
e_{v_y}=\frac{M_v}{2}\dot v_y^2+\frac{K_2}{2}[v_y(y+1)-v_y(y)]^2.
\]

With the stated bond-assignment convention, summing these maps reproduces exactly the classical lattice potential plus kinetic energy used by the dynamics.

For an x hop, `e_vx` is called the longitudinal Peierls component and `e_vy` the transverse component; the assignment is reversed for a y hop.

An important model-specific point follows. The transverse Peierls degree of freedom identified as a major contributor to the isotropic IP0b collective inertia does **not** propagate backward along x during an x hop: `vy` is coupled along y. It can nevertheless absorb energy and radiate laterally, increasing the multidimensional reorganization cost. IP1f therefore separates longitudinal backward radiation from transverse/lateral excitation instead of conflating them.

## Finite-temperature background subtraction

Absolute local energy at 300 K contains both thermal background and the bound polaron deformation. It cannot be labeled uniquely as free-phonon energy.

For each accepted electronic NN event, the transition-start time is `t=0`, not the later persistence-acceptance time. The event is aligned to +s and a carrier-centered longitudinal profile is built inside a transverse corridor of ±5 sites.

The analysis window is

\[
-200\,\mathrm{fs}\le t\le 2000\,\mathrm{fs}.
\]

A pre-event baseline is the mean over

\[
-200\le t\le -80\,\mathrm{fs}.
\]

The baseline is subtracted from energy and energy-current profiles. The immediate core `|s| <= 1` is excluded from front/back wake metrics. The resulting excess is an **event-conditioned wake diagnostic**, not a unique projection onto free normal modes.

## Wake observables

For 0–500, 500–1000 and 1000–2000 fs, IP1f records:

- signed excess-energy asymmetry between `s<0` and `s>0`;
- positive excess energy behind and ahead of the carrier;
- backward outward energy current and forward outward energy current;
- a flux bias `(J_back-J_front)/(|J_back|+|J_front|)`;
- positive excess in `u`, longitudinal Peierls and transverse Peierls sectors;
- transverse fraction of the Peierls excess.

The primary sign test for the reported phenomenon is positive **backward outward** energy flux after the carrier event.

## Descriptive wake velocity

The positive excess intermolecular energy behind the carrier defines a centroid

\[
\bar s_-(t)=\frac{\sum_{s<-1}s\,[\Delta E_v(s,t)]_+}{\sum_{s<-1}[\Delta E_v(s,t)]_+}.
\]

A linear fit over 200–1800 fs gives a descriptive wake velocity. Since all carrier events point to +s by construction,

\[
v_{\rm wake}<0
\]

means retrograde propagation.

This fitted velocity must be interpreted carefully. At `gamma_v=0.01 fs^-1` the harmonic intermolecular modes are strongly overdamped, so a sharp phonon quasiparticle group velocity is not expected. The weaker `gamma_v=0.002 fs^-1` protocol is the more meaningful test of propagating phonon memory. The harmonic maximum-group-velocity value from IP1p is retained as a reference scale, not a fit constraint.

## Overlapping hopping events

IP1d showed typical inter-event gaps of only a few hundred femtoseconds, while the fastest harmonic packet moves only a few lattice sites over ~2 ps. Individual wakes can therefore overlap physically.

IP1f does not hide this fact. It records the previous/next accepted NN-event gap and identifies events isolated by more than 1 ps on both sides. The principal result is event-conditioned averaging after aligning many events by their true direction; unrelated later motion should average down, whereas a direction-locked emitted wake should survive. The isolated-event count is reported as a sensitivity diagnostic. No non-zero isolated-event count is a numerical gate.

## IDC control

For every wake event, the distance to the nearest IDC collapse is recorded. Events with no IDC within ±100 fs are labeled `IDC-clean`. Primary aggregate results include all physical events, while the clean count allows a later restricted analysis if the wake signal appears collapse-sensitive.

## Production screen

IP1f uses the already validated IP1e finite-size setting:

- 40×40 PBC;
- T = 300 K;
- zero electric field;
- dt = 0.2 fs;
- CF4-Lanczos, m=6;
- IDC-BM, td=180 fs, retained as a numerical control and not a material calibration;
- projected intermolecular zero modes;
- 2 fs diagnostic sampling;
- 50 fs persistent electronic residence criterion;
- 15 ps total trajectory;
- 5 ps burn-in;
- two lattice/decoherence seed pairs;
- isotropic `J0y/J0x=1.0` and anisotropic mobile reference `0.15`;
- `gamma_u=0.01 fs^-1`;
- intermolecular bath values `gamma_v=0.01` and `0.002 fs^-1`.

This is 8 trajectories total. The 40×40 stationary-carrier ballistic wrap estimate is ~21.7 ps, so the complete 15 ps runs remain before the first full-wrap estimate.

## Saved artifacts

The JSON and Markdown artifacts contain event-level and aggregate wake metrics. A compressed NPZ additionally stores the pooled event-conditioned spatiotemporal profiles for each `(gamma_v, anisotropy)` condition:

- relative time;
- signed carrier-centered `s` axis;
- intramolecular excess profile;
- longitudinal Peierls excess profile;
- transverse Peierls excess profile;
- total intermolecular excess profile;
- projected longitudinal energy-current excess profile.

The NPZ is intentionally retained so that the wake can later be plotted as `s-t` maps without rerunning the expensive dynamics.

## Numerical gates

Numerical PASS requires:

1. all static relaxations converge;
2. all 8 requested trajectories complete;
3. exact IDC counts;
4. temperature remains within the established stochastic tolerance;
5. the size-aware generalized energy-balance criterion passes;
6. electronic norm error remains below `1e-10`;
7. projected zero modes remain below `1e-12`;
8. the 15 ps run stays before the stationary-carrier recurrence estimate;
9. generated wake diagnostics are finite;
10. the finite-size energy tolerance recorded in the artifact matches the shared size-aware helper.

No sign or magnitude of wake asymmetry, energy flux, fitted velocity, mode fraction, event count, or isotropic/anisotropic difference is a numerical PASS gate. Zero or forward wake is a valid physical result.

## Decision rule

If the weak-damping 40×40 calculation shows a reproducible positive backward-outward energy current and/or a negative event-conditioned wake velocity, the reported retrograde phonon emission is promoted from a visual historical observation to a quantified model mechanism.

The next question would then be energetic: how much reorganization energy is radiated per carrier displacement, and does isotropy increase the longitudinal and/or transverse radiative loss sufficiently to explain the large collective inertia and suppressed transport?

If the signal disappears under weak damping or changes sign across seeds, the older observation must be treated as protocol-sensitive and the next stage should focus on mode-resolved spectral analysis before making a drag interpretation.
