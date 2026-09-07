# IP1e — finite-size and intermolecular bath-memory screening

## Motivation

IP1d closed numerically under a 20x20 stationary-carrier phonon-recurrence guard and found a direction-specific structural precursor before persistent electronic nearest-neighbour relocations. The user also identified an older finite-size artifact: dynamic phonons emitted behind a moving carrier can traverse the periodic cell and later collide with the same carrier.

The present 20x20 production control uses gamma_v=0.01 fs^-1, for which the harmonic intermolecular modes are overdamped. This strongly suppresses coherent phonon memory, but gamma_v is only a numerical bath parameter, not a material-calibrated phonon lifetime. A mechanistic conclusion should therefore survive both a larger cell and a weaker intermolecular bath.

IP1e is a screening stage, not a kinetics stage. It asks whether the IP1d matched precursor/current conclusions remain qualitatively stable when the recurrence length is doubled and when the intermolecular modes are allowed to retain substantially more dynamical memory.

## Three protocols

All protocols remain zero field, 300 K, IDC-BM td=180 fs, dt=0.2 fs, CF4-Lanczos m=6, projected intermolecular zero modes, 2 fs diagnostics, and 50 fs electronic residence persistence. Both the isotropic J0y/J0x=1 condition and the established anisotropic J0y/J0x=0.15 mobile control are included.

### A. 20x20 baseline

- 20x20 PBC;
- gamma_u=0.01 fs^-1;
- gamma_v=0.01 fs^-1;
- 10 ps total, 2 ps burn-in;
- two independent seed pairs.

This reproduces a subset of IP1d with the same recurrence-guarded protocol.

### B. 40x40 finite-size control

- 40x40 PBC;
- gamma_u=0.01 fs^-1;
- gamma_v=0.01 fs^-1;
- 15 ps total, 5 ps burn-in;
- two independent seed pairs.

The harmonic stationary-carrier wrap time doubles from about 10.85 ps to about 21.69 ps. Thus the entire 15 ps run, including the +/-500 fs event windows, lies before the stationary wrap estimate.

### C. 40x40 weaker intermolecular damping

- 40x40 PBC;
- gamma_u=0.01 fs^-1;
- gamma_v=0.002 fs^-1;
- 15 ps total, 5 ps burn-in;
- two independent seed pairs.

Only the intermolecular damping is reduced, because the recurrence concern is specifically the propagating intermolecular lattice field. At gamma_v=0.002 fs^-1 the high-frequency intermolecular harmonic modes are underdamped, so this control retains substantially more phonon memory without simultaneously changing the intramolecular bath.

The longer 5 ps burn-in is used because the weaker v bath thermalizes more slowly. The existing temperature gate remains active; failure to reach the target stochastic temperature invalidates the weak-damping run numerically.

## Observables

IP1e reuses the validated IP1d event-conditioned matched-counterfactual diagnostics:

- complete persistent electronic NN events;
- matched continuous lattice advantage Delta Q_L,true - <Delta Q_L,counterfactual>;
- fraction of positive matched lattice advantage;
- TP1 current alignment on +/-100 fs;
- matched future-bond advantage and top-1 fraction at -100:-80 fs;
- long-lag matched future-bond advantage at -500:-400 fs;
- template signal strength and IDC association;
- numerical energy, norm and zero-mode gates.

No event-count comparison is promoted to a hopping-rate claim because N=2 seeds per protocol is only a sensitivity screen and because the physical dressed-event population is still being refined.

## Interpretation

The primary finite-size question is whether the 20x20 and 40x40 gamma_v=0.01 controls preserve the sign and order of magnitude of the matched lattice/bond/current diagnostics. Large qualitative changes would indicate that even the recurrence-guarded 20x20 mechanism is finite-size sensitive.

The primary bath-memory question is whether reducing gamma_v from 0.01 to 0.002 fs^-1 on 40x40 destroys, preserves or strengthens the direction-specific precursor. A large change is physically informative but must not be interpreted as a calibrated material effect because neither damping value is material fitted.

The weak-damping control is also the first protocol in this branch where propagating intermolecular phonon memory is intentionally allowed rather than strongly overdamped. If it remains numerically stable, a later stage can add explicit carrier-centered backward-wake observables and directly measure emitted vibrational energy and its propagation.

## Numerical gates

IP1e PASS requires:

1. pycompile and the complete regression suite pass;
2. the IP1p recurrence preflight passes for all three protocols;
3. every underlying IP1d dynamics run reports numerical PASS;
4. all requested trajectories complete;
5. temperatures satisfy the existing stochastic tolerance;
6. generalized energy-balance, norm, zero-mode and template-correlation gates remain satisfied.

No finite-size or damping agreement threshold is a numerical PASS gate. Those comparisons are physical results.

## Decision rule

Do not extract mobility, diffusion, activation energy, or a production hopping rate from IP1e. If the mechanism is qualitatively stable on 40x40 and under weaker intermolecular damping, proceed to explicit isolated-event/phonon-wake characterization before kinetics. If it is not stable, prioritize bath calibration, finite-size scaling and direct phonon propagation diagnostics.
