# IP1o — frozen-electronic-surface causal control of isotropic lattice memory

## Motivation

IP1n is numerically closed and demonstrates that the phase-structured trailing lattice-current pattern remains strong for 2 ps after electrical power is removed continuously. However, the fully coupled released branch develops two additional persistent carrier relocations after the switch (`819 -> 820` and `820 -> 819`) before the late-memory interval. Those electronic rearrangements can refresh the lattice current.

IP1o asks the stricter causal question:

> Is the post-hop current pattern already encoded in the lattice phase-space state at the release time, such that it persists even when subsequent electronic redistribution is forbidden?

## Branch point

Repeat the IP1n isotropic protocol until the first natural persistent x hop is accepted:

- 40x40 PBC;
- `J0y/J0x = 1.0`;
- initial field +10 mV/A along x;
- T=0;
- no thermostat;
- no IDC;
- dt=0.2 fs;
- current sample every 2 fs;
- energy sample every 10 fs;
- persistence criterion 50 fs;
- CF4-Lanczos, Krylov dimension 6.

The expected event is `820 -> 819`, transition start near 2826 fs and acceptance near 2874 fs, but the event must be detected rather than hard-coded.

At the exact accepted time `ts`, retain the Peierls phase continuously and set its time derivative to zero, exactly as in IP1n.

## Two zero-power continuations

Clone the branch-point state into:

1. **fully coupled released branch** — identical to the IP1n held-phase continuation; electronic and lattice degrees of freedom continue to evolve self-consistently;
2. **frozen-electronic-surface branch** — freeze the complete complex electronic wavefunction at its switch value and propagate only the classical lattice with velocity Verlet on the static Ehrenfest potential surface defined by that fixed wavefunction and held Peierls phase.

The frozen branch has:

- zero external electrical power because the held phase has zero rate;
- no electronic redistribution or hopping by construction;
- no force jump at the switch, because the fixed electronic state and held phase are exactly the state and phase used by the fully coupled force at `ts`;
- a conservative time-independent classical potential surface
  `V_lattice(q) + <psi_ts|H(q,A_ts)|psi_ts>`.

Thus energy conservation on this branch directly audits the implementation.

## PBC interpretation

The held vector potential corresponds to a static boundary twist on the finite periodic torus. IP1o therefore does not claim equivalence to a zero-phase PBC Hamiltonian. The relevant causal removal is **time-dependent electrical work and electronic redistribution**, not the static switch-time boundary twist.

## Observable and background

Use exactly the IP1n post-hop frame and pre-hop field background:

- fixed B -> C frame after event A -> B;
- trailing distances d=1,2,3,4;
- pre-hop B-centered reference from transition start minus 700 fs to transition start minus 100 fs;
- subtract the mean reference longitudinal profile.

Define

`A_trail(t) = sqrt(mean_d J_back(d,t)^2)` for d=1..4.

Construct the same 100 fs pre-hop background windows sampled every 20 fs. A post-switch 100 fs bin is pattern-separated when both RMS and peak amplitude are at or above the empirical 95th-percentile background.

## Frozen-surface memory gate

The frozen-electronic-surface branch supports lattice-encoded memory when:

1. at least three pattern-separated bins have centers more than 1000 fs after switch;
2. at least one passing center is >= `ts + 1500 fs`;
3. the electronic state remains bitwise unchanged from its switch value;
4. external work is exactly zero;
5. the frozen-surface total energy is conserved within the established 40x40 extensive tolerance;
6. the initial classical force equals the held-phase fully coupled force at the switch within numerical precision.

This is the primary IP1o physical decision.

## Comparison with fully coupled release

Report:

- fully coupled released sustained-pattern gate;
- frozen-surface sustained-pattern gate;
- late frozen/full RMS-amplitude ratio;
- time-resolved cosine similarity of the four-component trailing-current vectors;
- d=1..4 positive/net energies and positive-time fractions;
- any fully coupled post-switch persistent electronic events (diagnostic only).

Interpretation:

- **frozen PASS + full PASS**: the long-lived pattern is already stored in the lattice phase-space state and does not require subsequent electronic redistribution, although full backaction can modify it;
- **frozen FAIL + full PASS**: the zero-power IP1n pattern requires continuing electronic backaction/relocations and should not be labeled a purely lattice-stored wake;
- **both FAIL**: the IP1n signal is not reproduced under the stricter audit.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1o/IP1n/IP1m/IP1l/D3 tests pass;
3. full pytest passes;
4. recurrence preflight passes;
5. first persistent natural x event is found and used as branch point;
6. held phase matches the driven phase exactly;
7. frozen and fully coupled branch states are bitwise-identical at creation;
8. initial lattice force is continuous between driven and frozen-surface descriptions;
9. fixed electronic wavefunction remains bitwise unchanged in the frozen branch;
10. frozen external work is exactly zero;
11. frozen-surface energy conservation passes;
12. full released matter-energy conservation passes;
13. projected lattice zero modes remain controlled;
14. all pattern metrics are finite;
15. current trajectories are serialized.

Physical memory gates are not numerical acceptance gates.

## Interpretation guards

- The frozen-electronic branch is a causal numerical control, not a literal material protocol.
- A PASS demonstrates memory encoded in lattice positions/velocities evolving on the switch-time conservative electronic surface; it does not prove free-phonon propagation after complete electron-lattice decoupling.
- The held boundary twist remains part of the conservative surface.
- Alternating signs across d=1..4 remain a phase-structured pattern, not a fictitious monotonic retrograde flux.
- No mobility, hopping rate, activation energy, threshold field or calibrated phonon lifetime is inferred.

## Decision after IP1o

If the frozen branch passes, proceed to mode-resolved decomposition of the lattice-encoded response (wave number, frequency, polarization and intra/intermolecular partition) using a continuation that prevents event contamination.

If the frozen branch fails while the full released branch passes, first characterize the electronic-backaction cycle responsible for regenerating the zero-power lattice pattern before attempting a phonon-mode interpretation.