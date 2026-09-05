# Transport production roadmap

## Purpose

The validated dynamics roadmap D0--D6 ends with zero-field finite-temperature
one-carrier and pair-sector dynamics, including explicit lattice heat and
stochastic electronic-environment exchange.  The next phase is therefore not a
new generic dynamics benchmark.  It is the controlled construction of a
production transport workflow.

The central unresolved identity is the driven finite-temperature balance

\[
\Delta E_{\mathrm{matter}}
\simeq Q_{\mathrm{lattice}} + Q_{\mathrm{electronic}} + W_{\mathrm{field}},
\]

with all three exchanges accumulated explicitly from the already validated D3,
D4, and D5 components.

No mobility is reported until this balance, the transport observables, and the
ensemble/statistical protocol have been validated independently.

## Naming

This phase uses `TP` checkpoints rather than continuing the historical D0--D6
sequence.  The name describes the new goal: transport production rather than
another isolated dynamics component.

## TP0 -- driven finite-temperature IDC energy closure

Combine, for the one-polaron sector only:

- the D3 uniform-field Peierls phase and field-power work accounting;
- the D4 BAOAB lattice bath with caller-owned persistent RNG;
- the D5 reference IDC-BM electronic correction;
- projected intermolecular zero modes; and
- CF4--Lanczos matrix-free electronic propagation.

At every IDC event the collapse basis must be the **instantaneous field-dependent
Hamiltonian at the event time**.  Reusing the zero-field D5 eigensystem under a
finite field is forbidden.

The event changes electronic energy at fixed lattice coordinates, lattice
velocities, field, and time.  Its exact jump is accumulated as
`Q_electronic`.  The BAOAB O step contributes `Q_lattice`; trapezoidal
integration of `<psi|partial H/partial t|psi>` contributes `W_field`.

The TP0 closure gate is numerical and energetic.  It is not a mobility gate and
it does not claim a steady state.

### TP0 reference control

Use a small `4x4` periodic lattice so the combined bookkeeping can be audited
cheaply before any long production run:

- `T = 300 K`;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- `dt = 0.2 fs`;
- `E_x = 2 mV/angstrom`, `E_y = 0` as a numerical field control;
- IDC-BM with `t_d = 180 fs`, the validated one-polaron D5 reference interval;
- projected intermolecular zero modes;
- CF4--Lanczos with Krylov dimension 6, as validated for the one-particle path;
- four independent lattice/decoherence RNG pairs.

The field magnitude and IDC interval remain numerical controls, not material
predictions.

### TP0 pre-registered checks

For the small control, require:

1. field-aware IDC at zero field reproduces the D5 zero-field event
   probabilities/energies and the selected state up to a global phase;
2. the recorded IDC energy jump equals the direct before/after expectation
   value of the instantaneous field Hamiltonian;
3. a zero-field driven-IDC run has zero accumulated field work and reduces to
   the already validated zero-field energy balance within numerical tolerance;
4. finite-field trajectories satisfy
   `Delta E_matter - Q_lattice - Q_electronic - W_field` with maximum sampled
   absolute residual below `1e-4 eV`;
5. mean post-burn-in lattice temperature lies in `240--360 K`;
6. electronic norm error remains below `1e-10`;
7. projected zero-mode means remain below `1e-12`; and
8. the lattice remains bounded under the small numerical control.

No post-step wavefunction normalization or energy repair is allowed.

## TP1 -- transport observables

Only after TP0 closes, define and cross-check the observables needed for
transport:

- bond current from the derivative of the Hamiltonian with respect to Peierls
  phase / vector potential;
- charge polarization or center on a periodic lattice using a representation
  that does not jump at cell boundaries;
- unwrapped displacement where mathematically justified;
- mean-squared displacement for zero-field diffusion controls; and
- consistency between integrated current and polarization change.

The legacy site-index center is not sufficient by itself under periodic
boundaries.

## TP2 -- ensemble linear-response mobility

After TP1:

- use independent trajectory ensembles;
- sweep positive and negative weak fields;
- identify a field range where drift is linear in field;
- separate transient and analysis windows before fitting drift velocity;
- report uncertainty across trajectories; and
- cross-check drift and Einstein/diffusive information where applicable.

A single finite-field trajectory must never be called a mobility calculation.

## TP3 -- pair-sector driven transport

Only after the one-polaron TP0--TP2 workflow closes should field-driven
finite-temperature transport be extended to the D6 pair sectors.  The D6
reference decoherence choices remain sector dependent:

- singlet bipolaron: IDC-BM;
- distinguishable electron-hole exciton: IDC-DP.

Their field coupling and observables must be derived for the corresponding
many-body Hamiltonians rather than copied from the one-particle formula.

## TP4 -- profiling and CPU/GPU acceleration

Performance work begins after the production observable path is fixed.  Profile
real TP2 workloads first, then optimize the measured hot kernels.  Likely
candidates are repeated sparse/matrix-free Hamiltonian actions, CF4--Lanczos
basis construction, force evaluation, and trajectory-level ensemble
parallelism.

CPU threading and GPU support must preserve a validated single-thread CPU
reference.  No GPU backend is promoted solely because it is available.

## Scope boundary

TP0--TP4 are numerical/scientific validation checkpoints.  Material-specific
mobilities require an explicit parameter layer, convergence with lattice size
and trajectory length, calibrated or defensible bath/decoherence parameters,
and uncertainty analysis.  The generic controls in this repository are not
pentacene predictions unless that material layer is supplied and validated.
