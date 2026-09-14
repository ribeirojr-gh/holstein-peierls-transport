# IP1j local validation — first natural field-driven hop wake

Date: 2026-09-14  
Branch: `isotropic-polaron-barrier`  
Validated commit: `e129217c53e0a0b6139dc4d19458d2971ee90896`  
User-local artifact: `ip1j-local-validation/20260914T172224Z`

## Numerical closure

IP1j completed all preregistered local gates successfully.

- overall runner status: **PASS**;
- failed gates: none;
- pycompile: PASS;
- focused IP1j/IP1h/IP1f/D3/TP1 tests: PASS;
- full test suite: **424/424 PASS**;
- 40x40 zero-damping phonon-recurrence preflight: PASS;
- both requested 40x40 field-driven trajectories: PASS;
- static convergence, size-aware field-work balance, electronic norm, projected zero modes, event detection and finite diagnostics: PASS;
- stored profile payloads for both conditions: PASS.

The validation used Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1 under WSL2 x86_64, with BLAS/OpenMP/MKL restricted to one thread.

The recurrence audit gives a stationary-carrier ballistic wrap scale of about 21.693 ps for the 40x40 control, while the production run ends at 5 ps. The event-conditioned windows therefore remain well before the stationary-carrier wrap diagnostic.

## Protocol

The deterministic D3 field-driven calculation uses:

- cell: 40x40 PBC;
- `J0y/J0x = 1.0` and `0.15`;
- field: +10 mV/A along x;
- T = 0 K;
- no thermostat and no IDC;
- `dt = 0.2 fs`;
- total duration: 5 ps;
- harmonic intermolecular current sampled every 2 fs;
- persistent electronic residence criterion: 50 fs;
- first persistent NN x relocation as the wake event;
- pre-event baseline: -500 to -100 fs;
- post-event window: at most 1500 fs and truncated before a second persistent x event;
- event coordinates rotated so the carrier direction is +s.

The harmonic maximum intermolecular group-velocity scale for this control is `1.84391 sites/ps`. A packet is preregistered as validated when the d=1 -> d=2 correlation is at least 0.80 and the inferred packet speed does not exceed `1.05*vmax`.

## Natural carrier events

The field-driven carrier moves toward -x under the established electron-like sign convention.

### Isotropic control, J0y/J0x = 1.0

- TP1 displacement: `-3.86383 A`;
- first persistent event: 820 -> 819, -x;
- event start: `2826 fs`;
- one persistent x event in 5 ps;
- maximum field-work residual: `2.225e-6 eV`, versus a size-aware tolerance of `2.0e-4 eV`;
- maximum electronic norm error: `2.61e-12`.

### Anisotropic control, J0y/J0x = 0.15

- TP1 displacement: `-7.15776 A`;
- first persistent event: 820 -> 819, -x at `2496 fs`;
- second persistent event: 819 -> 818, -x at `4108 fs`;
- maximum field-work residual: `3.959e-6 eV`, versus a size-aware tolerance of `2.0e-4 eV`;
- maximum electronic norm error: `5.31e-12`.

## Preregistered wake result

| J0y/J0x | backward lag [fs] | backward speed [sites/ps] | backward corr. | backward packet | forward lag [fs] | forward speed [sites/ps] | forward corr. | forward packet |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1.00 | 512 | 1.9531 | 0.999862 | no | 854 | 1.1710 | 0.998758 | yes |
| 0.15 | 584 | 1.7123 | 0.998419 | **yes** | 1000 | 1.0000 | 0.869327 | yes by preregistered gate |

The anisotropic first natural hop therefore contains a preregistered, physically admissible backward-propagating intermolecular energy-current branch. Since +s is defined as the carrier direction, this directly establishes for this event

`v_packet . v_carrier < 0`.

This is an existence statement for a retrograde lattice-radiation component. It is not a statement that the retrograde component dominates the emitted energy.

## Energy directionality

At the d=2 boundary the integrated positive outward energies are:

### Isotropic

- backward: `1.74799e-4 eV`;
- forward: `1.11838e-2 eV`;
- backward fraction: `0.01539`;
- `D2 = -0.96922`.

### Anisotropic

- backward: `7.01227e-4 eV`;
- forward: `7.18224e-2 eV`;
- backward fraction: `0.00967`;
- `D2 = -0.98066`.

Thus the natural first-hop radiation at d=2 is strongly **forward dominated** in both controls. The validated anisotropic retrograde branch carries only about 1% of the positive outward d=2 energy over the complete 1.5 ps window.

This differs importantly from the controlled IP1g/IP1h relocation impulse, where the anisotropic long-range packet became strongly retrograde at d=4-6. The controlled quench therefore cannot be transferred quantitatively to the natural field-driven event.

## Post-upload sensitivity audit

The stored IP1j profiles were independently reanalyzed after the preregistered run without changing the acceptance result.

### Anisotropic backward branch is robust

The backward d1->d2 correlation maximum remains at `584 fs` with correlation `0.998419` when the lag-search interval is changed among 100-1400, 300-1000, 300-1400, 450-900 and 500-800 fs. The peak is internal to all of those intervals. The inferred speed therefore remains `1.7123 sites/ps`, below the harmonic maximum.

This makes the anisotropic retrograde existence result substantially more robust than a single-window fit.

### Anisotropic forward speed is not well localized

The preregistered forward maximum occurs at the 1000 fs upper bound. When the range is extended to 1400 fs the optimum moves to the new upper boundary with still higher correlation. Therefore the statement `forward_packet_validated=true` is retained as the preregistered binary result, but `1.0 sites/ps` should **not** be interpreted as a well-determined forward packet speed.

### Isotropic backward result is threshold-sensitive

The unconstrained optimum is at `512 fs`, corresponding to `1.9531 sites/ps`, just above the preregistered `1.05*vmax` ceiling. At the first sampled lag that satisfies the speed ceiling, `518 fs`, the correlation remains `0.999858`, only about `4e-6` below the unconstrained maximum. Wider lag windows also reveal alternative very-high-correlation delays.

Therefore the preregistered classification `backward_packet_validated=false` must not be rephrased as evidence that an isotropic backward branch is absent. It is best classified as **ambiguous under the present delay estimator**.

## Scientific closure

IP1j is closed for its stated scope.

The validated conclusions are:

1. a self-consistent natural field-driven anisotropic polaron relocation emits a measurable retrograde intermolecular lattice-energy-current branch;
2. that branch propagates from d=1 to d=2 with a robust correlation delay of about 584 fs, or 1.712 sites/ps;
3. the retrograde branch is energetically minor at d=2; the natural event is strongly forward dominated there;
4. the isotropic backward classification is numerically borderline and cannot support an absence claim;
5. the forward anisotropic delay is search-boundary limited and its speed is not quantitatively resolved;
6. IP1j does not establish mobility, hopping rate, activation energy, field threshold, material phonon lifetime, or a unique normal-mode group velocity.

The next checkpoint should test whether the weak retrograde component is genuinely event-associated and reproducible rather than a small residual of continuous field-driven lattice oscillation. That requires amplitude/background significance and replication across natural events or matched no-hop windows before any transport-mechanism conclusion is strengthened.
