# D0a frozen-H propagator benchmark

## Purpose

D0a is the first dynamics checkpoint after S0 and O0. It deliberately removes
all classical and explicitly time-dependent complications and tests only

`d psi / dt = -(i / hbar) H psi`

for one fixed Hermitian one-particle Holstein-Peierls Hamiltonian.

The objective is not to choose a production dynamics method from formal order
alone. D0a creates a common numerical interface and measures accuracy at a fixed
computational cost before D1 introduces an explicitly time-dependent field and
before D2 couples the electronic state back to the lattice.

No thermostat, electric field, lattice velocity, stochastic force, pair
Hamiltonian, MCTDHF orbital equation, GPU backend, or material fit is introduced
in D0a.

## Units and exact reference

The stationary code uses energies in eV. D0a expresses time in femtoseconds and
uses

`hbar = 0.6582119569 eV fs`.

For a frozen Hamiltonian the exact reference is

`psi(t + dt) = exp(-i H dt / hbar) psi(t)`.

`exact_spectral_step()` evaluates this expression through a complete Hermitian
eigendecomposition. A fresh eigendecomposition is performed on every call so a
repeated benchmark represents the archived `ZHEEV`-per-step algorithmic cost,
even though a frozen control could trivially cache the eigensystem.

The final accuracy reference for a trajectory of `n` equal steps is generated
separately from one exact spectral propagation over the total time `n dt`. This
reference-generation timing is reported but is not mixed with candidate timing.

## Candidate propagators

### Legacy full spectral exponential

This is the mathematical operation performed by the archived electronic update.
It is exact for frozen `H` up to eigensolver/floating-point error, but scales as
`O(N^3)` and stores a complete eigensystem. It is an accuracy and historical
cost reference, not a presumptive production choice.

### Classical RK4

RK4 is retained as the simplest matrix-vector baseline. One step requires four
applications of `H psi`. It is not exactly unitary and D0a never renormalizes the
state after a step; norm drift must remain visible in the benchmark.

### Fehlberg RKF 7(8)

D0a implements the classical 13-stage Fehlberg embedded 7(8) tableau and uses
the eighth-order member as the propagated state. The seventh/eighth difference
is retained as a local embedded-error diagnostic, but the D0a trajectory uses a
fixed step so wall-clock and `H psi` counts remain directly comparable.

The tableau is the classical Fehlberg 1968 7(8) method. This is distinct from
the adaptive Dormand-Prince eighth-order reference reserved for D0b, where the
spin-adapted orbital equations are nonlinear and state dependent.

### Hermitian Lanczos/Krylov exponential action

A short Lanczos basis approximates the exponential action without a full
Hamiltonian eigensystem. The implementation uses full reorthogonalization of the
small Krylov basis for reproducibility. A Krylov breakdown is treated as an
exact invariant-subspace termination rather than as a failure.

The dominant workspace scales approximately as `O(N m)` for Krylov dimension
`m`, and one step uses at most `m` applications of `H psi`.

### Fourth-order commutator-free Magnus: frozen limit

A standard two-exponential fourth-order commutator-free Magnus scheme samples
the time-dependent Hamiltonian at two Gauss nodes. For constant `H`, both
weighted generators reduce to `H/2`, all commutators vanish, and the method is
exactly the product of two half-step exponentials.

D0a therefore benchmarks `cfm4_frozen_limit_step()` as two half-step Lanczos
exponential actions. This is the exact frozen-H reduction and exposes its
relative two-exponential cost. The actual Gauss-node time-dependent CF4 formula
is intentionally deferred to D1, where it can be meaningfully distinguished
from a single exponential.

### Crank-Nicolson / Cayley control

Crank-Nicolson solves

`(I + i H dt / 2 hbar) psi_(n+1) = (I - i H dt / 2 hbar) psi_n`.

For Hermitian frozen `H` it is norm preserving without manual renormalization.
It provides an independent implicit/unitary control at the cost of one sparse
linear solve per step. D0a does not assume that this solve is cheap when `H`
changes later.

## Common metrics

Every candidate is propagated from the same complex non-eigenstate without
renormalization. The benchmark records:

- wavefunction norm error;
- raw state-vector error;
- state-vector error after removing one global phase;
- fidelity `|<psi_ref|psi>|^2` after norm normalization;
- electronic energy expectation `<psi|H|psi>/<psi|psi>`;
- energy-expectation error;
- wall-clock time and time per step;
- number of explicit `H psi` applications;
- number of complete eigendecompositions;
- number of linear solves;
- an approximate algorithmic workspace estimate;
- RKF7(8) embedded error estimate; and
- actual Lanczos dimension used.

The corrected energy definition is the propagated-state expectation value, not
the archived instantaneous-ground-state `evec(:,1)` diagnostic.

## Reproducible Holstein-Peierls control

`experiments/d0a_frozen_propagators.py` constructs a smooth deterministic
periodic lattice distortion and the historical one-polaron energy/coupling
scales. The initial electronic state is a normalized localized complex
wavepacket with a phase texture and is deliberately not an eigenstate.

The synthetic distortion is only a numerical control. It is not a relaxed
polaron, a finite-temperature snapshot, or a pentacene parameterization.

Example:

```bash
python experiments/d0a_frozen_propagators.py \
  --nx 8 --ny 8 \
  --dt-fs 0.1 \
  --steps 100 \
  --krylov-dimension 12 \
  --output benchmark/d0a-frozen.json
```

The pull-request CI runs a smaller `4x4`, 20-step smoke benchmark and uploads the
JSON result as an artifact. Unit tests separately validate exact two-level
controls, action counts, phase invariance, norm preservation, high-order RKF
accuracy, and exact full-dimension Lanczos reduction.

## D0a decision rule

D0a must not select a production propagator from one small smoke benchmark.
Before promotion, run a size/step/Krylov sweep including at least the historical
`dt = 0.1 fs` and larger trial steps, and compare accuracy at fixed wall-clock
cost. The most promising method must then survive D1 explicit-time-dependence
and D2 zero-temperature coupled energy-conservation gates.

In particular:

- a unitary method can still be too inaccurate in phase or too expensive;
- a high-order RK method can still lose norm or be inefficient at the target
  error;
- a small Krylov space can preserve norm while accumulating projection error;
- the CF4 frozen result cannot demonstrate fourth-order time-ordering accuracy,
  because all commutators vanish when `H` is constant; and
- no D0a result tests finite-temperature or nonadiabatic mean-field physics.

The next checkpoint after the full D0a size/accuracy sweep is D0b, the separate
benchmark of nonlinear/state-dependent spin-adapted orbital propagation.
