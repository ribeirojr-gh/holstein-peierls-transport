# D0a frozen-H propagator sweep results

## Status

D0a is complete.  The linear frozen-H benchmark now has both analytic/unit-test
controls and a size/time-step/Krylov-dimension accuracy-cost sweep on the actual
Holstein-Peierls one-particle Hamiltonian.

The sweep was run on the GitHub-hosted Ubuntu runner with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` so the
comparison measures algorithmic behavior rather than uncontrolled library
threading.  These timings are therefore a single-thread reference, not a final
CPU/GPU performance claim.

## Sweep protocol

The total propagated physical time was fixed at `2.0 fs`.  The tested cases
were:

- `4 x 4`, `8 x 8`, and `12 x 12` lattices at `dt = 0.05, 0.10, 0.20 fs`;
- the legacy-size `20 x 20` lattice (`N=400`) at `dt = 0.10 fs`;
- Lanczos/Krylov dimensions `m = 4, 6, 8, 12`;
- the legacy repeated full spectral exponential, RK4, Fehlberg RKF7(8)
  propagating with its eighth-order member, Crank-Nicolson, short Lanczos, and
  the frozen-H limit of the two-exponential fourth-order commutator-free Magnus
  scheme.

Every size/time-step case used one exact spectral exponential over the full
`2.0 fs` interval as the final-state reference.  Candidate states were never
renormalized automatically.

## Representative `dt = 0.10 fs` results

The phase-aligned state error is the primary accuracy discriminator in this
frozen-H test because norm and energy conservation alone can hide phase errors.

| lattice | method | ms / step | work / step | phase-aligned state error |
|---|---:|---:|---:|---:|
| `4 x 4` | legacy spectral | 0.126 | 1 eigendecomposition | `1.66e-14` |
| | RK4 | 0.082 | 4 Hpsi | `3.37e-9` |
| | RKF7(8) order 8 | 0.440 | 13 Hpsi | `1.07e-15` |
| | Lanczos `m=6` | 0.296 | 6 Hpsi | `8.88e-13` |
| | Lanczos `m=8` | 0.396 | 8 Hpsi | `8.49e-15` |
| | frozen CFM4 `m=6` | 0.603 | 12 Hpsi | `2.99e-14` |
| | Crank-Nicolson | 0.279 | 1 Hpsi + 1 solve | `2.61e-5` |
| `8 x 8` | legacy spectral | 0.690 | 1 eigendecomposition | `2.58e-14` |
| | RK4 | 0.091 | 4 Hpsi | `3.56e-9` |
| | RKF7(8) order 8 | 0.452 | 13 Hpsi | `1.60e-15` |
| | Lanczos `m=6` | 0.311 | 6 Hpsi | `1.05e-12` |
| | Lanczos `m=8` | 0.407 | 8 Hpsi | `6.82e-15` |
| | frozen CFM4 `m=6` | 0.620 | 12 Hpsi | `3.41e-14` |
| | Crank-Nicolson | 0.349 | 1 Hpsi + 1 solve | `2.85e-5` |
| `12 x 12` | legacy spectral | 3.720 | 1 eigendecomposition | `2.78e-14` |
| | RK4 | 0.097 | 4 Hpsi | `3.65e-9` |
| | RKF7(8) order 8 | 0.481 | 13 Hpsi | `1.90e-15` |
| | Lanczos `m=6` | 0.337 | 6 Hpsi | `1.71e-12` |
| | Lanczos `m=8` | 0.436 | 8 Hpsi | `2.58e-15` |
| | frozen CFM4 `m=6` | 0.669 | 12 Hpsi | `5.44e-14` |
| | Crank-Nicolson | 0.528 | 1 Hpsi + 1 solve | `2.67e-5` |
| `20 x 20` | legacy spectral | 61.085 | 1 eigendecomposition | `4.52e-14` |
| | RK4 | 0.110 | 4 Hpsi | `3.15e-9` |
| | RKF7(8) order 8 | 0.545 | 13 Hpsi | `2.61e-15` |
| | Lanczos `m=6` | 0.375 | 6 Hpsi | `1.45e-12` |
| | Lanczos `m=8` | 0.491 | 8 Hpsi | `7.04e-15` |
| | frozen CFM4 `m=6` | 0.761 | 12 Hpsi | `4.60e-14` |
| | Crank-Nicolson | 1.256 | 1 Hpsi + 1 solve | `2.36e-5` |

For the `20 x 20` reference case, the single-thread speedups relative to the
archived repeated full diagonalization are approximately:

- RK4: `556 x`;
- RKF7(8): `112 x`;
- Lanczos `m=6`: `163 x`;
- Lanczos `m=8`: `124 x`;
- frozen CFM4 `m=6`: `80 x`;
- Crank-Nicolson: `49 x`.

The absolute values are runner-specific; the scaling separation is the important
result.

## Time-step and Krylov-dimension trends

The sweep gives a clear hierarchy.

### Lanczos

`m=6` is already a strong accuracy-cost point over the tested interval:

- at `dt = 0.05 fs`, phase-aligned errors are roughly `3e-14` to `5e-14`;
- at `dt = 0.10 fs`, they are roughly `9e-13` to `2e-12`;
- at `dt = 0.20 fs`, they remain roughly `3e-11` to `6e-11`.

`m=8` reaches near-machine-precision final-state accuracy throughout the tested
range, with errors of order `1e-15` to `1e-14`.

Increasing to `m=12` provides no systematic accuracy benefit in this problem
because the `m=8` result is already at floating-point accuracy; it only increases
cost and reorthogonalization work.

### RKF7(8)

The eighth-order Fehlberg member remains at approximately machine precision for
all tested time steps and sizes.  It is therefore retained as the independent
high-accuracy explicit-RK reference.  Its fixed cost is 13 Hamiltonian actions
per step, so Lanczos `m=8` is both cheaper and comparably accurate for the frozen
linear problem.

### RK4

RK4 is the cheapest tested method per step.  Its final-state error shows the
expected strong time-step dependence: approximately `2e-10` at `0.05 fs`,
`3e-9` at `0.10 fs`, and `5e-8` at `0.20 fs`.  It remains useful as the simple
low-cost baseline, but it is not the preferred high-accuracy frozen-H method.

### Frozen-H CFM4 limit

For a constant Hamiltonian the two exponentials reduce to two half-step
exponentials of the same generator.  The measured cost is therefore almost
exactly twice the corresponding Lanczos step and it does not lie on the D0a
Pareto front.  This is expected and is not a reason to reject commutator-free
Magnus for explicitly time-dependent dynamics: its actual purpose is tested in
D1, where the two Gauss-node Hamiltonians differ.

### Crank-Nicolson

Crank-Nicolson preserves norm and energy essentially to floating-point accuracy,
but its phase-aligned state error is approximately `3e-5` at `dt=0.10 fs` and
scales poorly enough with time step to be noncompetitive here.  This directly
confirms that norm or energy conservation cannot be used as the sole propagator
selection criterion.

## D0a decision

D0a promotes the following methods to the next deterministic stages:

1. **short Hermitian Lanczos/Krylov exponential action** as the primary linear
   propagation candidate; `m=6` is the economical accuracy point and `m=8` the
   near-machine-precision point for the tested one-particle spectrum;
2. **RKF7(8), eighth-order member** as an independent high-accuracy explicit-RK
   control;
3. **RK4** as a transparent low-cost baseline;
4. **two-exponential fourth-order commutator-free Magnus + Krylov** for D1,
   because only explicitly time-dependent Hamiltonians can reveal its intended
   advantage.

The repeated full eigendecomposition is retained only as a small-system/reference
oracle.  Crank-Nicolson is not promoted as a primary candidate.

This is not yet a final production-propagator decision.  A frozen Hamiltonian
cannot test interpolation of a time-dependent Peierls phase, coupled
lattice-electronic energy conservation, or nonlinear spin-adapted orbital
propagation.  The operational sequence therefore remains

`D0a -> D0b -> D1 -> D2 -> ...`.

## Reproducibility

The complete raw sweep is produced by
`experiments/d0a_propagator_sweep.py`.  The temporary CI workflow used only to
obtain the hosted-runner measurement is intentionally not retained after the
results are recorded; normal regression CI remains lightweight.
