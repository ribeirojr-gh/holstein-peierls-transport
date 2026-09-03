# D1: Frozen-lattice electric-field dynamics

## Scope

D1 validates explicitly time-dependent electronic dynamics under a uniform electric field while the classical lattice is frozen. It extends the D0a/D0b foundations without yet introducing lattice motion, a thermostat, decoherence, GPU execution, or material-specific transport claims.

The archived dynamics applies a uniform electric field through a Peierls phase. In the present implementation the forward-bond phase along each lattice axis is

\[
\phi_\alpha(t) = -\frac{a_\alpha E_\alpha (t-t_0)}{\hbar},
\]

with energies in eV, time in fs, distances in angstrom, and electric field in V/angstrom. The reverse hopping is the complex conjugate, so the Hamiltonian remains Hermitian.

The default benchmark field, 2 mV/angstrom, is a legacy-scale numerical control. It must not be interpreted as a pentacene material prediction.

## D1a: linear one-particle propagation

The frozen-lattice TDSE is

\[
\frac{d|\psi\rangle}{dt} = -\frac{i}{\hbar}H(t)|\psi\rangle.
\]

The following methods are retained:

- **DOP853**: adaptive high-accuracy reference;
- **RK4**: transparent fixed-step baseline;
- **CF4-Lanczos**: two-exponential fourth-order commutator-free Magnus propagation at the two Gauss-Legendre nodes, using the validated D0a Lanczos exponential action.

Validation gates include exact reduction to the static Hamiltonian for zero field and at the field origin time, Hermiticity, field reversal, dense/sparse agreement, exact constant-H reduction to the D0a CF4 limit, fourth-order convergence on a noncommuting time-dependent control, norm preservation for CF4, and agreement with tightened DOP853 on a Holstein-Peierls control.

## D1b: spin-adapted projector propagation

The D0b variational projector equations are extended by replacing the fixed one-body matrix by the field-dependent matrix `T(t)`. The fourth-order RKMK scheme evaluates the one-body Hamiltonian at the correct RK stage times while preserving the common-unitary projector manifold.

Validation includes:

- recovery of the linear TDSE projector evolution in the closed, noninteracting limit;
- fourth-order convergence in time;
- singlet and triplet 4x4 S0 controls under the same electric field;
- preservation of Hermiticity, idempotency, mutual shell orthogonality, and particle number to floating-point accuracy.

## Reproducible benchmark

The benchmark was run on the GitHub-hosted Ubuntu 24.04 runner with Python 3.12.14, NumPy 2.5.2, and SciPy 1.18.1. BLAS/OpenMP thread counts were fixed to one. Timings are therefore hardware-dependent and are used only for relative algorithmic comparison.

### Linear 20x20 relaxed polaron

The lattice was first relaxed with the validated sparse static-polaron solver. Static preparation is excluded from the propagation timings. The frozen lattice was then propagated for 20 fs under a 2 mV/angstrom field along x.

| method | elapsed [ms] | H evaluations | H applications | phase-aligned error | norm error |
|---|---:|---:|---:|---:|---:|
| DOP853, rtol=1e-8 | 536.052 | 1214 | 1214 | 2.708e-13 | 2.698e-13 |
| RK4, dt=0.2 fs | 171.779 | 400 | 400 | 4.852e-05 | 4.848e-05 |
| CF4-Lanczos m=6, dt=0.2 fs | 194.717 | 200 | 1200 | 5.532e-09 | 1.321e-14 |
| CF4-Lanczos m=8, dt=0.2 fs | 217.943 | 200 | 1600 | 5.532e-09 | 1.099e-14 |
| RK4, dt=0.1 fs | 342.170 | 800 | 800 | 1.525e-06 | 1.521e-06 |
| CF4-Lanczos m=6, dt=0.1 fs | 383.733 | 400 | 2400 | 3.452e-10 | 3.941e-14 |
| CF4-Lanczos m=8, dt=0.1 fs | 433.948 | 400 | 3200 | 3.452e-10 | 5.551e-16 |

The equal state errors for `m=6` and `m=8` show that the Krylov truncation is already negligible at `m=6`; the fourth-order time discretization dominates. Increasing the Krylov dimension therefore adds cost without improving the propagated state for this control.

At the same 0.2 fs step, CF4-Lanczos reduces the state error by almost four orders of magnitude relative to RK4 while preserving the norm to roundoff. This is the principal D1a structure-preserving candidate.

### Spin-adapted 4x4 controls

The S0 singlet and triplet controls were optimized at zero field and then propagated for 1 fs under the same 2 mV/angstrom field. No artificial orbital kick was applied; the field itself initiates the dynamics.

| multiplicity | method | elapsed [ms] | RHS evaluations | projector error | RDM error | idempotency | orthogonality |
|---|---|---:|---:|---:|---:|---:|---:|
| singlet | DOP853, rtol=1e-8 | 237.045 | 302 | 7.227e-16 | 6.358e-16 | 9.721e-16 | 2.421e-16 |
| singlet | RKMK4, dt=0.04 fs | 79.896 | 100 | 2.276e-09 | 1.722e-09 | 2.298e-15 | 1.414e-15 |
| singlet | RKMK4, dt=0.02 fs | 155.207 | 200 | 1.423e-10 | 1.076e-10 | 2.673e-15 | 1.143e-15 |
| singlet | RKMK4, dt=0.01 fs | 310.767 | 400 | 8.891e-12 | 6.728e-12 | 3.631e-15 | 9.127e-16 |
| triplet | DOP853, rtol=1e-8 | 187.798 | 302 | 5.820e-16 | 5.219e-16 | 7.454e-16 | 2.423e-16 |
| triplet | RKMK4, dt=0.04 fs | 62.331 | 100 | 1.840e-09 | 1.615e-09 | 2.286e-15 | 1.150e-15 |
| triplet | RKMK4, dt=0.02 fs | 125.897 | 200 | 1.150e-10 | 1.009e-10 | 3.071e-15 | 7.760e-16 |
| triplet | RKMK4, dt=0.01 fs | 252.346 | 400 | 7.190e-12 | 6.310e-12 | 3.300e-15 | 2.224e-15 |

The error decreases by approximately a factor of 16 when the time step is halved, consistent with fourth-order convergence. Projector constraints remain at roundoff throughout.

## D1 decision

D1 is considered validated when the final repository CI remains green after removal of the temporary benchmark workflow.

The numerical hierarchy carried forward is:

1. **DOP853** remains the adaptive high-accuracy reference for explicitly time-dependent controls.
2. **CF4-Lanczos with Krylov dimension 6** is the preferred linear structure-preserving candidate for field-dependent one-particle propagation. `m=8` is retained only as a precision cross-check, not as the default.
3. **RKMK4** remains the preferred spin-adapted projector integrator because it combines fourth-order accuracy with exact common-unitary manifold preservation.
4. **RK4** remains a transparent baseline but is not promoted to the production path because norm and state errors are substantially larger at comparable step sizes.

These decisions are numerical-method decisions only. The transition to D2 must independently validate the coupled electron-lattice integrator and total-energy conservation at zero temperature and zero field.