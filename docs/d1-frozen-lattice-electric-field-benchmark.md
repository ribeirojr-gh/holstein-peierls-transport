# D1 frozen-lattice electric-field benchmark

## Scope

D1 validates explicitly time-dependent electronic propagation under a uniform electric field while the classical lattice is frozen.  It has two deliberately separate sectors:

1. **D1a — linear one-polaron propagation:** the validated static 20x20 polaron is relaxed first, then its lattice is frozen and the electric field is switched on at `t=0`.
2. **D1b — spin-adapted projector propagation:** the validated 4x4 gapped S0 singlet and triplet controls are optimized at zero field, then propagated with the same time-dependent one-body field term.

The field scale `2 mV/angstrom` is retained as a legacy/numerical control.  It is not a pentacene material prediction.  All benchmark timings below use one CPU thread (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`).  Static preparation is excluded from the propagation timings.

## Field convention

The modern implementation preserves the sign convention audited from the archived dynamics.  For a uniform field component along a lattice axis,

`phi_axis(t) = -a_axis E_axis (t - t0) / hbar`,

and the reverse hopping uses the complex-conjugate phase.  This guarantees a Hermitian instantaneous Hamiltonian.  Zero field and `t=t0` reduce to the validated static Hamiltonian.

## D1a validation gates

The following gates pass:

- exact zero-field reduction to the static Hamiltonian;
- exact `t=t0` reduction;
- Hermiticity and `H(-E) = H(E)*`;
- dense/sparse agreement;
- correct frozen-H reduction of the two-exponential fourth-order commutator-free Magnus scheme;
- fourth-order global convergence for both RK4 and CF4 on a noncommuting time-dependent control;
- norm preservation of CF4 to floating-point accuracy; and
- agreement of the 4x4 Holstein-Peierls field control with a tightened DOP853 reference.

## D1a 20x20 polaron benchmark

The static polaron is relaxed with the sparse electronic solver and optimized gradient.  Its lattice is then frozen and a field of `2 mV/angstrom` is applied along x for `20 fs`.

| method | elapsed [ms] | H evaluations | H applications | phase-aligned error | norm error |
|---|---:|---:|---:|---:|---:|
| DOP853, `rtol=1e-8` | 536.052 | 1214 | 1214 | 2.708e-13 | 2.698e-13 |
| RK4, `dt=0.2 fs` | 171.779 | 400 | 400 | 4.852e-05 | 4.848e-05 |
| CF4-Lanczos, `m=6`, `dt=0.2 fs` | 194.717 | 200 | 1200 | 5.532e-09 | 1.321e-14 |
| CF4-Lanczos, `m=8`, `dt=0.2 fs` | 217.943 | 200 | 1600 | 5.532e-09 | 1.099e-14 |
| RK4, `dt=0.1 fs` | 342.170 | 800 | 800 | 1.525e-06 | 1.521e-06 |
| CF4-Lanczos, `m=6`, `dt=0.1 fs` | 383.733 | 400 | 2400 | 3.452e-10 | 3.941e-14 |
| CF4-Lanczos, `m=8`, `dt=0.1 fs` | 433.948 | 400 | 3200 | 3.452e-10 | 5.551e-16 |

### D1a numerical decision

The fourth-order commutator-free Magnus + Lanczos path is promoted as the **primary linear electronic candidate for D2**.

For this 20x20 control, `m=6` is already converged with respect to Krylov dimension: increasing to `m=8` produces no measurable improvement in the state error but increases both Hamiltonian applications and wall time.  At `dt=0.2 fs`, CF4-Lanczos is only about 13% slower than RK4 while reducing the phase-aligned state error by nearly four orders of magnitude and preserving the norm to roundoff.  It is also substantially faster than the adaptive DOP853 accuracy reference.

Therefore D2 should carry forward:

- **CF4-Lanczos (`m=6`)** as the primary linear production candidate;
- **RK4** as an independent simple baseline; and
- **DOP853** as the high-accuracy reference.

The `0.2 fs` step is a validated D1 starting point, not a universal production time step.  D2 must re-establish the acceptable step from coupled total-energy conservation and force accuracy once the lattice moves.

## D1b validation gates

The time-dependent spin-adapted extension applies the same `T(t)` one-body term at the correct stage times of the nonlinear orbital/projector equations.  The following gates pass:

- the time-dependent RKMK implementation reduces to the validated D0b result for a constant one-body Hamiltonian;
- the noninteracting closed-shell limit reproduces the corresponding linear TDSE projector evolution;
- the common-unitary update preserves Hermiticity, idempotency, mutual shell orthogonality and particle number to floating-point accuracy;
- global fourth-order convergence is recovered; and
- both the S0 singlet and high-spin triplet controls agree with tightened DOP853 references under the same `2 mV/angstrom` field.

## D1b 4x4 spin-adapted benchmark

| multiplicity | method | elapsed [ms] | RHS evals | projector error | RDM error | idempotency | orthogonality |
|---|---|---:|---:|---:|---:|---:|---:|
| singlet | DOP853, `rtol=1e-8` | 237.045 | 302 | 7.227e-16 | 6.358e-16 | 9.721e-16 | 2.421e-16 |
| singlet | RKMK4, `dt=0.04 fs` | 79.896 | 100 | 2.276e-09 | 1.722e-09 | 2.298e-15 | 1.414e-15 |
| singlet | RKMK4, `dt=0.02 fs` | 155.207 | 200 | 1.423e-10 | 1.076e-10 | 2.673e-15 | 1.143e-15 |
| singlet | RKMK4, `dt=0.01 fs` | 310.767 | 400 | 8.891e-12 | 6.728e-12 | 3.631e-15 | 9.127e-16 |
| triplet | DOP853, `rtol=1e-8` | 187.798 | 302 | 5.820e-16 | 5.219e-16 | 7.454e-16 | 2.423e-16 |
| triplet | RKMK4, `dt=0.04 fs` | 62.331 | 100 | 1.840e-09 | 1.615e-09 | 2.286e-15 | 1.150e-15 |
| triplet | RKMK4, `dt=0.02 fs` | 125.897 | 200 | 1.150e-10 | 1.009e-10 | 3.071e-15 | 7.760e-16 |
| triplet | RKMK4, `dt=0.01 fs` | 252.346 | 400 | 7.190e-12 | 6.310e-12 | 3.300e-15 | 2.224e-15 |

### D1b numerical decision

RKMK4 remains the **primary spin-adapted candidate for D2 and later coupled dynamics**.  At `dt=0.04 fs` it gives approximately `2e-9` projector/RDM errors while requiring about one third of the wall time of the moderate DOP853 reference, and it preserves the projector manifold without any post-step repair.

DOP853 remains the tightened nonlinear reference.  The D1 `0.04 fs` step is again only a starting point; the coupled D2 energy-conservation gate must determine the production step size.

## D1 conclusion

D1 closes the frozen-lattice electric-field checkpoint.  The field convention, linear time-dependent propagation and spin-adapted nonlinear propagation are now independently validated.

The next checkpoint is **D2 — coupled zero-temperature electron-lattice dynamics at zero electric field**.  Its central validation requirements are:

- use the corrected electronic energy expectation of the propagated state rather than an instantaneous ground-state diagnostic;
- couple electronic and lattice stages consistently in time;
- preserve electronic norm/projector constraints;
- measure total-energy drift at zero field and zero thermostat;
- compare candidate time steps and electronic propagators at fixed physical trajectory length; and
- keep lattice integration, electronic propagation and diagnostics as separable interfaces in preparation for later finite-temperature and GPU work.
