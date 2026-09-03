# D0b spin-adapted frozen-geometry projector benchmark

## Why D0b is separate from D0a

D0a solved a linear problem,

`i hbar d psi / dt = H psi`,

with a fixed Hermitian Hamiltonian. The spin-adapted open-shell backend is different: every shell sees a Fock matrix that depends on the instantaneous shell projectors. Its electronic equation is therefore nonlinear and state-dependent. The D0a observation that short Lanczos is excellent for a linear exponential action cannot be promoted automatically to this problem.

Miranda, Fisher, Stella and Horsfield derived a fixed-coefficient MCTDHF working equation for general open-shell states (J. Chem. Phys. 134, 244101, 2011, DOI `10.1063/1.3600397`) and used the resulting nonadiabatic framework for conjugated-polymer dynamics (J. Chem. Phys. 134, 244102, 2011, DOI `10.1063/1.3600404`). Later implementations of the same polymer framework also state explicitly that the coupled electronic equations are integrated with an eighth-order Runge-Kutta method with step-size control (for example Chinese Physics B 26, 107103, 2017, DOI `10.1088/1674-1056/26/10/107103`).

D0b therefore uses an adaptive eighth-order explicit integrator as the numerical reference rather than treating the state-dependent Fock operator as a frozen matrix exponential.

## Projector formulation

The S0 static implementation already expresses the open-shell energy through occupied-shell projectors `P_mu`, occupations `n_mu`, and shell Fock matrices `F_mu`. Its orbital-rotation derivative is

`M = sum_mu n_mu [P_mu, F_mu]`.

For real-time propagation we use the same restricted orbital manifold. Add the virtual complement

`Q = I - sum_mu P_mu`

as an occupation-zero subspace. The off-diagonal blocks of one common anti-Hermitian generator are

`K_ab = P_a M P_b / (i hbar (n_a - n_b))`, for `n_a != n_b`.

All occupied shells then evolve with

`dP_mu/dt = [K, P_mu]`.

A single common generator is essential. Independently evolving shell `mu` with `F_mu` would not, in general, preserve orthogonality between different occupied shells.

Rotations between equal-occupation subspaces are gauge-fixed to zero. This is also the explicit tangent restriction already used by the S0 optimizer. In the minimal open-shell singlet this means that direct rotations between the two separate singly occupied frontier shells are not introduced in D0b. D0b is therefore the real-time extension of the **currently validated S0 fixed-coefficient manifold**; it is not a claim that every possible configuration-amplitude degree of freedom of a larger MCTDHF expansion has already been implemented.

## Why projectors are propagated in the benchmark

Individual orbitals contain arbitrary phases and, inside redundant subspaces, arbitrary unitary gauges. Projectors are the natural gauge-invariant variables for checking the electronic state. They also expose the exact structural invariants directly:

- `P_mu = P_mu^dagger`;
- `P_mu^2 = P_mu`;
- `P_mu P_nu = 0` for `mu != nu`;
- `Tr(gamma) = sum_mu n_mu rank(P_mu)`;
- `gamma = sum_mu n_mu P_mu`.

The comparison metric is therefore a normalized Frobenius distance between shell projectors, supplemented by the spin-summed RDM distance.

## Numerical references and candidates

### Tight adaptive DOP853

SciPy's DOP853 implementation is the high-accuracy numerical reference. It is an explicit eighth-order Dormand-Prince-family integrator with adaptive step-size control. We do **not** claim that its tableau is identical to every historical eighth-order RK implementation used in the polymer literature; its role here is to provide a reproducible, independently tightened reference solution for the same nonlinear RHS.

### Fixed-step RK4

RK4 is retained as the transparent fourth-order baseline. D0b explicitly measures any drift of the projector constraints rather than hiding it through post-step orthonormalization.

### Predictor exponential midpoint

The predictor exponential midpoint control evaluates the variational generator at the initial state, takes a unitary half-step to predict the midpoint, evaluates a second generator there, and exponentiates it for the full step. Because each update is a common unitary similarity transformation, Hermiticity, idempotency, shell orthogonality and particle number are preserved to roundoff.

The benchmark confirms, however, that this simple construction is only second order. It is useful as a structure-preserving control but is not competitive with the fourth-order candidates at the accuracy targets relevant here.

### Fourth-order Runge-Kutta-Munthe-Kaas (RKMK4)

RKMK4 integrates the nonlinear projector flow on the anti-Hermitian Lie algebra rather than updating the projectors directly. The fourth-order Runge-Kutta stages use the inverse differential of the exponential map (`dexp^-1`) through the commutator terms needed at fourth order; the accumulated algebra element is then applied through one common unitary similarity transformation.

This provides the key combination sought in D0b:

- fourth-order convergence comparable to ordinary RK4;
- no post-step projector repair;
- projector idempotency and mutual shell orthogonality preserved to floating-point roundoff;
- cost comparable to RK4 for the validated 4x4 control.

## Validation gates

The D0b foundation is accepted only if all of the following hold:

1. the dynamic energy functional exactly matches the validated S0 static functional on a valid projector state;
2. the variational generator is anti-Hermitian and its RHS is tangent to `P^2=P`;
3. the closed-shell, noninteracting limit reproduces the exact TDSE projector evolution;
4. fixed RK4 exhibits fourth-order step-halving convergence in the analytic limit and converges to a tightened DOP853 reference in an interacting nonlinear control;
5. the variational flow has zero first-order energy derivative at frozen geometry;
6. tight DOP853 conserves energy, particle number and projector constraints to the requested numerical tolerance;
7. the predictor exponential midpoint update preserves the projector manifold to floating-point accuracy;
8. the equal-occupation singlet frontier-frontier generator block remains zero under the explicit S0 gauge restriction;
9. RKMK4 exhibits fourth-order convergence in both the analytic closed-shell limit and an interacting nonlinear control; and
10. RKMK4 preserves projector idempotency and shell orthogonality to roundoff without repair.

All D0b unit/integration gates pass together with the pre-existing D0a and six S0 regression jobs.

## Final 4x4 benchmark

The final benchmark uses the same validated S0 numerical control:

- isotropic `J1 = J2 = 0.100 eV`;
- isotropic `alpha1 = alpha2 = 3.0 eV/A`;
- checkerboard validation gap `2.0 eV`;
- density-density controls `U = 0.525 eV` and nearest-neighbour `V = 0.08 eV`;
- a small allowed occupied-virtual kick of `0.05 rad`;
- frozen geometry;
- total propagation time `0.4 fs`;
- one BLAS/OpenMP thread.

The timings below are runner-specific and should not be interpreted as hardware-independent performance claims.

| multiplicity | method | dt [fs] | projector error | RDM error | energy drift [eV] | max projector-constraint error | elapsed [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| singlet | RK4 | 0.04 | 3.584e-09 | 1.441e-09 | 6.573e-14 | 1.447e-12 | 20.89 |
| singlet | RKMK4 | 0.04 | 3.584e-09 | 1.441e-09 | 2.474e-12 | 1.278e-15 | 25.82 |
| singlet | RK4 | 0.01 | 1.400e-11 | 5.630e-12 | 1.776e-15 | 3.261e-15 | 88.29 |
| singlet | RKMK4 | 0.01 | 1.400e-11 | 5.630e-12 | 1.243e-14 | 3.020e-15 | 84.75 |
| triplet | RK4 | 0.04 | 1.916e-09 | 8.599e-10 | 5.329e-14 | 7.587e-13 | 15.22 |
| triplet | RKMK4 | 0.04 | 1.916e-09 | 8.600e-10 | 1.327e-12 | 1.713e-15 | 14.85 |
| triplet | RK4 | 0.01 | 7.484e-12 | 3.359e-12 | 0.000e+00 | 1.560e-15 | 58.66 |
| triplet | RKMK4 | 0.01 | 7.485e-12 | 3.360e-12 | 3.553e-15 | 4.378e-15 | 58.65 |

The moderate DOP853 control reaches projector errors of approximately `4e-16` (singlet) and `3e-16` (triplet), establishing a tight independent reference. The predictor exponential midpoint is approximately twice as fast as RK4 at the same time step but gives projector errors of order `1e-6` to `1e-7`, confirming the expected second-order behavior.

## D0b numerical decision

D0b closes with the following hierarchy:

1. **DOP853** remains the tightened adaptive reference for nonlinear frozen-geometry validation.
2. **RKMK4** is the preferred fixed-step, structure-preserving candidate to carry into coupled spin-adapted dynamics because it combines fourth-order accuracy with exact manifold preservation at essentially RK4-like cost in the validated control.
3. **RK4** remains an independent transparent fourth-order regression baseline.
4. **Predictor exponential midpoint** remains a useful second-order geometric control but is not promoted to the main dynamics path.

This decision applies to the **current fixed-coefficient S0 projector manifold**. It does not yet select the final integrator for a future enlarged MCTDHF ansatz with explicitly time-dependent configuration coefficients.

## What D0b does not yet include

- lattice motion or Ehrenfest forces;
- Peierls electric-field phases;
- finite-temperature baths;
- intersystem crossing or spin-orbit coupling;
- time-dependent configuration coefficients beyond the fixed-coefficient S0 manifold;
- GPU or threaded performance tuning.

Those remain later checkpoints. D1 can now introduce explicit time dependence from the electric field while retaining DOP853 as a reference and carrying RKMK4 as the primary structure-preserving fixed-step candidate. D2 can then couple the electronic state back to the classical lattice.