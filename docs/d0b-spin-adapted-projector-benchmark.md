# D0b spin-adapted frozen-geometry projector benchmark

## Why D0b is separate from D0a

D0a solved a linear problem,

`i hbar d psi / dt = H psi`,

with a fixed Hermitian Hamiltonian.  The spin-adapted open-shell backend is
different: every shell sees a Fock matrix that depends on the instantaneous
shell projectors.  Its electronic equation is therefore nonlinear and
state-dependent.  The D0a observation that short Lanczos is excellent for a
linear exponential action cannot be promoted automatically to this problem.

Miranda, Fisher, Stella and Horsfield derived a fixed-coefficient MCTDHF
working equation for general open-shell states (J. Chem. Phys. 134, 244101,
2011, DOI `10.1063/1.3600397`) and used the resulting nonadiabatic framework for
conjugated-polymer dynamics (J. Chem. Phys. 134, 244102, 2011, DOI
`10.1063/1.3600404`).  Later implementations of the same polymer framework also
state explicitly that the coupled electronic equations are integrated with an
eighth-order Runge-Kutta method with step-size control (for example Chinese
Physics B 26, 107103, 2017, DOI `10.1088/1674-1056/26/10/107103`).

D0b therefore uses an adaptive eighth-order explicit integrator as the numerical
reference rather than treating the state-dependent Fock operator as a frozen
matrix exponential.

## Projector formulation

The S0 static implementation already expresses the open-shell energy through
occupied-shell projectors `P_mu`, occupations `n_mu`, and shell Fock matrices
`F_mu`.  Its orbital-rotation derivative is

`M = sum_mu n_mu [P_mu, F_mu]`.

For real-time propagation we use the same restricted orbital manifold.  Add the
virtual complement

`Q = I - sum_mu P_mu`

as an occupation-zero subspace.  The off-diagonal blocks of one common
anti-Hermitian generator are

`K_ab = P_a M P_b / (i hbar (n_a - n_b))`, for `n_a != n_b`.

All occupied shells then evolve with

`dP_mu/dt = [K, P_mu]`.

A single common generator is essential.  Independently evolving shell `mu` with
`F_mu` would not, in general, preserve orthogonality between different occupied
shells.

Rotations between equal-occupation subspaces are gauge-fixed to zero.  This is
also the explicit tangent restriction already used by the S0 optimizer.  In the
minimal open-shell singlet this means that direct rotations between the two
separate singly occupied frontier shells are not introduced in D0b.  D0b is
therefore the real-time extension of the **currently validated S0 fixed-
coefficient manifold**; it is not a claim that every possible configuration-
amplitude degree of freedom of a larger MCTDHF expansion has already been
implemented.

## Why projectors are propagated in the benchmark

Individual orbitals contain arbitrary phases and, inside redundant subspaces,
arbitrary unitary gauges.  Projectors are the natural gauge-invariant variables
for checking the electronic state.  They also expose the exact structural
invariants directly:

- `P_mu = P_mu^dagger`;
- `P_mu^2 = P_mu`;
- `P_mu P_nu = 0` for `mu != nu`;
- `Tr(gamma) = sum_mu n_mu rank(P_mu)`;
- `gamma = sum_mu n_mu P_mu`.

The comparison metric is therefore a normalized Frobenius distance between
shell projectors, supplemented by the spin-summed RDM distance.

## Numerical references and candidates

### Tight adaptive DOP853

SciPy's DOP853 implementation is used as the modern high-accuracy reference.
It is an explicit eighth-order Dormand-Prince-family integrator with adaptive
step-size control.  We do **not** claim that its tableau is identical to every
historical eighth-order RK implementation used in the polymer literature; the
role here is a reproducible, independently tightened reference solution for the
same nonlinear RHS.

### Fixed-step RK4

RK4 is retained as the transparent baseline.  D0b requires explicit
step-halving convergence against the adaptive reference and reports any drift
of projector constraints rather than hiding it through post-step
orthonormalization.

### Predictor exponential midpoint control

A structure-preserving nonlinear control evaluates the variational generator at
the initial state, takes a unitary half-step to predict the midpoint, evaluates
a second generator there, and exponentiates it for the full step.  Because each
update is a common unitary similarity transformation, Hermiticity,
idempotency, shell orthogonality and particle number are preserved to roundoff.

This method is included to test whether a self-consistent exponential strategy
is attractive for the nonlinear problem.  It is not assumed to inherit the D0a
Lanczos result, and it is not yet a production propagator.

## Validation gates

The D0b foundation is accepted only if all of the following hold:

1. the dynamic energy functional exactly matches the validated S0 static
   functional on a valid projector state;
2. the variational generator is anti-Hermitian and its RHS is tangent to
   `P^2=P`;
3. the closed-shell, noninteracting limit reproduces the exact TDSE projector
   evolution;
4. fixed RK4 exhibits fourth-order step-halving convergence in that analytic
   limit and converges to a tightened DOP853 reference in an interacting
   nonlinear control;
5. the variational flow has zero first-order energy derivative at frozen
   geometry;
6. tight DOP853 conserves energy, particle number and projector constraints to
   the requested numerical tolerance;
7. the predictor exponential midpoint update preserves the projector manifold
   to floating-point accuracy; and
8. the equal-occupation singlet frontier-frontier generator block remains zero
   under the explicit S0 gauge restriction.

## What D0b does not yet include

- lattice motion or Ehrenfest forces;
- Peierls electric-field phases;
- finite-temperature baths;
- intersystem crossing or spin-orbit coupling;
- time-dependent configuration coefficients beyond the fixed-coefficient S0
  manifold;
- GPU or threaded performance tuning.

Those remain later checkpoints.  After D0b selects a numerically trustworthy
nonlinear electronic integration strategy, D1 can introduce explicit time
dependence from the electric field and D2 can couple the electronic state back
to the classical lattice.
