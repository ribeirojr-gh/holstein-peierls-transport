# Legacy dynamics audit — `hp2D.f90`

## Purpose

This document records the behavior of the archived two-dimensional Holstein-Peierls dynamics before any modern propagator is implemented. The goal is to preserve the original physics as a reference while separating necessary corrections from optional algorithmic improvements.

The audited archived source is `hp2D.f90`, used together with the generated `parameters.inc`, the stationary lattice output, Intel MKL eigensolvers, and MKL VSL random-number routines.

No production dynamics algorithm is selected by this audit.

## Reference run encoded by the archived driver

The archived shell driver generates a representative `20 x 20` periodic lattice (`N=400`) with

- `J0x = 0.100 eV`;
- `J0y = 0.015 eV`;
- `alpha_1 = 3.0 eV/angstrom`;
- `alpha_2x = alpha_2y = 0.4 eV/angstrom`;
- `K1 = 16.51 eV/angstrom^2`;
- `K2 = 0.51 eV/angstrom^2`;
- `dt = 100 as = 0.1 fs`;
- `tfinal = 10,000,000 as = 10 ps`; and
- four MKL threads for the dense diagonalization.

A `10 ps` trajectory at `0.1 fs` therefore contains `100,000` full electronic propagation steps.

The archived example has `Temp=0`, all damping constants zero, and random forces disabled. Temperature handling nevertheless remains present in the source and is audited below.

## Electronic propagation in the legacy code

For each full electronic step the code:

1. constructs the complete dense complex Hermitian `N x N` Hamiltonian;
2. copies it to `evec`;
3. calls LAPACK/MKL `ZHEEV('V','U',...)`, obtaining **all** eigenvalues and eigenvectors;
4. projects the current wavefunction into the instantaneous eigenbasis;
5. multiplies each coefficient by `exp(-i E_j dt / hbar)`; and
6. transforms back to the site basis.

For a Hamiltonian frozen over one time step this is the exact matrix-exponential action

`psi(t+dt) = exp[-i H(t) dt/hbar] psi(t)`.

The archived comments call this the "Ono's method" update.

The important distinction for modernization is that **full diagonalization is a means of evaluating the exponential action, not a physical requirement of the Ehrenfest forces**.

## Dominant avoidable computational costs

### 1. Full dense diagonalization every full step

`ZHEEV` scales cubically with the number of sites and stores all eigenvectors. The physical Hamiltonian, however, contains only onsite and nearest-neighbour terms. A matrix-vector action `H psi` is therefore sparse/matrix-free and scales linearly with the number of sites for the one-particle model.

For the `20 x 20` archived reference, the full eigensystem is recomputed roughly `100,000` times in a `10 ps` trajectory.

### 2. A second explicit `O(N^3)` eigenvector-overlap calculation

Immediately after diagonalization, the archived code computes

`COEF(i,j) = sum_k conj(evec_new(k,j)) evec_old(k,i)`

with three explicit nested loops. It then constructs instantaneous level occupations `n_occ` from this overlap matrix.

The output statements for `n_occ` are commented out in the archived source, yet the full `O(N^3)` overlap calculation still executes at every full electronic step.

This work is not required for propagating `psi` or evaluating the Ehrenfest force. In a modern reference implementation it should therefore be absent by default and enabled only as an explicitly requested diagnostic, preferably at a reduced sampling frequency.

### 3. Full `N x N` density matrix

The archived routine constructs

`rho(i,j) = psi(i) conj(psi(j))`

for all pairs of sites. The force expressions only require local populations and nearest-neighbour coherences. For the one-particle model these can be evaluated directly from `psi`, reducing this part from `O(N^2)` storage/work to `O(N)`.

This is directly analogous to the optimization already validated for the stationary one-polaron solver.

## Important diagnostic inconsistency

The archived `energies` routine evaluates the electronic contribution as

`sum_ij conj(evec(i,1)) H(i,j) evec(j,1)`.

During a nonadiabatic trajectory, however, the propagated electronic state is `psi`, not necessarily the instantaneous ground-state eigenvector `evec(:,1)`.

Consequently the archived electronic/total-energy diagnostic does **not** generally represent the energy of the propagated Ehrenfest state.

The modern dynamics must use

`E_el(t) = <psi(t)|H(t)|psi(t)>`

for one-particle propagation, with the analogous pair-state expectation value for bipolaron/exciton dynamics.

This correction is required independently of which propagator is ultimately selected.

## Classical lattice stepping

The ionic sector is a stochastic velocity-Verlet/BBK-like split update:

- one half-kick;
- coordinate drift;
- force reevaluation after the electronic propagation; and
- the second half-kick.

Friction and Gaussian random forces are included in the velocity updates.

The original framework therefore implements a mixed quantum-classical Ehrenfest-type dynamics: the quantum state is propagated explicitly, while the lattice coordinates are classical and respond to expectation-value forces.

## Temperature and fluctuation-dissipation implementation

The archived source generates three independent Gaussian random-force arrays for `u`, `vx`, and `vy`. Their amplitudes are proportional to

`sqrt(2 k_B T gamma / dt)`,

which is the expected classical white-noise fluctuation-dissipation scaling for a Markovian Langevin bath.

The conceptual choice — thermostat the lattice and allow electron-phonon coupling to transmit thermal fluctuations to the electronic subsystem — is physically meaningful for a semiclassical low-density carrier model. It should not be discarded simply because the implementation is old.

Several implementation details do require validation before finite-temperature production use:

1. the random-number stream is recreated and deleted for each force array and each half-step, with seeds derived directly from the integer simulation time;
2. independent random forces are generated on the two half-step calls within a full step; whether this is statistically consistent depends on the intended discrete Langevin scheme;
3. the kick uses damping parameters `dc1`/`dc2`, while the force routine also contains damping parameters `dc3`/`dc4`; any non-identical use must be checked for fluctuation-dissipation consistency;
4. several intermolecular damping terms use `vxdt(i)`/`vydt(i)` inside loops whose physical site index is `k`; this is a likely indexing issue if `dc4` is nonzero and must be tested against the intended equations before correction; and
5. the active instantaneous-temperature diagnostic is `T = 3 KE/(N k_B)`. For three classical coordinates per site, equipartition instead gives `T = 2 KE/(3 N k_B)`. The latter expression is present in the source but commented out. The active diagnostic is therefore larger by a factor of `4.5` and must not be used as a reference finite-temperature observable.

These issues are latent in the archived zero-temperature example because all damping constants are zero and random forces are disabled.

## Why RK4/RK8 are useful benchmark candidates

Direct explicit Runge-Kutta propagation of

`d psi / dt = -(i/hbar) H(t) psi`

requires only repeated matrix-vector actions. With the nearest-neighbour Hamiltonian implemented matrix-free, each stage is `O(N)` for the one-particle model and avoids the full eigensystem.

RK4 is therefore an important simple baseline. A high-order RK method (including an eighth-order reference implementation) is useful for testing whether fewer/larger time steps compensate for the greater number of stages.

However, ordinary explicit Runge-Kutta schemes are not exactly unitary. Norm, phase, and energy errors must be measured rather than hidden by routine renormalization. A production decision cannot be based on wall-clock time alone.

## Other propagation families that should be benchmarked

### Krylov/Lanczos exponential action

For a Hermitian sparse or matrix-free Hamiltonian, a short Lanczos basis can approximate

`exp(-i H dt/hbar) psi`

directly. This retains the exponential/unitary character of the legacy propagation while avoiding a complete diagonalization. It is especially attractive because the bipolaron and exciton Hamiltonian actions are already matrix-free.

### Commutator-free Magnus + Krylov exponential action

The Hamiltonian changes during the trajectory because of both lattice motion and the electric-field phase. Higher-order commutator-free Magnus methods evaluate weighted Hamiltonians at intermediate times and retain an exponential form without constructing explicit commutators. They are natural candidates when explicit time dependence becomes important.

### Chebyshev polynomial propagation

Chebyshev expansion can provide highly accurate exponential action using only repeated `H psi` operations when useful spectral bounds are available. It should be considered as an accuracy reference, although its relative efficiency for a Hamiltonian changing every short time step must be benchmarked.

### Crank-Nicolson / implicit midpoint

A Cayley step is norm preserving for Hermitian Hamiltonians but requires a linear solve at each step. Because the Hamiltonian is sparse and changes continuously, its cost should be measured rather than assumed.

### Local bond splitting

The Holstein-Peierls Hamiltonian is a sum of onsite and nearest-neighbour bond terms. Even/odd or graph-coloured bond splittings could exponentiate small local blocks and are naturally parallel/GPU-friendly. This option has greater implementation complexity and should only be pursued if the simpler Krylov/Magnus approaches do not already remove the bottleneck sufficiently.

## Modern numerical literature relevant to propagator choice

Representative method benchmarks include:

- A. Alvermann and H. Fehske, *J. Comput. Phys.* **230**, 5930-5956 (2011), "High-order commutator-free exponential time-propagation of driven quantum systems", DOI `10.1016/j.jcp.2011.04.006`. The paper develops unitary commutator-free propagators up to eighth order and discusses Krylov evaluation for large sparse problems.
- A. Gomez Pueyo, M. A. L. Marques, A. Rubio, and A. Castro, *J. Chem. Theory Comput.* **14**, 3040-3052 (2018), "Propagators for the Time-Dependent Kohn-Sham Equations: Multistep, Runge-Kutta, Exponential Runge-Kutta, and Commutator Free Magnus Methods", DOI `10.1021/acs.jctc.8b00197`. Their cost-versus-accuracy study found a simplified fourth-order commutator-free Magnus scheme particularly robust and efficient for the tested time-dependent electronic-structure problems.

These papers do not prove that the same method will be optimal for the present Holstein-Peierls Hamiltonian. They justify including exponential/Magnus approaches in the benchmark rather than comparing only Runge-Kutta orders.

## Modern finite-temperature/nonadiabatic context

The archived ionic Langevin idea remains a defensible baseline, but two conceptually separate questions should not be conflated:

1. **How should the classical lattice bath be integrated?** Modern Langevin splittings such as BAOAB can substantially improve configurational sampling at a given time step. See B. Leimkuhler and C. Matthews, *J. Chem. Phys.* **138**, 174102 (2013), DOI `10.1063/1.4802990`, and subsequent thermostat-splitting literature.
2. **Is mean-field Ehrenfest dynamics itself sufficient for electronic thermalization/decoherence?** Modern organic-semiconductor literature shows that this can be regime dependent. Ehrenfest can give useful mobilities in some regimes, but it need not preserve the correct equilibrium electronic distribution. Examples include W. Si and C.-Q. Wu, *J. Chem. Phys.* **143**, 024103 (2015), DOI `10.1063/1.4926534`; S. Giannini and J. Blumberger, *Acc. Chem. Res.* **55**, 819-830 (2022), DOI `10.1021/acs.accounts.1c00675`; and J. E. Runeson, T. J. G. Drayton, and D. E. Manolopoulos, *J. Chem. Phys.* **161**, 144102 (2024), DOI `10.1063/5.0226001`.

Therefore a better Langevin integrator does not by itself cure possible Ehrenfest overcoherence or incorrect electronic equilibrium. Those are separate physical approximations and should be tested separately.

## Proposed validation sequence

Before finite temperature, fields, or pair dynamics are modernized, use the following deterministic hierarchy.

### D0 — frozen Hamiltonian

For a small lattice, compare every candidate propagator against the legacy full diagonalization/exponential reference for a fixed Hermitian `H`.

Measure:

- wavefunction norm error;
- state fidelity/phase-aligned state error;
- `|<psi_ref|psi>|^2`;
- electronic-energy expectation error;
- wall-clock time;
- number of `H psi` evaluations; and
- memory usage.

### D1 — explicitly time-dependent field, frozen lattice

Keep lattice coordinates fixed and apply the Peierls electric-field phase. Compare against a highly converged reference with progressively reduced time step.

### D2 — coupled zero-temperature electron-lattice dynamics, zero field

This is the most important conservation test. Use the corrected `<psi|H|psi>` electronic energy and require controlled drift of total energy and electronic norm.

### D3 — field-driven zero-temperature dynamics

Compare charge-center displacement, IPR/localization, current/drift observables, and energy delivered by the field.

### D4 — finite-temperature lattice bath

Only after deterministic dynamics is validated should stochastic thermostat variants be compared. The first comparison should keep the same classical Langevin/FDT physics while changing only the numerical integrator (legacy BBK-like versus BAOAB/GJF-type schemes).

### D5 — electronic thermalization/decoherence model

Only after D4 should the project decide whether pure Ehrenfest is adequate for the target regime or whether a controlled extension such as decoherence-corrected dynamics, surface hopping/mapping approaches, generalized open-system dynamics, or quantum-phonon methods is scientifically required.

### D6 — bipolaron and exciton dynamics

Once the one-particle algorithm is selected, port the validated matrix-free propagator interface to the correlated two-particle and distinguishable electron-hole sectors. A method based only on matrix-vector actions is strongly preferred because complete diagonalization scales prohibitively in the pair Hilbert spaces.

## Benchmark candidates for the first deterministic comparison

The initial benchmark should include at least:

1. legacy full diagonalization + frozen-step exponential — accuracy reference only;
2. classical RK4;
3. a well-defined eighth-order explicit RK implementation;
4. short-Lanczos/Krylov exponential action;
5. fourth-order commutator-free Magnus with Krylov exponential action; and
6. optionally Chebyshev or Crank-Nicolson as an independent unitary/high-accuracy control.

No method should be promoted because of formal order alone. The selection metric is **accuracy at fixed wall-clock cost for the actual Holstein-Peierls trajectory**, including norm, phase/fidelity, forces, energy conservation, transport observables, and scalability.

## Architectural consequence

The new dynamics layer should expose a propagator interface independent of the lattice integrator and thermostat. This allows electronic propagation, classical integration, temperature control, and diagnostics to be changed and benchmarked independently.

This separation is essential for later CPU threading and GPU acceleration: the dominant primitive should become a batched/matrix-free Hamiltonian action rather than a dense eigendecomposition.
