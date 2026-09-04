# D6b — coupled zero-temperature bipolaron and exciton dynamics

## Scope

D6b is the first moving-lattice validation of the two pair sectors.  It extends the D2 deterministic velocity-Verlet/CF4-Lanczos architecture to

1. the correlated symmetric-spatial singlet bipolaron; and
2. the distinguishable electron-hole exciton.

D6b deliberately excludes a lattice bath, electronic decoherence, electric field, mobility inference, and material-specific calibration.

## Electronic representation

The electronic state is the ordered product-basis amplitude matrix `Psi[i,j]`, flattened in C order for propagation.  The Hilbert-space dimension is `N^2`.

The D6a complex-safe matrix-free action is retained.  No dense `N^2 x N^2` Hamiltonian is formed in D6b.

For the bipolaron the physical singlet spatial sector satisfies

`Psi[i,j] = Psi[j,i]`.

For the distinguishable electron-hole model no exchange symmetry is imposed.  This remains the spin-blind direct-interaction reference model and must not be relabelled as a physical singlet/triplet exciton.

## Propagated-state reduced density matrices

For normalized pair states:

- bipolaron spin-summed one-body RDM: `gamma = 2 Psi Psi^dagger`, with `Tr gamma = 2`;
- electron RDM: `gamma_e = Psi Psi^dagger`, with `Tr gamma_e = 1`;
- hole RDM: `gamma_h = Psi^T Psi*`, with `Tr gamma_h = 1`.

All pair Ehrenfest forces are evaluated from these propagated complex RDMs, not from an instantaneous adiabatic ground state.

## Ehrenfest gradient

The current pair interactions are lattice-coordinate independent.  Onsite/short-range/long-range pair interactions therefore enter `<Psi|H_pair|Psi>` but have no explicit derivative with respect to `u`, `vx` or `vy`.  They affect the classical force indirectly through the propagated RDM.

The Holstein components use the RDM diagonal.  The Peierls components use the real parts of nearest-neighbour coherences.  This is the complex-state extension of the already validated static Hellmann-Feynman gradients.

D6b explicitly validates these expressions against central finite differences of the propagated-state energy.

## Classical masses

The existing pair parameter classes are static-model objects and do not silently inherit a mass.  D6b introduces `PairLatticeMasses`, expressed in `eV fs^2 / A^2`, as an explicit dynamics input.

The numerical validation control converts the archived one-polaron masses

- `m_u = 7.5e10 eV as^2/A^2` -> `7.5e4 eV fs^2/A^2`;
- `m_v = 1.5e11 eV as^2/A^2` -> `1.5e5 eV fs^2/A^2`.

These are legacy/control values, not newly calibrated bipolaron or exciton material parameters.

## Split algorithm

One D6b step is

1. pair Ehrenfest force at `(q_n, Psi_n)`;
2. velocity half kick;
3. full lattice drift to `q_{n+1}`;
4. electronic propagation across the linearly interpolated path `q(t)`;
5. pair Ehrenfest force at `(q_{n+1}, Psi_{n+1})`;
6. second velocity half kick.

The electronic step uses the same symmetric fourth-order two-exponential commutator-free Magnus formula as D1/D2, but each weighted pair Hamiltonian is assembled as a **matrix-free pair action**.  The lattice-independent interaction matrix is cached by a moving-Hamiltonian factory.

D6a showed that `m=8` is sufficient for the frozen 3x3 control.  D6b re-tests `m=6,8,12`; `m=8` is not promoted until the moving-lattice result is inspected.

## Independent reference

A full-system adaptive DOP853 integrator evolves the pair amplitudes, three lattice coordinates per site, and three lattice velocities per site in one ODE system.  It is used only on small controls because the pair Hilbert space grows as `N^2`.

## Pre-registered D6b gates

### Unit/definition gates

1. complex bipolaron force agrees with finite differences;
2. complex exciton force agrees with finite differences;
3. bipolaron norm, exchange symmetry and RDM trace are preserved;
4. exciton norm and both RDM traces are preserved;
5. short split trajectory agrees with tightened DOP853;
6. short zero-temperature propagated-state total energy is stable;
7. legacy mass conversion is exact;
8. all D6a frozen-H tests remain green;
9. full pytest remains green.

### 4x4 numerical benchmark

The reference window is 4 fs.  The split is compared at `dt = 0.2, 0.1, 0.05 fs`; Krylov dimensions `m = 6, 8, 12` are compared at `dt = 0.2 fs`.

A separate 100 fs stability run uses `dt = 0.2 fs`, `m=8`.

The benchmark requires, independently for bipolaron and exciton:

- the `dt=0.1 fs` combined electronic/lattice reference error to be at most 60% of the `dt=0.2 fs` error;
- `m=8` phase-aligned electronic error below `1e-5` over the short reference window;
- maximum sampled total-energy drift below `1e-4 eV` over 100 fs;
- maximum electronic norm error below `1e-10`;
- exchange/RDM-trace constraint error below `1e-10`;
- finite lattice evolution with maximum coordinate excursion below `0.2 A`.

These are numerical closure controls, not material-accuracy tolerances.

## After D6b

Only if these gates close should D6 proceed to longer/scaled pair trajectories and then decide how to introduce finite temperature, fields and any pair-sector decoherence model.  The one-carrier IDC-BM model from D5 must **not** be copied blindly to an `N^2` pair sector: its equilibrium meaning, adiabatic basis cost and physical collapse channels require a separate validation.
