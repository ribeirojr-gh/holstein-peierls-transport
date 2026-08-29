# Modernization Plan

## 1. Legacy workflow map

The archived workflow is composed of six source files:

- `rprop.f90` — static charge/lattice optimization and polaron formation.
- `hp2D.f90` — time-dependent electronic and lattice dynamics.
- `velocities.f90` — periodic polaron-position reconstruction, mean velocity, and inverse participation ratio analysis.
- `graphics.f90` — legacy Gnuplot-based post-processing and figure generation.
- `scritp.sh` — parameter generation, electric-field sweep, compilation, and execution orchestration.
- `mkl_vsl.f90` — Intel MKL VSL interface used by the legacy stochastic dynamics.

The Google Drive `legacy/` archive is the authoritative immutable copy of these files.

## 2. Input dependency graph

```text
scritp.sh
  |
  +--> parameters1.inc --> rprop.f90
  |                         |
  |                         +--> outU.dat
  |                         +--> outVX.dat
  |                         +--> outVY.dat
  |
  +--> parameters.inc -----> hp2D.f90
                              |
                              +--> time-dependent charge/lattice data
                                      |
                                      +--> graphics.f90
                                      +--> velocities.f90
```

The legacy field sweep contains ten field values: 0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, and 3.8, inserted in `parameters.inc` as negative values in units of `d-3`.

## 3. Scientific decomposition

### Phase I — static polaron formation

The first implementation target reproduces the RPROP optimization of the two-dimensional periodic Holstein–Peierls lattice. It must contain independent modules for:

- lattice topology and periodic neighbours;
- model parameters and units;
- electronic Hamiltonian construction;
- ground-state eigensolution;
- charge-density observables;
- lattice potential energy;
- analytic energy gradients;
- RPROP state and update rules;
- convergence criteria;
- checkpoint/restart and structured output.

### Phase II — dynamics

The dynamics implementation is deferred until the static solver passes regression tests. The archived `hp2D.f90` shows that this phase will require:

- a complex Hermitian Hamiltonian with electric-field phase factors;
- propagation of the electronic wavefunction in the instantaneous eigenbasis;
- classical propagation of `u`, `vx`, and `vy`;
- optional damping and stochastic forces;
- deterministic random-number seeding for reproducibility;
- trajectory output and transport observables;
- periodic-position unwrapping for polaron velocity.

## 4. Reference implementation strategy

### Milestone A — faithful dense CPU implementation

Use NumPy/SciPy in double precision and reproduce the legacy algorithm before changing its mathematics. Keep a documented `legacy` RPROP behavior where necessary.

### Milestone B — numerical validation

Validation must include:

1. Hamiltonian matrix equality for identical lattice configurations.
2. Ground-state eigenvalue/eigenvector agreement.
3. Analytic-gradient checks by finite differences.
4. Iteration-level comparison of RPROP updates.
5. Charge normalization and periodic-boundary tests.
6. Comparison of converged `u`, `vx`, `vy`, charge density, total energy, and polaron formation energy.
7. Reproduction of published static-polaron benchmarks where parameter sets match the literature.

### Milestone C — optimized CPU implementation

Only after Milestone B:

- eliminate the full density matrix where only local and nearest-neighbour elements are required;
- avoid full-spectrum diagonalization when only the electronic ground state is required;
- introduce sparse/operator formulations;
- benchmark BLAS threading and parameter-level parallelism;
- avoid nested thread oversubscription.

### Milestone D — GPU backend

Implement an optional CuPy/CuPyX backend behind the same solver API and compare every GPU result against the validated CPU reference within explicit numerical tolerances.

## 5. Reproducibility requirements

Every modern run should record:

- complete parameter set and units;
- Git commit SHA and package version;
- numerical backend and precision;
- Python, NumPy/SciPy/CuPy versions;
- CPU/GPU model and thread configuration;
- convergence history;
- initial-condition provenance;
- random seed when stochastic terms are enabled;
- output checksum where publication-grade datasets are produced.

## 6. Repository policy

- New code lives in `src/holstein_peierls/`.
- Legacy source files remain in the Drive archive and are not modified.
- Tests are mandatory for physics kernels before performance refactoring.
- Scientific corrections are separate commits from performance changes.
- Publication calculations use tagged releases.
